import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from prody import parsePDB
from scipy.spatial.transform import Rotation as R
from load_nmr import element_list_stripped
import random
from entmax import entmax_bisect_loss

import matplotlib.pyplot as plt
import tqdm

elem = [e.upper() for e in element_list_stripped]
PAD_ID = len(elem)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_SCHEDULER = False  # Set to True if you want to use a learning rate scheduler

class AtomEmbedding(nn.Module):
    def __init__(self, num_atom_types, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, emb_dim)
        self.w_r = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.w_theta = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.w_phi = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.p_r = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.p_theta = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.p_phi = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.vec_r = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.vec_theta = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.vec_phi = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.integ_dim = nn.Linear(emb_dim * 4, emb_dim)

    def forward(self, x, type):
        emb_type = self.embedding(type)
        r = torch.norm(x, dim=2, keepdim=True)
        theta = torch.acos(torch.clamp(x[:, :, 2:3] / (r + 1e-8), -1+1e-7, 1-1e-7))
        phi = torch.atan2(x[:, :, 1:2], x[:, :, 0:1])
        emb_r = torch.cos(self.w_r * r + self.p_r) * self.vec_r
        emb_theta = torch.cos(self.w_theta * theta + self.p_theta) * self.vec_theta
        emb_phi = torch.cos(self.w_phi * phi + self.p_phi) * self.vec_phi
        emb = torch.cat([emb_type, emb_r, emb_theta, emb_phi], dim=-1)
        return self.integ_dim(emb), (r, theta, phi)

class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.p = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.vec = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.integ = nn.Linear(emb_dim, emb_dim)

    def forward(self, positions):
        positions = positions.unsqueeze(2).float()
        emb = torch.cos(self.w * positions + self.p) * self.vec
        return self.integ(emb)

class SinusoidalFeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ff = nn.Linear(in_features, out_features)

    def forward(self, x):
        return x + F.gelu(self.ff(x))

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = SinusoidalFeedForward(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            pad_mask = pad_mask.to(x.device)
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x, key_padding_mask=pad_mask)
        attn_output = self.act(attn_output)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)
        x = self.ff(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        #self.cross_attention_xv = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention_ev = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_integ = SinusoidalFeedForward(embed_dim, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = SinusoidalFeedForward(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x, bottleneck, pad_mask=None, causal_mask=None):
        x = self.norm1(x)
        attn_output, _ = self.self_attention(x, x, x, attn_mask=causal_mask, key_padding_mask=pad_mask)
        attn_output = self.act(attn_output)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)
        #cross_x, _ = self.cross_attention_xv(x, bottleneck, x)
        cross_e, _ = self.cross_attention_ev(x, bottleneck, bottleneck)
        cross_output = self.cross_integ(cross_e)
        cross_output = self.act(cross_output)
        x = x + self.dropout(cross_output)
        x = self.norm3(x)
        x = self.ff(x)
        return x

class Encoder(nn.Module):
    def __init__(self, num_atom_types, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = AtomEmbedding(num_atom_types, embed_dim)
        self.pos_emb = PositionalEmbedding(embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, xc, xt, pad_mask=None):
        x, pos_backup = self.embedding(xc, xt)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), x.size(1))
        x = x + self.pos_emb(pos)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x, pos_backup

class Decoder(nn.Module):
    def __init__(self, num_atom_types, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = AtomEmbedding(num_atom_types, embed_dim)
        self.pos_emb = PositionalEmbedding(embed_dim)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, xc, xt, bottleneck, pad_mask=None, causal_mask=None):
        x, _ = self.embedding(xc, xt)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), x.size(1))
        x = x + self.pos_emb(pos)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, bottleneck, pad_mask, causal_mask)
        return x

class AtomicAutoencoder(nn.Module):
    def __init__(self, num_atom_types, embed_dim, num_heads, num_layers, dropout=0.1, bottleneck_tokens=4):
        super().__init__()
        self.encoder = Encoder(num_atom_types, embed_dim, num_heads, num_layers, dropout)
        self.decoder = Decoder(num_atom_types, embed_dim, num_heads, num_layers, dropout)
        self.type_proj = nn.Linear(embed_dim, num_atom_types)
        self.coord_proj = nn.Linear(embed_dim, 3)
        
        # Bottleneck parameters
        self.bottleneck_tokens = bottleneck_tokens
        self.bottleneck_dim = embed_dim // bottleneck_tokens
        
        # Projection layers for bottleneck
        self.bottleneck_down = nn.Linear(embed_dim, self.bottleneck_dim)
        self.bottleneck_up = nn.Linear(self.bottleneck_dim, embed_dim)

    def forward(self, xc, xt, pad_mask=None):
        seq_len = xc.size(1)
        x, pos = self.encoder(xc, xt, pad_mask)
        
        # New bottleneck: take first N tokens instead of mean pooling
        batch_size = x.size(0)
        
        # Get first N tokens (or all available tokens if sequence is shorter)
        actual_bottleneck_tokens = min(self.bottleneck_tokens, seq_len)
        if pad_mask is not None:
            # Find the first non-padded tokens for each sequence in the batch
            bottleneck_tokens = []
            for b in range(batch_size):
                # Get indices of non-padded tokens
                non_pad_indices = torch.where(~pad_mask[b])[0]
                if len(non_pad_indices) >= actual_bottleneck_tokens:
                    # Take first N non-padded tokens
                    selected_indices = non_pad_indices[:actual_bottleneck_tokens]
                else:
                    # If not enough non-padded tokens, take what we have and pad
                    selected_indices = non_pad_indices
                    # Pad with the last available token if needed
                    while len(selected_indices) < actual_bottleneck_tokens:
                        selected_indices = torch.cat([selected_indices, selected_indices[-1:]])
                
                bottleneck_tokens.append(x[b, selected_indices])
            bottleneck_tokens = torch.stack(bottleneck_tokens)  # [batch_size, bottleneck_tokens, embed_dim]
        else:
            # No padding mask, just take first N tokens
            bottleneck_tokens = x[:, :actual_bottleneck_tokens, :]  # [batch_size, bottleneck_tokens, embed_dim]
        
        # Project down and back up to ensure total dimensionality constraint
        # Shape: [batch_size, bottleneck_tokens, embed_dim] -> [batch_size, bottleneck_tokens, bottleneck_dim]
        bottleneck_compressed = self.bottleneck_down(bottleneck_tokens)
        # Shape: [batch_size, bottleneck_tokens, bottleneck_dim] -> [batch_size, bottleneck_tokens, embed_dim]
        bottleneck_tokens = self.bottleneck_up(bottleneck_compressed)
        
        # Create shifted inputs for decoder
        shifted_xc = torch.zeros_like(xc)
        shifted_xt = torch.full_like(xt, PAD_ID)
        shifted_xc[:, 1:, :] = xc[:, :-1, :]
        shifted_xt[:, 1:] = xt[:, :-1]
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=xc.device) * float('-inf'), diagonal=1)
        
        # Pass multiple bottleneck tokens to decoder
        dec_output = self.decoder(shifted_xc, shifted_xt, bottleneck_tokens, pad_mask, causal_mask)
        atom_types = self.type_proj(dec_output)
        coords = self.coord_proj(dec_output)
        return atom_types, coords, pos

    def coord_mse_loss(self, coords_pred, coords_true, pad_mask):
        mask = (~pad_mask).unsqueeze(2)
        coords_pred = coords_pred[mask.repeat(1, 1, 3)]
        coords_true = coords_true[mask.repeat(1, 1, 3)]
        return F.mse_loss(coords_pred, coords_true)

EPOCHS = 25_000
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_ID = len(elem)

def load_pdb(file_path, model=0):
    atoms = parsePDB(file_path)
    num_models = atoms._n_csets
    if model >= num_models:
        raise ValueError(f"Model index {model} out of range. PDB file contains {num_models} models.")
    atoms.setACSIndex(model)
    ca_atoms = atoms.select("calpha")
    return atoms, ca_atoms

def get_atom_env(atoms, ca_atoms, idx, neighbor_distance=5.0):
    assert 0 <= idx < len(ca_atoms), "Index out of range for CA atoms."
    ca_atom = ca_atoms[idx]
    ca_pos = ca_atom.getCoords()
    neighbours = atoms.select(f'within {neighbor_distance} of target', target=ca_pos)
    coords = neighbours.getCoords()
    types = neighbours.getElements()
    types = [elem.index(element.upper()) for element in types]
    coords = coords - ca_pos
    r = np.linalg.norm(coords, axis=1)
    indices = np.argsort(r)
    coords = coords[indices]
    types = [types[i] for i in indices]
    return coords, types

data_dir = "training/data"
common_dir = os.path.join(data_dir, "common")
uncommon_dir = os.path.join(data_dir, "uncommon")
common_files = [os.path.join(common_dir, f) for f in os.listdir(common_dir) if f.endswith('.pt')]
uncommon_files = [os.path.join(uncommon_dir, f) for f in os.listdir(uncommon_dir) if f.endswith('.pt')]

def get_train_example(gen: np.random.Generator, common_files: list[str], uncommon_files: list[str]):
    if uncommon_files and gen.random() < 0.5:
        file = gen.choice(uncommon_files)
    else:
        file = gen.choice(common_files)
    data = torch.load(file)
    envs = data['envs']
    idx = gen.integers(0, len(envs))
    env = envs[idx]
    coords = env['coords']
    types = env['types']
    return coords, types

model = AtomicAutoencoder(len(element_list_stripped) + 1, 256, 8, 4, bottleneck_tokens=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
if USE_SCHEDULER:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, cooldown=250, min_lr=1e-6)

def train():
    gen = np.random.default_rng()
    model.train()
    losses = []
    for epoch in (pbar := tqdm.tqdm(range(EPOCHS))):
        try:
            x1c = []
            x1t = []
            for i in range(BATCH_SIZE):
                coords, types = get_train_example(gen, common_files, uncommon_files)
                x1c.append(coords)
                x1t.append(types)
            max_len = max(len(x) for x in x1c)
            pad_mask = torch.ones(BATCH_SIZE, max_len, dtype=torch.bool)
            for i in range(len(x1c)):
                original_len = len(x1c[i])
                x1c[i] = F.pad(x1c[i], (0, 0, 0, max_len - original_len), value=0)
                x1t[i] = F.pad(x1t[i], (0, max_len - original_len), value=PAD_ID)
                pad_mask[i, :original_len] = False
            pad_mask = pad_mask.to(device)
            x1c = torch.stack(x1c).to(device)
            x1t = torch.stack(x1t).to(device)
            optimizer.zero_grad()
            output_atom_logits, coords_pred, pos_true = model(x1c, x1t, pad_mask)
            coord_loss = model.coord_mse_loss(coords_pred, x1c, pad_mask)
            # Use original loss calculation method
            pad_tok = F.one_hot(torch.tensor(PAD_ID), num_classes=output_atom_logits.size(-1)).float().to(device)
            x = output_atom_logits.view(-1, output_atom_logits.size(-1))
            tv = x1t.view(-1).unsqueeze(1)
            x = torch.where(tv != PAD_ID, x, pad_tok)
            y = x1t.view(-1)
            loss_mask = (x1t != PAD_ID).view(-1)          # 1 = keep, 0 = pad
            cat_loss = entmax_bisect_loss(
                x.view(-1, x.size(-1))[loss_mask],
                x1t.view(-1)[loss_mask]
            ).mean()
            loss = cat_loss + coord_loss
            loss.backward()
            optimizer.step()
            if USE_SCHEDULER:
                previous_lr = optimizer.param_groups[0]['lr']
                scheduler.step(loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != previous_lr:
                    print(f"Learning rate updated from {previous_lr} to {new_lr}")
            pbar.set_description(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.2f} (Cat Loss: {cat_loss.item():.2f}, Coord Loss: {coord_loss.item():.2f})')
            losses.append(loss.item())
        except ValueError as e:
            print(f"Error during training: {e}")
            continue
    torch.save(model.state_dict(), "autoregressive_atomic_autoencoder.pth")
    print("Model saved as autoregressive_atomic_autoencoder.pth")

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.close()

if __name__ == "__main__":
    train()
    print("Training complete.")
