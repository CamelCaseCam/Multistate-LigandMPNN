'''
Training script for bidirectional folding/diffusion model based on atomic autoencoder.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from Bio import SeqIO
from scipy.spatial.transform import Rotation as R
from load_nmr import element_list_stripped
import random
from entmax import entmax_bisect_loss

import matplotlib.pyplot as plt
import tqdm

from encoder import AtomicAutoencoder

ENCODER_PATH = 'autoregressive_atomic_autoencoder.pth'
VERBOSE = False
EPOCHS = 10

# Load encoder model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = AtomicAutoencoder(len(element_list_stripped) + 1, 256, 8, 4, bottleneck_tokens=8).to(device)
if os.path.exists(ENCODER_PATH):
    autoencoder.load_state_dict(torch.load(ENCODER_PATH, map_location='cpu'))
    print(f'Loaded encoder model from {ENCODER_PATH}')

autoencoder.eval()

amino_acid_tokens = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",   
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    "X", # For unknown amino acids/masked tokens
]

# Define the sequence encoder stack
class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len=512, dropout=0.1):
        super(SequenceEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.max_len = max_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def get_pos_embedding(self, B, L):
        # Sinusoidal positional encoding
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(0).repeat(B, 1)
        position = position.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-np.log(10000.0) / self.embed_dim))
        pos_emb = torch.zeros(B, L, self.embed_dim)
        pos_emb[:, :, 0::2] = torch.sin(position * div_term)
        pos_emb[:, :, 1::2] = torch.cos(position * div_term)
        
        return pos_emb.to(device)
    
    def forward(self, sequence, mask=None):
        """
        Args:
            sequence: (B, L) tensor of sequence tokens
            mask: (B, L) boolean mask where True indicates padding
        Returns:
            encoded: (B, L, embed_dim) encoded sequence tensor
        """
        B, L = sequence.shape
        
        # Create embeddings
        seq_emb = self.embedding(sequence)  # (B, L, embed_dim)
        pos_emb = self.get_pos_embedding(B, L)

        # Combine embeddings
        x = seq_emb + pos_emb
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.layer_norm(x)

        return x
    
def load_pdbs(pdb_ids : list[str], is_uncommon_list : list[bool], max_len=512):
    '''
    Load a batch of PDB files and return tensors for their sequences and local alpha carbon environments.
    Skip if file not found.
    '''
    sequences = []
    coords = []
    types = []
    max_batch_len = 0
    max_atoms = 80
    for pdb_id, is_uncommon in zip(pdb_ids, is_uncommon_list):
        datapath = f"training/data/{'uncommon' if is_uncommon else 'common'}/{pdb_id}.pt"
        if not os.path.exists(datapath):
            print(f"PDB file {datapath} not found, skipping.")
            continue
        envs = torch.load(datapath)["envs"]
        # Make sure it's not longer than max_len
        if len(envs) > max_len:
            print(f"PDB {pdb_id} has more than {max_len} CA atoms, skipping.") if VERBOSE else None
            continue
        max_batch_len = max(max_batch_len, len(envs))
        sequence = [amino_acid_tokens.index(env['residue']) for env in envs]
        sequence = torch.tensor(sequence, dtype=torch.long)
        sequences.append(sequence)

        pdb_coords = []
        pdb_types = []
        for env in envs:
            c = env['coords']
            t = env['types']
            num_atoms = len(c)
            if num_atoms > max_atoms:
                c = c[:max_atoms]
                t = t[:max_atoms]
            else:
                c_pad = torch.zeros((max_atoms - num_atoms, 3), dtype=torch.float32)
                c = torch.cat([c, c_pad], dim=0)
                t_pad = torch.full((max_atoms - num_atoms,), len(element_list_stripped), dtype=torch.long)
                t = torch.cat([t, t_pad], dim=0)
            pdb_coords.append(c)
            pdb_types.append(t)
        coords.append(torch.stack(pdb_coords, dim=0))
        types.append(torch.stack(pdb_types, dim=0))
    if not sequences:
        return None, None, None
    # Pad sequences, coords, and types to max_batch_len
    padded_sequences = []
    padded_coords = []
    padded_types = []
    for i in range(len(sequences)):
        seq = sequences[i]
        coord = coords[i]
        type_ = types[i]
        if len(seq) < max_batch_len:
            padding = torch.full((max_batch_len - len(seq),), fill_value=amino_acid_tokens.index("X"), dtype=torch.long)
            seq = torch.cat([seq, padding], dim=0)
            coord_padding = torch.zeros((max_batch_len - len(coord), max_atoms, 3), dtype=torch.float32)
            coord = torch.cat([coord, coord_padding], dim=0)
            type_padding = torch.full((max_batch_len - len(type_), max_atoms), fill_value=len(element_list_stripped), dtype=torch.long)
            type_ = torch.cat([type_, type_padding], dim=0)
        padded_sequences.append(seq)
        padded_coords.append(coord)
        padded_types.append(type_)
    padded_sequences = torch.stack(padded_sequences, dim=0).to(device)
    padded_coords = torch.stack(padded_coords, dim=0).to(device)
    padded_types = torch.stack(padded_types, dim=0).to(device)
    return padded_sequences, padded_coords, padded_types

class Encoder(nn.Module):
    # Note: vocab_size should include the special token for padding and the token for unknown aminos.
    def __init__(self, atomic_autoencoder : AtomicAutoencoder, vocab_size, embed_dim, num_heads, num_layers, max_len=512, dropout=0.1, atomic_encoder_batch_size=256):
        super(Encoder, self).__init__()
        self.sequence_encoder = SequenceEncoder(vocab_size, embed_dim, num_heads, num_layers, max_len, dropout)
        self.atomic_autoencoder = atomic_autoencoder
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.atomic_encoder_batch_size = atomic_encoder_batch_size
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = self.sequence_encoder.get_pos_embedding  # Reuse the pos embed function
        
    def forward(self, sequence, coord, type_tensor):
        B, L = sequence.shape

        # Flatten and then unflatten to encode atomic environments
        with torch.no_grad():
            coord_flat = coord.view(B * L, -1, 3)
            type_flat = type_tensor.view(B * L, -1)
            atomic_embeddings = torch.zeros((B * L, self.embed_dim), device=coord.device)
            for i in range(0, B * L, self.atomic_encoder_batch_size):
                end = min(i + self.atomic_encoder_batch_size, B * L)
                output, _ = self.atomic_autoencoder.encoder(coord_flat[i:end], type_flat[i:end])   # (B * L, N, embed_dim)
                # Get the bottleneck
                bottleneck_tokens = output[:, :self.atomic_autoencoder.bottleneck_tokens, :]  # Assume at least 8 atoms near each CA
                bottleneck_compressed = self.atomic_autoencoder.bottleneck_down(bottleneck_tokens)  # (B * L, 8, embed_dim // 8)
                # Cat and pad
                emb = bottleneck_compressed.view(end - i, -1)
                if emb.shape[1] < self.embed_dim:
                    emb = F.pad(emb, (0, self.embed_dim - emb.shape[1]), value=0)
                else:
                    emb = emb[:, :self.embed_dim]
                atomic_embeddings[i:end] = emb
            atomic_embeddings = atomic_embeddings.view(B, L, self.embed_dim)

        # Get sequence embeddings
        sequence_embeddings = self.sequence_encoder(sequence, mask=None)

        # Add pos to atomic as well
        pos_emb = self.pos_embed(B, L)
        atomic_embeddings = atomic_embeddings + pos_emb

        return sequence_embeddings, atomic_embeddings

class BidirectionalLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn_seq = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_env = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_seq_to_env = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_env_to_seq = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff_seq = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ff_env = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1_seq = nn.LayerNorm(embed_dim)
        self.norm2_seq = nn.LayerNorm(embed_dim)
        self.norm3_seq = nn.LayerNorm(embed_dim)
        self.norm1_env = nn.LayerNorm(embed_dim)
        self.norm2_env = nn.LayerNorm(embed_dim)
        self.norm3_env = nn.LayerNorm(embed_dim)

    def forward(self, seq, env):
        # Self attention
        seq_res = self.self_attn_seq(seq, seq, seq)[0]
        seq = self.norm1_seq(seq + seq_res)
        env_res = self.self_attn_env(env, env, env)[0]
        env = self.norm1_env(env + env_res)

        # Cross attention
        seq_res = self.cross_attn_seq_to_env(seq, env, env)[0]
        seq = self.norm2_seq(seq + seq_res)
        env_res = self.cross_attn_env_to_seq(env, seq, seq)[0]
        env = self.norm2_env(env + env_res)

        # FF
        seq_res = self.ff_seq(seq)
        seq = self.norm3_seq(seq + seq_res)
        env_res = self.ff_env(env)
        env = self.norm3_env(env + env_res)

        return seq, env

class BidirectionalDiffusion(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, num_timesteps=1000, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layers = nn.ModuleList([BidirectionalLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.out_seq = nn.Linear(embed_dim, embed_dim)
        self.out_env = nn.Linear(embed_dim, embed_dim)

    def forward(self, seq, env, t_seq, t_env):
        # t_seq, t_env: (B,)
        t_seq_emb = self.time_embed((t_seq / self.num_timesteps).unsqueeze(-1).float())
        t_env_emb = self.time_embed((t_env / self.num_timesteps).unsqueeze(-1).float())

        for layer in self.layers:
            seq = seq + t_seq_emb.unsqueeze(1)
            env = env + t_env_emb.unsqueeze(1)
            seq, env = layer(seq, env)

        pred_noise_seq = self.out_seq(seq)
        pred_noise_env = self.out_env(env)

        return pred_noise_seq, pred_noise_env

if __name__ == "__main__":
    # -------------------------------------------------------------
    # 1) Parameters
    # -------------------------------------------------------------
    embed_dim = 128
    num_heads = 8
    num_encoder_layers = 3
    num_diffusion_layers = 8

    # The total number of diffusion timesteps (T)
    num_timesteps = 1000

    max_len = 512
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = EPOCHS   # make sure EPOCHS is defined somewhere

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------
    # 2) Build a linear beta schedule (β₁,…,β_T)
    # -------------------------------------------------------------
    # Here we choose a simple linear schedule from β₁=1e-4 up to β_T=0.02.
    # You can adjust these endpoints or try cosine schedules, etc.
    beta_start = 1e-4
    beta_end   = 2e-2
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)  # shape: [T]

    # α_t = 1 - β_t
    alphas = 1.0 - betas                                                             # shape: [T]
    # \bar α_t = ∏_{i=1}^t α_i
    alpha_bars = torch.cumprod(alphas, dim=0)                                        # shape: [T]

    # For convenience, we'll later index alpha_bars[t-1] when t ∈ [1…T].

    # -------------------------------------------------------------
    # 3) Get PDB lists
    # -------------------------------------------------------------
    common_pdbs = [f.split('.')[0] for f in os.listdir('training/data/common') if f.endswith('.pt')]
    uncommon_pdbs = [f.split('.')[0] for f in os.listdir('training/data/uncommon') if f.endswith('.pt')]
    all_pdbs = [(pdb, False) for pdb in common_pdbs] + [(pdb, True) for pdb in uncommon_pdbs]

    # -------------------------------------------------------------
    # 4) Instantiate models + optimizer
    # -------------------------------------------------------------
    encoder = Encoder(
        autoencoder,
        vocab_size=len(amino_acid_tokens),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_encoder_layers,
        max_len=max_len
    ).to(device)

    diffusion = BidirectionalDiffusion(
        embed_dim=embed_dim,
        num_layers=num_diffusion_layers,
        num_heads=num_heads,
        num_timesteps=num_timesteps
    ).to(device)

    optimizer = torch.optim.Adam(
        list(diffusion.parameters()) + list(encoder.parameters()),
        lr=learning_rate
    )

    # -------------------------------------------------------------
    # 5) Training loop
    # -------------------------------------------------------------
    for epoch in (pbar := tqdm.tqdm(range(num_epochs))):
        random.shuffle(all_pdbs)
        total_loss = 0.0
        num_iters = 0

        for i in range(0, len(all_pdbs), batch_size):
            batch = all_pdbs[i:i+batch_size]
            pdb_ids = [p[0] for p in batch]
            is_uncommon = [p[1] for p in batch]

            # ---------------------------------------------------------
            # a) Load data (seq_emb, atomic_emb) from your encoder
            # ---------------------------------------------------------
            sequences, coords, types = load_pdbs(pdb_ids, is_uncommon, max_len)
            if sequences is None:
                continue

            # seq_emb:   [batch_size, seq_len, embed_dim]
            # atomic_emb:[batch_size, num_atoms, embed_dim]    (for example)
            seq_emb, atomic_emb = encoder(sequences, coords, types)

            # ---------------------------------------------------------
            # b) Sample random timesteps t_seq, t_env ∈ {1,…,T}
            # ---------------------------------------------------------
            batch_size_actual = sequences.shape[0]
            t_seq = torch.randint(
                low=1, high=num_timesteps + 1,
                size=(batch_size_actual,),
                device=device,
                dtype=torch.long
            )  # shape: [B]
            t_env = torch.randint(
                low=1, high=num_timesteps + 1,
                size=(batch_size_actual,),
                device=device,
                dtype=torch.long
            )  # shape: [B]

            # ---------------------------------------------------------
            # c) For each example, look up ᾱ_t  = alpha_bars[t-1]
            # ---------------------------------------------------------
            # alpha_bars[t_seq - 1] : [B]
            alpha_bar_seq = alpha_bars[t_seq - 1].view(-1, 1, 1)   # reshape to [B,1,1] for broadcasting
            alpha_bar_env = alpha_bars[t_env - 1].view(-1, 1, 1)   # shape [B,1,1]

            # ---------------------------------------------------------
            # d) Sample noise ε ∼ N(0,I)  for both streams
            # ---------------------------------------------------------
            # (Match the shape of seq_emb / atomic_emb exactly.)
            noise_seq = torch.randn_like(seq_emb)      # [B, seq_len, embed_dim]
            noise_env = torch.randn_like(atomic_emb)   # [B, num_atoms, embed_dim]

            # ---------------------------------------------------------
            # e) Construct the noisy inputs x_t  via exact DDPM formula:
            #    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * ε
            # ---------------------------------------------------------
            sqrt_alpha_bar_seq = torch.sqrt(alpha_bar_seq)            # [B,1,1]
            sqrt_one_minus_alpha_bar_seq = torch.sqrt(1 - alpha_bar_seq)  # [B,1,1]

            # noisy_seq: [B, seq_len, embed_dim]
            noisy_seq = sqrt_alpha_bar_seq * seq_emb + sqrt_one_minus_alpha_bar_seq * noise_seq

            sqrt_alpha_bar_env = torch.sqrt(alpha_bar_env)            # [B,1,1]
            sqrt_one_minus_alpha_bar_env = torch.sqrt(1 - alpha_bar_env)  # [B,1,1]

            # noisy_env: [B, num_atoms, embed_dim]
            noisy_env = sqrt_alpha_bar_env * atomic_emb + sqrt_one_minus_alpha_bar_env * noise_env

            # ---------------------------------------------------------
            # f) Run the diffusion network to predict the noise at step t
            #
            #    Your BidirectionalDiffusion should accept:
            #      (noisy_seq, noisy_env, t_seq, t_env)
            #
            #    and output:
            #      (pred_noise_seq, pred_noise_env),
            #    where each prediction has the same shape as the original noise.
            # ---------------------------------------------------------
            pred_noise_seq, pred_noise_env = diffusion(noisy_seq, noisy_env, t_seq, t_env)

            # ---------------------------------------------------------
            # g) Compute the MSE loss against the *true* ε
            #    (This matches the DDPM derivation exactly.)
            # ---------------------------------------------------------
            loss_seq = F.mse_loss(pred_noise_seq, noise_seq)
            loss_env = F.mse_loss(pred_noise_env, noise_env)
            loss = loss_seq + loss_env

            # ---------------------------------------------------------
            # h) Backprop + step
            # ---------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---------------------------------------------------------
            # i) Logging
            # ---------------------------------------------------------
            num_iters += 1
            total_loss += loss.item()
            avg_loss = total_loss / num_iters
            pbar.set_description(
                f"Epoch {epoch+1}/{num_epochs}  batch {i}/{len(all_pdbs)}  Loss: {avg_loss:.4f}"
            )

    # -------------------------------------------------------------
    # 6) Save the diffusion model (and optionally the encoder)
    # -------------------------------------------------------------
    torch.save(diffusion.state_dict(), 'bidirectional_diffusion_8l.pth')
    torch.save(encoder.state_dict(),   'encoder_8l.pth')
