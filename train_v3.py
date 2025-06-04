'''
Defines a diffusion-based variant of LigandMPNN. The model first encodes the ligand/structure information for each state 
using the existing LigandMPNN encoder. It then uses a two-layer decoder to denoise sequence information. 

Each decoder layer consists of a CNN layer, the output of which is cat'd with the embeddings and processed by a 
linear layer. 

The output of the final layer is passed through a sigmoid activation to produce the final output (N x 21)
'''

from model_utils import ProteinMPNN, cat_neighbors_nodes
import torch
from load_nmr import get_dicts_from_pdb
import numpy as np
import tqdm
from data_utils import featurize, alphabet
import os
import math


from download_data import download_one

from models.common import categorical_diffuse, betas, w, t

MODEL_PATH = "model_params/ligandmpnn_v_32_030_25.pt"
EPOCHS = 200
MIN_STATES = 2
MAX_STATES = 6
NUM_TRAIN_PER_EPOCH = 40
BATCH_SIZE = 16
NUM_DIFF_STEPS = 100
DIFF_NOISE_SCALE = 1.0
MAX_TOTAL_SEQ_LEN = 512
MAX_BROADCAST_VAL = 1e4
TEST_WITH_CPU = False
USE_SCHEDULER = True
PROB_SELF = 2.0
device = torch.device("cuda") if torch.cuda.is_available() and not TEST_WITH_CPU else torch.device("cpu")

# Add these new constants near the top (e.g., after EPOCHS, etc.)
MOMENTUM = 0.8  # Momentum for weight updates (0.9 is a common value for smoothing)
TEST_WEIGHT = False  # If True, exit after 2 epochs, print weights, and skip loss curve

AMINO_DIM = 21
MASK_ID = 21
TOTAL_DIM = 22

checkpoint = torch.load(MODEL_PATH, map_location=device)
atom_context_num = checkpoint["atom_context_num"]
ligand_mpnn_use_side_chain_context = True
k_neighbors = checkpoint["num_edges"]
LOAD_PREV = True

class DiffLayer(torch.nn.Module):
    def __init__(self, hidden_dim, rank=16, attn_impl="flash"):  # "flash" or "xformers"
        super().__init__()
        self.h = hidden_dim
        self.r = rank
        self.attn_impl = attn_impl

        # local conv
        self.conv = torch.nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)

        self.broadcast_conv = torch.nn.Conv1d(hidden_dim, hidden_dim, 3, stride=3, padding=1)

        # SE gate
        self.se_fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 4)
        self.se_fc2 = torch.nn.Linear(hidden_dim // 4, hidden_dim)

        # low-rank projections
        self.num_heads = rank
        self.head_dim  = hidden_dim
        qkv_dim = rank * hidden_dim

        self.proj_q = torch.nn.Linear(hidden_dim, qkv_dim, bias=False)
        self.proj_k = torch.nn.Linear(hidden_dim, qkv_dim, bias=False)
        self.proj_v = torch.nn.Linear(hidden_dim, qkv_dim, bias=False)
        self.proj_out = torch.nn.Linear(qkv_dim, hidden_dim, bias=False)
        self.ln_attn = torch.nn.LayerNorm(hidden_dim)

        # final projection
        self.linear = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.drop  = torch.nn.Dropout(0.1)

    def _flash_attn(self, Q, K, V):
        # Q,K,V: (B,L,r)
        B, L, r = Q.shape
        Qh = Q.transpose(1,2)        # (B,r,L)
        Kh = K.transpose(1,2)
        Vh = V.transpose(1,2)
        ctx = torch.nn.functional.scaled_dot_product_attention(
                  Qh, Kh, Vh, dropout_p=0.0)
        ctx = ctx.transpose(1,2).reshape(B, L, r)   # (B,L,r)
        return ctx

    def _xformers_attn(self, Q, K, V):
        from xformers.ops import memory_efficient_attention
        B, L, _ = Q.shape
        Q4 = Q.view(B, L, self.num_heads, self.head_dim).to(torch.float16)
        K4 = K.view(B, L, self.num_heads, self.head_dim).to(torch.float16)
        V4 = V.view(B, L, self.num_heads, self.head_dim).to(torch.float16)
        ctx = memory_efficient_attention(Q4, K4, V4)          # (B,L,h,d)
        ctx = ctx.view(B, L, -1).to(Q.dtype)                  # (B,L,h*d)
        return ctx

    def forward(self, x, embeddings):        # x: (B,L,H)
        # 1. local conv
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        x = self.norm1(torch.relu(x))
        x_orig = x.clone()

        # 2. SE gate
        g = torch.sigmoid(self.se_fc2(torch.relu(self.se_fc1(x.mean(1)))))
        x = x * g.unsqueeze(1)

        # 3. low-rank attention (flash or xformers)
        '''Q, K, V = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        if self.attn_impl == "flash":
            context = self._flash_attn(Q, K, V)
        elif self.attn_impl == "xformers":
            context = self._xformers_attn(Q, K, V)

        if self.attn_impl is not None:
            x = self.ln_attn(x + self.proj_out(context))'''

        # 4. concat structural embeddings
        '''x = self.linear(torch.cat([x, embeddings], dim=-1))
        x = self.norm2(torch.relu(x))
        x = self.drop(x)'''
        '''
        Trying an attention-based combination instead of linear. This choice of QKV should allow the model to associate
        residues using structural information
        '''
        '''Q = self.proj_q(x)
        K = self.proj_k(embeddings)
        V = self.proj_v(embeddings)
        if self.attn_impl == "flash":
            context = self._flash_attn(Q, K, V)
        elif self.attn_impl == "xformers":
            context = self._xformers_attn(Q, K, V)

        if self.attn_impl is not None:
            x_emb = self.ln_attn(embeddings + self.proj_out(context))
        
        x = self.linear(torch.cat([x_orig, x_emb], dim=-1))
        x = self.norm2(torch.relu(x))
        x = self.drop(x)'''

        # Broadcasting
        x_broadcast = x.clone()
        while x_broadcast.shape[1] > 1:
            x_broadcast = self.broadcast_conv(x_broadcast.transpose(1,2)).transpose(1,2) 
        x_broadcast = x_broadcast.repeat(1, x.shape[1], 1)

        x = self.linear(torch.cat([x_orig, x_broadcast], dim=-1))
        x = self.norm2(torch.relu(x))
        x = self.drop(x)

        return x

class diffmodel(torch.nn.Module):
    def __init__(self, orig_model, k_neighbors=32, EXV_dim=384):
        super(diffmodel, self).__init__()
        self.orig_model = orig_model
        self.hidden_dim = orig_model.hidden_dim
        self.k_neighbors = k_neighbors
        self.EXV_dim = EXV_dim

        '''
        Define a linear model to combine embeddings across states

        States are combined by using the linear layer to create a ratio across all states of embeddings
        '''
        self.collect_weights = torch.nn.Linear(EXV_dim * k_neighbors, self.hidden_dim)
        self.collect_weights_attn = torch.nn.Linear(EXV_dim, self.hidden_dim)
        self.comb_weights = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.comb_weights_attn = torch.nn.Linear(EXV_dim * 2, EXV_dim)

        # For encoding sequence information
        self.seq_enc_layer = torch.nn.Embedding(TOTAL_DIM, self.hidden_dim)

        self.decoder_layers = torch.nn.ModuleList(
            [DiffLayer(self.hidden_dim, rank=16, attn_impl="xformers") for _ in range(4)]
        )
        self.W_out = torch.nn.Linear(self.hidden_dim, 21)
        self.t_embed = torch.nn.Embedding(NUM_DIFF_STEPS, self.hidden_dim)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, feature_dict_list):
        # Get embeddings from the original model
        assert len(feature_dict_list) > 0
        raise NotImplementedError("This model is not implemented yet")
    
    def integrate_embeddings(self, h_EXV_encoder_fw):
        # First, stack the embeddings across nodes so (num_states, L, k, H) -> (num_states, L, H * k)
        h_EXV_encoder_fw_stacked = h_EXV_encoder_fw.view(h_EXV_encoder_fw.shape[0], h_EXV_encoder_fw.shape[1], -1)
        # There may be fewer than 32 neighbors for some nodes, so we need to pad the embeddings along the final dimension
        h_EXV_encoder_fw_stacked = torch.nn.functional.pad(h_EXV_encoder_fw_stacked, (0, self.k_neighbors * self.EXV_dim - h_EXV_encoder_fw_stacked.shape[2]), value=0)
        # Then, apply the linear layer to get the combined weights
        h_EXV_encoder_fw = self.collect_weights(h_EXV_encoder_fw_stacked)

        # Now, create the ratio
        ratio = torch.zeros_like(h_EXV_encoder_fw)
        for state_idx in range(1, h_EXV_encoder_fw.shape[0]):
            last = h_EXV_encoder_fw[state_idx - 1]
            current = h_EXV_encoder_fw[state_idx]
            inp = torch.cat([last, current], dim=-1)
            ratio[state_idx] = self.comb_weights(inp)
        # Now, use the ratio to combine the embeddings (summing to 1)
        ratio = ratio / (torch.sum(ratio, dim=-1, keepdim=True) + 1e-8)
        # Summing time!
        embeddings = torch.sum(ratio * h_EXV_encoder_fw, dim=0, keepdim=True)
        return embeddings
    
    def integrate_embeddings_attn(self, h_V):
        ratio = torch.zeros_like(h_V)
        for state_idx in range(1, h_V.shape[0]):
            last = h_V[state_idx - 1]
            current = h_V[state_idx]
            inp = torch.cat([last, current], dim=-1)
            ratio[state_idx] = self.comb_weights(inp)
        # Now, use the ratio to combine the embeddings (summing to 1)
        ratio = ratio / (torch.sum(ratio, dim=-1, keepdim=True) + 1e-8)
        # Summing time!
        embeddings = torch.sum(ratio * h_V, dim=0, keepdim=True)
        return embeddings
    
def train_oneshot_backup(model : diffmodel, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only = False):
    #Validate input
    if not feature_dict_list or not isinstance(feature_dict_list, list):
        raise ValueError("feature_dict_list must be a non-empty list")
    
    num_states = len(feature_dict_list)
    if num_states < 2:
        raise ValueError("At least two states must be provided")
    output_dict = {}

    optimizer.zero_grad()

    # Use first feature dict to get basic parameters
    B_decoder = len(feature_dict_list)
    S_true = feature_dict_list[0]["S"]
    mask = feature_dict_list[0]["mask"]
    chain_mask = feature_dict_list[0]["chain_mask"]
    if eval_only:
        # Score actual sequence with the original model
        orig_output = orig_model.score(feature_dict_list[0], use_sequence=True)
        orig_logits = orig_output["logits"]
        mean_orig_loss = torch.mean(loss_fn(orig_logits.squeeze(0), S_true[0, :].long()))
        output_dict["loss_orig"] = mean_orig_loss.detach().cpu().item()
    
    # Convert mask and chain mask to (len(feature_dict_list), L) shape
    mask = mask.repeat(B_decoder, 1)
    chain_mask = chain_mask.repeat(B_decoder, 1)

    bias = feature_dict_list[0]["bias"]
    randn = feature_dict_list[0]["randn"]
    temperature = feature_dict_list[0]["temperature"]

    B, L = S_true.shape
    state_num = B_decoder
    device = S_true.device

    # Encode all states
    h_V_list = []
    h_E_list = []
    E_idx_list = []
    for feature_dict in feature_dict_list:
        h_V, h_E, E_idx = model.orig_model.encode(feature_dict)
        h_V_list.append(h_V)
        h_E_list.append(h_E)
        E_idx_list.append(E_idx)
    h_V = torch.cat(h_V_list, dim=0)

    chain_mask = mask * chain_mask
    decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(randn)))

    # Handle symmetry (assuming same symmetry across states)
    E_idx_base = E_idx_list[0].repeat(B_decoder, 1, 1)
    permutation_matrix_reverse = torch.nn.functional.one_hot(
        decoding_order, num_classes=L
    ).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(L, L, device=device))),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )
    mask_attend = torch.gather(order_mask_backward, 2, E_idx_base).unsqueeze(-1)
    mask_1D = mask.view([B_decoder, L, 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    # Initialize state-specific variables
    S_true = S_true.repeat(1, 1)
    chain_mask = chain_mask.repeat(B_decoder, 1)
    mask = mask.repeat(B_decoder, 1)
    bias = bias.repeat(B_decoder, 1, 1)

    all_probs = torch.zeros((B_decoder, L, 21), device=device)
    all_log_probs = torch.zeros((B_decoder, L, 21), device=device)
    S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=device)
    h_S = torch.zeros((B_decoder, L, model.hidden_dim), device=device)

    # Initialize separate h_V stacks for each state
    h_V_stack = [h_V]
    h_E   = torch.cat(h_E_list,   dim=0)      # (num_states, L, k, H)
    E_idx = torch.cat(E_idx_list, dim=0)      # (num_states, L, k)

    # Pre-compute encoder embeddings
    h_EXV_encoder_fw_list = []
    for state_idx in range(num_states):
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S[state_idx:state_idx+1]), h_E[state_idx].unsqueeze(0), E_idx[state_idx].unsqueeze(0))
        h_EXV_encoder = cat_neighbors_nodes(h_V_stack[0][state_idx].unsqueeze(0), h_EX_encoder, E_idx[state_idx].unsqueeze(0))
        h_EXV_encoder_fw_list.append((mask_fw[state_idx:state_idx+1] * h_EXV_encoder).detach())
    h_EXV_encoder_fw = torch.cat(h_EXV_encoder_fw_list, dim=0).detach()

    embeddings = model.integrate_embeddings_attn(h_V)
    # Repeat embeddings for batches
    embeddings = embeddings.repeat(len(diff_levels), 1, 1)
    # Embeddings are now of the shape (1, L, H)
    # Now, we can start the diffusion process
    # First, encode the ground truth sequence
    true_seq = S_true.long()
    # Now, we can start the diffusion process (parallelized)
    x = torch.zeros((len(diff_levels), L), device=device)
    true_seq = true_seq.detach()
    gen = torch.Generator()
    
    self_diff_levels = torch.tensor([99], device="cpu")
    self_diff_levels = torch.clamp(self_diff_levels, 0, NUM_DIFF_STEPS - 1).repeat(len(diff_levels))
    x = categorical_diffuse(true_seq.repeat(len(diff_levels), 1), self_diff_levels, gen, device)
    x_tm1 = x.clone()
    self_diff_levels = self_diff_levels.to(device)
    # Generate a new sequence from noise and feed it into the model
    x_self_emb = model.seq_enc_layer(x.long())
    t_self_vec = model.t_embed(self_diff_levels)
    emb_copy = embeddings.clone()
    for layer in model.decoder_layers:
        x_self_emb = x_self_emb + t_self_vec.unsqueeze(1)
        x_self_emb = layer(x_self_emb, emb_copy)
    
    # Get probs
    self_logits = model.W_out(x_self_emb)

    # Get loss
    loss = torch.zeros(1, device=device)
    for i in range(len(diff_levels)):
        logits_i = self_logits[i]
        tar_i = x_tm1[i].long()
        ce_t = loss_fn(logits_i, S_true.long().squeeze(0))
        loss += (1 * ce_t).mean() if not eval_only else ce_t.mean()
    loss.backward()
    optimizer.step()

    output_dict["loss"] = loss.detach().cpu().item() / len(diff_levels)

    return output_dict

DATA_PATH = "training/train_multistate.json"
PDB_PATH = "training/pdbs/"

model_type = "edgeemb_attn"
MODEL_NAMES = [
    "parallel_threetensor",
    "inc_layer",
    "norm_layer",
    "attn_integ",
    "initlayer",
    "attn_seq",
    "init_attn_seq",
    "conv_node",
    "initlayer_hd",
    "initlayer_hdt",
    "edgeemb_attn",
]

import json

with open(DATA_PATH, "r") as f:
    data = json.load(f)

orig_model = ProteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=k_neighbors,
    atom_context_num=atom_context_num,
    model_type="ligand_mpnn",
    ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    device=device,
    debug=False,
)

# Load the requested model type
model = None
if model_type not in MODEL_NAMES:
    raise ValueError(f"Model type {model_type} not found. Available types: {MODEL_NAMES}")
if model_type == "ablated_embeddings":
    from models.ablated_embeddings import diffmodel
    model = diffmodel(orig_model)
elif model_type == "attn_h_V":
    from models.attn_h_V import diffmodel
    model = diffmodel(orig_model)
elif model_type == "declayer_attn_h_V":
    from models.declayer_attn_h_V import diffmodel
    model = diffmodel(orig_model)
elif model_type == "attn_h_EXVV":
    from models.attn_h_EXVV import diffmodel
    model = diffmodel(orig_model)
elif model_type == "twolayer_gelu":
    from models.twolayer_gelu import diffmodel
    model = diffmodel(orig_model)
elif model_type == "parallel_threetensor":
    from models.parallel_threetensor import diffmodel
    model = diffmodel(orig_model)
elif model_type == "inc_layer":
    from models.inc_layer import diffmodel
    model = diffmodel(orig_model)
elif model_type == "norm_layer":
    from models.norm_layer import diffmodel
    model = diffmodel(orig_model)
elif model_type == "attn_integ":
    from models.attn_integ import diffmodel
    model = diffmodel(orig_model)
elif model_type == "initlayer":
    from models.initlayer import diffmodel
    model = diffmodel(orig_model)
elif model_type == "attn_seq":
    from models.attn_seq import diffmodel
    model = diffmodel(orig_model)
elif model_type == "init_attn_seq":
    from models.init_attn_seq import diffmodel
    model = diffmodel(orig_model)
elif model_type == "conv_node":
    from models.conv_node import diffmodel
    model = diffmodel(orig_model)
elif model_type == "initlayer_hd":
    from models.initlayer_hd import diffmodel
    model = diffmodel(orig_model)
elif model_type == "initlayer_hdt":
    from models.initlayer_hdt import diffmodel
    model = diffmodel(orig_model)
elif model_type == "edgeemb_attn":
    from models.edgeemb_attn import diffmodel
    model = diffmodel(orig_model)

model.to(device)


orig_model.load_state_dict(checkpoint["model_state_dict"])
orig_model.to(device)

# Set up optimizer with limited weights for finetuning
trainable_params = list(model.parameters())

optimizer = torch.optim.Adam(
    trainable_params,
    lr=1e-4,
    weight_decay=0.0,
)
# Set up loss function
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

# Set up scheduler
if USE_SCHEDULER:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

def categorical_diffuse_mix(x0, xt, t_idx, rng, model, device):
    """
    x0     : (B, L, H) ground-truth tokens (0‥TOTAL_DIM-1)
    xt     : (B, L, H) model-output tokens (0‥TOTAL_DIM-1)
    t_idx  : scalar or (B,)   current timestep 0‥T-1
    rng    : torch.Generator   (to keep things reproducible)
    model  : model to use for embedding
    device : device to use for the model

    returns x_t  : (B, L) corrupted tokens
    """
    beta  = betas[t_idx]                      # scalar
    keep_mask = torch.bernoulli(
        (1.0 - beta.unsqueeze(1).unsqueeze(2)).expand_as(x0), generator=rng
    ).bool().to(device)                                 # True → keep original

    # apply corruption
    x_t = torch.where(keep_mask, x0, xt)
    return x_t


def train(model : diffmodel, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only = False):
    if model_type not in MODEL_NAMES:
        raise ValueError(f"Model type {model_type} not found. Available types: {MODEL_NAMES}")
    if model_type == "attn_integ":
        from models.attn_integ import train as train_attn_integ
        return train_attn_integ(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "initlayer":
        from models.initlayer import train as train_initlayer
        return train_initlayer(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "attn_seq":
        from models.attn_seq import train as train_attn_seq
        return train_attn_seq(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "init_attn_seq":
        from models.init_attn_seq import train as train_init_attn_seq
        return train_init_attn_seq(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "conv_node":
        from models.conv_node import train as train_conv_node
        return train_conv_node(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "initlayer_hd":
        from models.initlayer_hd import train as train_initlayer_hd
        return train_initlayer_hd(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "initlayer_hdt":
        from models.initlayer_hdt import train as train_initlayer_hdt
        return train_initlayer_hdt(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "edgeemb_attn":
        from models.edgeemb_attn import train as train_edgeemb_attn
        return train_edgeemb_attn(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)

def train_oneshot(model : diffmodel, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only = False):
    if model_type not in MODEL_NAMES:
        raise ValueError(f"Model type {model_type} not found. Available types: {MODEL_NAMES}")
    if model_type == "ablated_embeddings":
        from models.ablated_embeddings import train_oneshot as train_oneshot_ablated
        return train_oneshot_ablated(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "attn_h_V":
        from models.attn_h_V import train_oneshot as train_oneshot_attn_h_V
        return train_oneshot_attn_h_V(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "declayer_attn_h_V":
        from models.declayer_attn_h_V import train_oneshot as train_oneshot_declayer_attn_h_V
        return train_oneshot_declayer_attn_h_V(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "attn_h_EXVV":
        from models.attn_h_EXVV import train_oneshot as train_oneshot_attn_h_EXVV
        return train_oneshot_attn_h_EXVV(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "twolayer_gelu":
        from models.twolayer_gelu import train_oneshot as train_oneshot_twolayer_gelu
        return train_oneshot_twolayer_gelu(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "parallel_threetensor":
        from models.parallel_threetensor import train_oneshot as train_oneshot_parallel_threetensor
        return train_oneshot_parallel_threetensor(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "inc_layer":
        from models.inc_layer import train_oneshot as train_oneshot_inc_layer
        return train_oneshot_inc_layer(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "norm_layer":
        from models.norm_layer import train_oneshot as train_oneshot_norm_layer
        return train_oneshot_norm_layer(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "attn_integ":
        from models.attn_integ import train_oneshot as train_oneshot_attn_integ
        return train_oneshot_attn_integ(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "initlayer":
        from models.initlayer import train_oneshot as train_oneshot_initlayer
        return train_oneshot_initlayer(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "attn_seq":
        from models.attn_seq import train_oneshot as train_oneshot_attn_seq
        return train_oneshot_attn_seq(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "init_attn_seq":
        from models.init_attn_seq import train_oneshot as train_oneshot_init_attn_seq
        return train_oneshot_init_attn_seq(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "conv_node":
        from models.conv_node import train_oneshot as train_oneshot_conv_node
        return train_oneshot_conv_node(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "initlayer_hd":
        from models.initlayer_hd import train_oneshot as train_oneshot_initlayer_hd
        return train_oneshot_initlayer_hd(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "initlayer_hdt":
        from models.initlayer_hdt import train_oneshot as train_oneshot_initlayer_hdt
        return train_oneshot_initlayer_hdt(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    elif model_type == "edgeemb_attn":
        from models.edgeemb_attn import train_oneshot as train_oneshot_edgeemb_attn
        return train_oneshot_edgeemb_attn(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only)
    

def featurize_dicts(outputs, device):
    with torch.no_grad():
    # Get dicts = outputs[..., 0]
        dicts = (output[0] for output in outputs)
        icodes = [output[3] for output in outputs]  # Can't use a generator here :(
        chain_mask = torch.tensor(
            np.array(
                [
                    item in outputs[0][0]["chain_letters"]
                    for item in outputs[0][0]["chain_letters"]
                ],
                dtype=np.int32,
            ), device=device
        )
        omit_AA = torch.tensor(
            np.array([False for AA in alphabet]).astype(np.float32), device=device
        )

        bias_AA = torch.zeros([21], dtype=torch.float32, device=device)
        randn = None
        output_dicts = []
        for i, dict in enumerate(dicts):
            dict["chain_mask"] = chain_mask
            output_dicts.append(featurize(dict, 
                                cutoff_for_score=8.0, 
                                use_atom_context=1, 
                                number_of_ligand_atoms=atom_context_num,
                                model_type="ligand_mpnn"
            ))
            output_dicts[i]["batch_size"] = 1
            B, L, _, _ = output_dicts[i]["X"].shape  # batch size should be 1 for now.
            output_dicts[i]["temperature"] = 0.05
            R_idx_list = list(output_dicts[i]["R_idx"].cpu().numpy())  # residue indices
            chain_letters_list = list(dict["chain_letters"])  # chain letters
            encoded_residues = []
            for j, R_idx_item in enumerate(R_idx_list):
                tmp = str(chain_letters_list[j]) + str(R_idx_item) + icodes[i][j]
                encoded_residues.append(tmp)

            bias_AA_per_residue = torch.zeros(
                [len(encoded_residues), 21], dtype=torch.float32, device=device
            )

            omit_AA_per_residue = torch.zeros(
                [len(encoded_residues), 21], dtype=torch.float32, device=device
            )

            output_dicts[i]["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )

            output_dicts[i]["symmetry_residues"] = [[]]
            output_dicts[i]["symmetry_weights"] = [[]]

            if randn == None:
                randn = torch.randn(
                        [output_dicts[i]["batch_size"], output_dicts[i]["mask"].shape[1]], device=device
                )
            output_dicts[i]["randn"] = randn
    return output_dicts

NMR_V2_DATA_PATH = "training/nmr_train.txt"
NMR_V2_TEST_DATA_PATH = "training/nmr_test.txt"

def load_pdb_id(pdbid, generator):
    '''
    Loads the corresponding PDB file and returns the data, downloading it if necessary
    '''
    # Check if the file exists
    if not os.path.exists(PDB_PATH + pdbid.lower() + ".pdb"):
        # Download the PDB file
        result = download_one(pdbid, verbose=False)
        if not result:
            print(f"Failed to download {pdbid}. Skipping.")
            return None
    # Load the PDB file
    pdb_path = PDB_PATH + pdbid.lower() + ".pdb"
    feature_dict_list = get_dicts_from_pdb(pdb_path, device)
    total_num_states = len(feature_dict_list)
    if total_num_states < MIN_STATES:
        print(f"Not enough states in NMR data for PBD {pdbid}")
        return None
    num_states = generator.integers(MIN_STATES, MAX_STATES + 1)
    selected_states = generator.choice(total_num_states, num_states, replace=False) if num_states < total_num_states else np.arange(total_num_states)
    selected_states = [feature_dict_list[i] for i in selected_states]
    selected_states = featurize_dicts(selected_states, device)

    return selected_states

# Inside train_pipeline() function, add initializations at the start
def train_pipeline():
    global LOAD_PREV
    # Initialize step weights uniformly
    step_weights = torch.ones(NUM_DIFF_STEPS, device=device) / NUM_DIFF_STEPS

    gen = np.random.default_rng()
    with open(NMR_V2_DATA_PATH, "r") as f:
        trainlines = f.readlines()
    trainlines = [line.strip().lower() for line in trainlines]

    with open(NMR_V2_TEST_DATA_PATH, "r") as f:
        testlines = f.readlines()
    testlines = [line.strip().lower() for line in testlines]
    for p in trainable_params:
        p._prev = p.clone()  # snapshot of the weights

    subset_val = gen.choice(len(testlines), 5, replace=False)
    feature_dict_list_train = [[load_pdb_id(testlines[i].strip(), gen), testlines[i].strip()] for i in subset_val]
    hard = np.array([90,91,92,93,94,95,96,97,98,99])

    for epoch in range(EPOCHS):
        try:
            # NEW: Early exit for TEST_WEIGHT
            if TEST_WEIGHT and epoch >= 2:  # Exit after 2 epochs (0 and 1)
                print("TEST_WEIGHT is True: Exiting after 2 epochs.")
                break
            # NMR data
            epoch_loss = 0
            # NEW: Dictionary to track total loss and count per step for the epoch
            step_loss_dict = {step: {'total_loss': 0.0, 'count': 0} for step in range(NUM_DIFF_STEPS)}
            # Shuffle the indices (objectively terrible way of doing this)
            subset = gen.choice(len(trainlines), NUM_TRAIN_PER_EPOCH, replace=False)
            num = 0
            optimizer.zero_grad()
            for i in (pbar := tqdm.tqdm(subset)):
                num += 1
                pdb_name = trainlines[i].strip()
                feature_dict_list = load_pdb_id(trainlines[i].strip(), gen)
                if feature_dict_list is None:
                    num -= 1
                    continue
                # Get sequence length
                seq_len = feature_dict_list[0]["X"].shape[1]
                if seq_len * len(feature_dict_list) > MAX_TOTAL_SEQ_LEN:
                    # Check if we can reduce the number of states
                    if seq_len * 2 > MAX_TOTAL_SEQ_LEN:
                        print(f"Please remove {pdb_name} from the training set, too long")
                        continue
                    else:
                        print(f"Reducing states for {pdb_name} to fit in memory")
                        feature_dict_list = feature_dict_list[:2]
                # NEW: Sample diff_levels based on dynamic weights (replace old selection logic)
                # Ensure step_weights are normalized and on CPU for sampling
                normalized_weights = (step_weights / step_weights.sum()).cpu().numpy()
                diff_levels = gen.choice(NUM_DIFF_STEPS, BATCH_SIZE, replace=True, p=normalized_weights)
                
                # Train on the selected states
                try:
                    out_dict = train(model, feature_dict_list, loss_fn, optimizer, diff_levels, device)
                    state_loss = out_dict["loss"]
                    # NEW: Aggregate per-step losses
                    step_losses = out_dict.get("step_losses", [])  # List of losses per diff_level
                    for idx, step in enumerate(diff_levels):
                        if idx < len(step_losses):
                            step_loss_dict[step]['total_loss'] += step_losses[idx]
                            step_loss_dict[step]['count'] += 1
                except torch.OutOfMemoryError:
                    print(f"Out of memory for {pdb_name}, please remove from training set")
                    num -= 1
                    continue
                epoch_loss += state_loss
                pbar.set_description(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss/(num):.4f}")
            # Train 
            # Divide gradients by the number of sequences
            if num == 0:
                print("No sequences to train on, skipping epoch")
                continue
            for param in trainable_params:
                if False and param.grad is not None:
                    param.grad.mul_(1.0 / num)
            #optimizer.step()
            with torch.no_grad():
                delta = 0.0
                total = 0.0
                for p in trainable_params:
                    if p.grad is None:                # should NOT happen except for encoder
                        continue
                    delta += (p - p._prev).pow(2).sum()   # L2 change
                    total += p.pow(2).sum()
                    p._prev.copy_(p)                      # update the snapshot
                print(f"relative weight-change  {torch.sqrt(delta/total):.3e}")

            # NEW: Compute w_new and update step_weights after training loop
            used_steps = [step for step, data in step_loss_dict.items() if data['count'] > 0]
            w_new = torch.zeros(NUM_DIFF_STEPS, device=device)
            if used_steps:
                mean_w_new = 0.0
                for step in used_steps:
                    avg_loss = step_loss_dict[step]['total_loss'] / step_loss_dict[step]['count']
                    expected_unmasked = 1.0 - betas[step].item()  # Expected % unmasked
                    w_new[step] = avg_loss * expected_unmasked
                    mean_w_new += w_new[step].item()
                mean_w_new /= len(used_steps)
                for step in range(NUM_DIFF_STEPS):
                    if step not in used_steps:
                        w_new[step] = mean_w_new  # Assign mean for unused steps
            else:
                # Fallback if no steps were used (rare)
                w_new = torch.ones(NUM_DIFF_STEPS, device=device) / NUM_DIFF_STEPS

            # Print for TEST_WEIGHT
            if TEST_WEIGHT:
                print(f"Epoch {epoch + 1}: w_t (previous step_weights): {step_weights.cpu().numpy()}")
                print(f"Epoch {epoch + 1}: w_new: {w_new.cpu().numpy()}")
            
            # Apply momentum and normalize
            step_weights = MOMENTUM * step_weights + (1 - MOMENTUM) * w_new
            step_weights = step_weights / step_weights.sum()  # Normalize to probabilities
            
            if TEST_WEIGHT:
                print(f"Epoch {epoch + 1}: w_{{t+1}} (updated step_weights): {step_weights.cpu().numpy()}")

            # Testing
            test_loss = 0
            orig_test_loss = 0
            # Pick random PDBs from the test set
            num=0
            for feature_dict_list, pdb_name in feature_dict_list_train:
                diff_levels = gen.choice(NUM_DIFF_STEPS, BATCH_SIZE, replace=False)
                num+=1
                if feature_dict_list is None:
                    num -= 1
                    continue
                
                # Train on the selected states
                try:
                    out_dict = train(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only=True)
                    state_loss = out_dict["loss"]
                    orig_loss = out_dict["loss_orig"]
                except torch.OutOfMemoryError:
                    print(f"Out of memory for {pdb_name}, please remove from test set")
                    num -= 1
                    continue
                test_loss += state_loss
                orig_test_loss += orig_loss
            print(f"Test Loss: {test_loss/num:.4f} Orig Loss: {orig_test_loss/num:.4f}")
            if USE_SCHEDULER:
                scheduler.step(test_loss / num)
        except AttributeError:
            print(f"Protein {pdb_name} failed, please remove from training or test set")
        except KeyboardInterrupt:
            print("Training interrupted")
            break
        
    if not TEST_WEIGHT:
        # Save the model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "training/multistate_diff_checkpoints/" + model_type + "_checkpoint.pt",
        )
        
        # After training, calculate a loss curve by timestep and display it
        losses = []
        for timestep in range(NUM_DIFF_STEPS):
            ts_loss = 0
            for feature_dict_list, pdb_name in feature_dict_list_train:
                try:
                    out_dict = train(model, feature_dict_list, loss_fn, optimizer, [timestep], device, eval_only=True)
                    ts_loss += out_dict["loss"]
                except torch.OutOfMemoryError:
                    print(f"Out of memory for {pdb_name}, please remove from test set")
                    continue
            losses.append(ts_loss / len(feature_dict_list_train))
    
        # Plot the loss curve
        import matplotlib.pyplot as plt
        # Create the base plot
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Diffusion timestep")
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.plot(range(NUM_DIFF_STEPS), losses, color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        # Create a second y-axis for the betas
        ax2 = ax1.twinx()
        ax2.set_ylabel("Beta", color="tab:red")
        ax2.plot(range(NUM_DIFF_STEPS), betas[:NUM_DIFF_STEPS], color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        # Show the plot
        plt.title("Loss curve by timestep")
        plt.show()
    else:
        print("TEST_WEIGHT is True: Skipping loss curve display.")


if __name__ == "__main__":
    train_pipeline()