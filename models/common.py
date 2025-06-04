import torch
import math

from model_utils import cat_neighbors_nodes

AMINO_DIM = 21
MASK_ID = 21
TOTAL_DIM = 22

NUM_DIFF_STEPS = 100

t = torch.arange(NUM_DIFF_STEPS + 1, dtype=torch.float32) / NUM_DIFF_STEPS
alphas = torch.cos((t + 0.008) / 1.008 * math.pi/2) ** 2
#betas  = 1 - alphas[1:] / alphas[:-1]
#betas = t
# Sine betas
betas = (torch.sin((t) * math.pi / 2) ** 2).detach()
w = torch.sqrt(betas)
w = w * (betas.mean() / w.mean())


def categorical_diffuse(x0, t_idx, rng, device):
    """
    x0     : (B, L) ground-truth integer tokens (0‥TOTAL_DIM-1)
    t_idx  : scalar or (B,)   current timestep 0‥T-1
    rng    : torch.Generator   (to keep things reproducible)

    returns x_t  : (B, L) corrupted tokens
    """
    beta  = betas[t_idx]                      # scalar
    keep_mask = torch.bernoulli(
        (1.0 - beta.unsqueeze(1)).expand_as(x0), generator=rng
    ).bool().to(device)                                 # True -> keep original

    # draw replacement tokens (uniform over TOTAL_DIM)
    rand_tokens = torch.randint(
        0, 21, x0.shape, generator=rng
    ).to(device)

    # apply corruption
    x_t = torch.where(keep_mask, x0, rand_tokens)
    return x_t

def train_oneshot_common(model, feature_dict_list, diff_levels, device):
    """Common preprocessing for all train_oneshot implementations"""
    if not feature_dict_list or not isinstance(feature_dict_list, list):
        raise ValueError("feature_dict_list must be a non-empty list")
    
    num_states = len(feature_dict_list)
    if num_states < 2:
        raise ValueError("At least two states must be provided")
    
    # Use first feature dict to get basic parameters
    B_decoder = len(feature_dict_list)
    S_true = feature_dict_list[0]["S"]
    mask = feature_dict_list[0]["mask"]
    chain_mask = feature_dict_list[0]["chain_mask"]
    
    # Convert mask and chain mask to (len(feature_dict_list), L) shape
    mask = mask.repeat(B_decoder, 1)
    chain_mask = chain_mask.repeat(B_decoder, 1)

    bias = feature_dict_list[0]["bias"]
    randn = feature_dict_list[0]["randn"]
    temperature = feature_dict_list[0]["temperature"]

    B, L = S_true.shape
    state_num = B_decoder
    
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

    # Initialize separate h_V stacks for each state
    h_V_stack = [h_V]
    h_E = torch.cat(h_E_list, dim=0)      # (num_states, L, k, H)
    E_idx = torch.cat(E_idx_list, dim=0)  # (num_states, L, k)

    # Pre-compute encoder embeddings
    h_EXV_encoder_fw_list = []
    for state_idx in range(num_states):
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(torch.zeros((B_decoder, L, model.hidden_dim), device=device)), 
                                          h_E[state_idx].unsqueeze(0), 
                                          E_idx[state_idx].unsqueeze(0))
        h_EXV_encoder = cat_neighbors_nodes(h_V_stack[0][state_idx].unsqueeze(0), 
                                           h_EX_encoder, 
                                           E_idx[state_idx].unsqueeze(0))
        h_EXV_encoder_fw_list.append((mask_fw[state_idx:state_idx+1] * h_EXV_encoder).detach())
    h_EXV_encoder_fw = torch.cat(h_EXV_encoder_fw_list, dim=0).detach()
    
    return {
        "S_true": S_true,
        "mask": mask,
        "chain_mask": chain_mask,
        "bias": bias,
        "h_V": h_V,
        "h_E": h_E,
        "E_idx": E_idx,
        "h_EXV_encoder_fw": h_EXV_encoder_fw,
        "B": B,
        "L": L,
        "diff_levels": diff_levels,
        "device": device
    }

def diffusion_step_common(model, embeddings, diff_levels, S_true, device, use_pos_enc=False):
    """Common diffusion step for all implementations"""
    # Repeat embeddings for batches
    embeddings = embeddings.repeat(len(diff_levels), 1, 1)
    L = embeddings.shape[1]
    
    # Now, we can start the diffusion process
    # First, encode the ground truth sequence
    true_seq = S_true.long()
    
    # Generate noise
    gen = torch.Generator()
    
    # Apply diffusion
    self_diff_levels = torch.tensor(diff_levels, device="cpu")
    self_diff_levels = torch.clamp(self_diff_levels, 0, NUM_DIFF_STEPS - 1)
    x = categorical_diffuse(true_seq.repeat(len(diff_levels), 1), self_diff_levels, gen, device)
    x_tm1 = x.clone()
    self_diff_levels = self_diff_levels.to(device)
    
    # Generate a new sequence from noise and feed it into the model
    x_self_emb = model.seq_enc_layer(x.long())
    if use_pos_enc:
        pos = torch.arange(L, device=x.device)         # (L,)
        pos_emb = model.pos_embed(pos)                 # (L, H)
        x_self_emb = x_self_emb + pos_emb              # broadcast over batch
    t_self_vec = model.t_embed(self_diff_levels)
    
    # Process through decoder layers
    for layer in model.decoder_layers:
        x_self_emb = x_self_emb + t_self_vec.unsqueeze(1)
        x_self_emb = layer(x_self_emb, embeddings)
    
    # Get logits
    logits = model.W_out(x_self_emb)
    
    return logits, x_tm1

def categorical_diffuse_mask(x0, t_idx, rng, device):
    """
    A diffusion process that masks tokens with 'X' (20) instead of random amino acids.
    
    x0     : (B, L) ground-truth integer tokens (0‥20)
    t_idx  : scalar or (B,)   current timestep 0‥T-1
    rng    : torch.Generator   (to keep things reproducible)

    returns:
        x_t  : (B, L) corrupted tokens with some positions masked as 'X' (20)
        mask : (B, L) boolean mask indicating which positions were masked (True = masked)
    """
    beta = betas[t_idx]                      # scalar
    keep_mask = torch.bernoulli(
        (1.0 - beta.unsqueeze(1)).expand_as(x0), generator=rng
    ).bool().to(device)                      # True -> keep original, False -> mask

    # Make sure at least one position is masked in each sequence
    for i in range(x0.shape[0]):
        if keep_mask[i].sum() == x0.shape[1]:
            idx = (i, *torch.randint(x0.shape[1], (1,), generator=rng))
            keep_mask[idx] = False
    
    # Replace non-kept positions with 'X' (20)
    mask_token = torch.full_like(x0, 20)
    x_t = torch.where(keep_mask, x0, mask_token)
    
    # Return the corrupted sequence and the mask of which positions were corrupted
    return x_t, ~keep_mask  # invert keep_mask to get mask_positions