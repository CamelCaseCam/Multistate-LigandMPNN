# diffuse.py

import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from train_v3 import ProteinMPNN, featurize_dicts, get_dicts_from_pdb, categorical_diffuse, MAX_STATES, MIN_STATES
from model_utils import cat_neighbors_nodes
from data_utils import alphabet, restype_int_to_str
from models.attn_integ import diffmodel

ORIG_MODEL_PATH = "model_params/ligandmpnn_v_32_030_25.pt"
MODEL_PATH = "training/multistate_diff_checkpoints/attn_integ_checkpoint.pt"
TEST_PATH = "training/full_tests/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
NUM_DIFF_STEPS = 100

# Pre-compute diffusion parameters
t = torch.arange(NUM_DIFF_STEPS + 1, dtype=torch.float32) / NUM_DIFF_STEPS
alphas = torch.cos((t + 0.008) / 1.008 * torch.pi/2) ** 2
#betas = 1 - alphas[1:] / alphas[:-1]
betas = t
w = torch.sqrt(betas)
w = w * (betas.mean() / w.mean())

def load_pdb_from_file(pdb_path, generator, device):
    feature_dict_list = get_dicts_from_pdb(pdb_path, device)
    total_num_states = len(feature_dict_list)
    if total_num_states < MIN_STATES:
        print(f"Not enough states in NMR data for {pdb_path}")
        return None
    max_states = min(MAX_STATES, total_num_states)
    num_states = generator.integers(MIN_STATES, max_states + 1)
    selected_states = generator.choice(total_num_states, num_states, replace=False)
    selected_states = [feature_dict_list[i] for i in selected_states]
    selected_states = featurize_dicts(selected_states, device)

    return selected_states

def diffuse_step(model, h_V, h_EXV, S_t, t, temperature=0.1):
    """
    Perform one step of diffusion denoising.
    
    Args:
        model: diffmodel instance
        embeddings: Tensor (B, L, H) of pre-computed structural embeddings
        S_t: Tensor (B, L) of current sequence tokens
        t: Tensor (B,) of current timestep indices
        temperature: float, sampling temperature
    
    Returns:
        S_next: Tensor (B, L) of next sequence tokens
        probs: Tensor (B, L, 20) of sampling probabilities
        log_probs: Tensor (B, L, 21) of log probabilities
    """
    L = S_t.shape[1]

    # Embed the sequence and timestep
    x_self_emb = model.seq_enc_layer(S_t.long())
    pos = torch.arange(L, device=S_t.device)         # (L,)
    pos_emb = model.pos_embed(pos)                 # (L, H)
    t_self_vec = model.t_embed(torch.tensor([t], device=S_t.device))  # (1, H)

    # Set up initial values
    seq_tensor = x_self_emb + pos_emb
    node_tensor = h_V
    chemical_env_tensor = h_EXV
    with torch.no_grad():
        # Process through decoder layers
        for layer in model.decoder_layers:
            seq_tensor = seq_tensor + t_self_vec.unsqueeze(1)
            seq_tensor, node_tensor, chemical_env_tensor = layer(seq_tensor, node_tensor, chemical_env_tensor)
        
        # Get logits
        logits = model.W_out(torch.cat([seq_tensor, node_tensor, chemical_env_tensor], dim=-1))
        
        # Compute probabilities
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)  # (B, L, 21)
        probs_sample = probs[:, :, :20] / torch.sum(probs[:, :, :20], dim=-1, keepdim=True)  # (B, L, 20)
        S_next = torch.multinomial(probs_sample.view(-1, 20), 1).view(S_t.shape)  # (B, L)
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, L, 21)
        
        return S_next, probs_sample, log_probs

def compute_consensus(probs, threshold=0.85):
    """Compute consensus sequence from probabilities."""
    max_probs, max_idx = probs.max(dim=-1)
    consensus = torch.where(max_probs > threshold, max_idx, torch.tensor(-1, device=probs.device))
    return "".join(restype_int_to_str.get(aa.item(), "-") for aa in consensus[0])

def compute_entropy(probs):
    """Compute Shannon entropy of probability distribution."""
    W_i = -torch.sum(probs * torch.log2(probs + 1e-8), dim=1)
    return torch.sum(W_i).item()

def sequence_recovery(pred_seq, true_seq):
    """Compute sequence recovery rate."""
    return (pred_seq == true_seq).float().mean().item()

def diffusion_pipeline(model, feature_dict_list, temperatures, score_seq, seed=None, start_step=0, num_steps=100, debug=False):
    """
    Run diffusion pipeline over multiple states.
    
    Args:
        model: diffmodel instance
        feature_dict_list: List of feature dictionaries for multiple states
        temperatures: float or list of floats for sampling temperature
        score_seq: bool, whether to compute sequence recovery
        seed: int or None, for reproducibility
        start_step: int, starting diffusion step
        num_steps: int, number of diffusion steps
        debug: bool, whether to return debug statistics
    
    Returns:
        final_seq: str, final generated sequence
        stats: list of [consensus_seq, seq_recovery] per step
        debug_stats: list of [probs, entropy, beta, input_seq, temp] per step (if debug=True)
    """
    # Set up reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()

    # Handle temperature input
    if isinstance(temperatures, (int, float)):
        temp_list = [float(temperatures)] * num_steps
    else:
        temp_list = list(temperatures) + [temperatures[-1]] * (num_steps - len(temperatures))

    # Pre-compute embeddings
    num_states = len(feature_dict_list)
    h_V_list, h_E_list, E_idx_list = [], [], []
    for feature_dict in feature_dict_list:
        h_V, h_E, E_idx = model.orig_model.encode(feature_dict)
        h_V_list.append(h_V)
        h_E_list.append(h_E)
        E_idx_list.append(E_idx)
    h_V = torch.cat(h_V_list, dim=0).to(DEVICE)

    B, L = feature_dict_list[0]["S"].shape
    S_true = feature_dict_list[0]["S"].to(DEVICE)
    mask = feature_dict_list[0]["mask"].to(DEVICE)
    chain_mask = feature_dict_list[0]["chain_mask"].to(DEVICE)
    randn = torch.randn([B, L], generator=generator).to(DEVICE)

    # Compute decoding order and masks
    chain_mask = mask * chain_mask
    decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))
    E_idx_base = E_idx_list[0]
    permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=L).float()
    order_mask_backward = torch.einsum("ij, biq, bjp->bqp", (1 - torch.triu(torch.ones(L, L, device=DEVICE))), permutation_matrix_reverse, permutation_matrix_reverse)
    mask_attend = torch.gather(order_mask_backward, 2, E_idx_base).unsqueeze(-1)
    mask_1D = mask.view([B, L, 1, 1])
    mask_fw = mask_1D * (1.0 - mask_attend)

    # Compute embeddings
    h_S = torch.zeros((B, L, model.hidden_dim), device=DEVICE)
    h_EXV_encoder_fw_list = []
    for state_idx in range(num_states):
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E_list[state_idx], E_idx_list[state_idx])
        h_EXV_encoder = cat_neighbors_nodes(h_V_list[state_idx], h_EX_encoder, E_idx_list[state_idx])
        h_EXV_encoder_fw_list.append((mask_fw * h_EXV_encoder).detach())
    h_EXV_encoder_fw = torch.cat(h_EXV_encoder_fw_list, dim=0).detach()
    # Get embeddings using h_V processed with h_EXV
    h_V, h_EXV = model.integrate_embeddings(h_V.detach(), h_EXV_encoder_fw.detach())

    # Initialize sequence
    S_t = torch.full((B, L), 20, dtype=torch.long, device=DEVICE)  # Start with all 'X'
    true_seq = S_true[0]

    stats = []
    debug_stats = [] if debug else None

    # Diffusion process
    for step in range(NUM_DIFF_STEPS - 1 - start_step, -1, -1):
        t = torch.full((B,), step, device=DEVICE)
        
        # Noise the sequence if not at final step
        if step > 0:
            beta = betas[t.to("cpu")]
            keep_mask = torch.bernoulli(
                (1.0 - beta.unsqueeze(1)).expand_as(S_t), generator=generator
            ).bool().to(DEVICE)
            replacement_tokens = torch.full_like(S_t, 20) # 20 = 'X'   
            S_t = torch.where(keep_mask, S_t, replacement_tokens)  # (B, L) 

        
        # Denoise
        S_next, probs, log_probs = diffuse_step(model, h_V, h_EXV, S_t, t, temp_list[step - start_step])
        S_t = S_next

        # Compute stats
        consensus = compute_consensus(probs)
        seq_rec = sequence_recovery(S_t, true_seq) if score_seq else None
        stats.append([consensus, seq_rec] if score_seq else [consensus])

        if debug:
            entropy = compute_entropy(probs)
            beta = betas[step].item()
            input_seq = "".join(restype_int_to_str[aa.item()] for aa in S_t[0])
            debug_stats.append([probs.cpu(), entropy, beta, input_seq, temp_list[step - start_step]])

    # Convert final sequence to string
    final_seq = "".join(restype_int_to_str[aa.item()] for aa in S_t[0])

    return (final_seq, stats, debug_stats) if debug else (final_seq, stats)

def test_diffusion():
    """Test diffusion pipeline on examples in full_tests folder."""
    # Load model
    orig_checkpoint = torch.load(ORIG_MODEL_PATH, map_location=DEVICE)
    orig_model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=orig_checkpoint["num_edges"],
        atom_context_num=orig_checkpoint["atom_context_num"],
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=True,
        device=DEVICE,
    )
    orig_model.load_state_dict(orig_checkpoint["model_state_dict"])
    model = diffmodel(orig_model)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Load test examples
    test_files = [f for f in os.listdir(TEST_PATH) if f.endswith('.pdb')]
    gen = np.random.default_rng()  # Fixed seed for reproducibility

    seq_rec_all = np.zeros(NUM_DIFF_STEPS)
    entropy_all = np.zeros(NUM_DIFF_STEPS)

    for test_file in test_files:
        # Get feature dicts
        test_path = os.path.join(TEST_PATH, test_file)
        feature_dict_list = load_pdb_from_file(test_path, gen, DEVICE)

        # Run pipeline
        seq, stats, debug_stats = diffusion_pipeline(
            model,
            feature_dict_list,
            temperatures=[0.5] * 100,
            score_seq=True,
            debug=True
        )

        seq_rec = [stat[1] for stat in stats]
        entropy = [stat[1] for stat in debug_stats]
        seq_rec_all += np.array(seq_rec)
        entropy_all += np.array(entropy)
        print(f"File: {test_file}")
        print(f"Final Sequence: {seq}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(range(NUM_DIFF_STEPS), seq_rec_all / len(test_files), label="Sequence Recovery", color='blue')
    ax2.plot(range(NUM_DIFF_STEPS), entropy_all / len(test_files), label="Shannon Entropy", color='orange')

    ax1.set_title("Sequence Recovery per Step")
    ax1.set_xlabel("Diffusion Step")
    ax1.set_ylabel("Sequence Recovery")
    ax1.legend()

    ax2.set_title("Shannon Entropy per Step")
    ax2.set_xlabel("Diffusion Step")
    ax2.set_ylabel("Entropy")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_diffusion()