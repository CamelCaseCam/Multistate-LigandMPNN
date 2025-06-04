# models/initlayer.py
import torch
import math
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

from model_utils import cat_neighbors_nodes, DecLayer

from .common import TOTAL_DIM, NUM_DIFF_STEPS, train_oneshot_common, categorical_diffuse_mask, categorical_diffuse, betas

class InitLayer(torch.nn.Module):
    def __init__(self, hidden_dim, rank=16, attn_impl="flash"):
        super().__init__()
        self.h = hidden_dim
        self.r = rank
        self.attn_impl = attn_impl

        # Linear + GELU layer to update node tensor from sequence
        self.node_update_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        # Linear + GELU layer to update chemical environment tensor from sequence
        self.chem_env_update_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sequence, node, chemical_environment):
        # Update node tensor using sequence information
        node_input = torch.cat([sequence, node], dim=-1)
        node = node + self.node_update_layer(node_input)
        
        # Update chemical environment tensor using sequence information
        chem_env_input = torch.cat([sequence, chemical_environment], dim=-1)
        chemical_environment = chemical_environment + self.chem_env_update_layer(chem_env_input)
        
        # Sequence tensor remains unchanged
        return sequence, node, chemical_environment
    
class DiffLayer(torch.nn.Module):
    def __init__(self, hidden_dim, rank=16, attn_impl="flash"):  # "flash" or "xformers" or "naive"
        super().__init__()
        self.h = hidden_dim
        self.r = rank
        self.attn_impl = attn_impl

        # Linear + GELU layer to generate the sequence tensor
        self.seq_integration_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        # Removed the conv layer

        # Linear + GELU layer to generate the node tensor
        self.node_gen_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        self.num_heads = rank
        self.head_dim  = hidden_dim
        qkv_dim = rank * hidden_dim

        self.proj_q = torch.nn.Linear(hidden_dim, qkv_dim, bias=False)
        self.proj_k = torch.nn.Linear(hidden_dim, qkv_dim, bias=False)
        self.proj_v = torch.nn.Linear(hidden_dim, qkv_dim, bias=False)
        self.proj_out = torch.nn.Linear(qkv_dim, hidden_dim, bias=False)
        self.ln_attn = torch.nn.LayerNorm(hidden_dim)

        # Final linear + GELU layer to generate the new chemical environment tensor
        self.chem_env_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def _flash_attn(self, Q, K, V):
        # ... existing code ...
        B, L, r = Q.shape
        Qh = Q.transpose(1,2)        # (B,r,L)
        Kh = K.transpose(1,2)
        Vh = V.transpose(1,2)
        ctx = torch.nn.functional.scaled_dot_product_attention(
                  Qh, Kh, Vh, dropout_p=0.0)
        ctx = ctx.transpose(1,2).reshape(B, L, r)   # (B,L,r)
        return ctx

    def _xformers_attn(self, Q, K, V):
        # ... existing code ...
        if memory_efficient_attention is not None:
            B, L, _ = Q.shape
            Q4 = (Q.view(B, L, self.num_heads, self.head_dim) / math.sqrt(self.head_dim)).to(torch.float16)
            K4 = (K.view(B, L, self.num_heads, self.head_dim) / math.sqrt(self.head_dim)).to(torch.float16)
            V4 = V.view(B, L, self.num_heads, self.head_dim).to(torch.float16)
            ctx = memory_efficient_attention(Q4, K4, V4)          # (B,L,h,d)
            ctx = ctx.view(B, L, -1).to(Q.dtype)                  # (B,L,h*d)
            return ctx
        else:
            return self._naive_attn(Q, K, V)
    
    def _naive_attn(self, Q, K, V):
        # ... existing code ...
        B, L, r = Q.shape
        Qh = Q.transpose(1,2)        # (B,r,L)
        Kh = K.transpose(1,2)
        Vh = V.transpose(1,2)

        ctx = torch.bmm(Qh, Kh.transpose(1,2)) / math.sqrt(r)
        ctx = torch.nn.functional.softmax(ctx, dim=-1)
        ctx = torch.bmm(ctx, Vh)
        ctx = ctx.transpose(1,2).reshape(B, L, r)   # (B,L,r)
        return ctx

    def forward(self, sequence, node, chemical_environment):
        # If x is on CPU, change our attention implementation to naive
        if sequence.device.type == "cpu":
            self.attn_impl = "naive"
        
        # 1. Generate new sequence tensor (without conv layer)
        context_stacked = torch.cat([sequence, node, chemical_environment], dim=-1)
        seq_d = self.seq_integration_layer(context_stacked)
        sequence = sequence + seq_d

        # 2. Generate new node tensor
        context_stacked = torch.cat([sequence, node], dim=-1)
        node = node + self.node_gen_layer(context_stacked)

        # 3. Generate new chemical environment tensor from nodes
        # 3.1 low-rank attention (flash or xformers)
        Q, K, V = self.proj_q(chemical_environment), self.proj_k(node), self.proj_v(node)
        if self.attn_impl == "flash":
            context = self._flash_attn(Q, K, V)
        elif self.attn_impl == "xformers":
            context = self._xformers_attn(Q, K, V)
        elif self.attn_impl == "naive":
            context = self._naive_attn(Q, K, V)

        if self.attn_impl is not None:
            extracted_information = self.ln_attn(node + self.proj_out(context))

        # Integrate the new information into the chemical environment tensor
        context_stacked = torch.cat([extracted_information, chemical_environment], dim=-1)
        chemical_environment = chemical_environment + self.chem_env_layer(context_stacked)

        return sequence, node, chemical_environment
    
class AttentionIntegrator(torch.nn.Module):
    def __init__(self, seq_idx, emb_idx, hidden_dim, rank=16):
        '''
        For integrating information across a small-scale block of positions using self-attention. 

        For the EXV tensor, it maps (num_states, L, k, H) -> (num_states, L, k, H) weights, and then the nodes are combined 
        by weighted average. 

        For combined EXV or h_V, it maps (num_states, L, H) -> (num_states, L, H) weights, and then the states are combined
        by weighted average.

        Note that this is only suitable for unordered data (except for the sequence dimension, which is ignored), as it 
        does not use any positional information. We'll use vanilla nn.MultiheadAttention for this because on such a 
        small scale it doesn't make sense to use a more complex implementation.

        Also, note that no matter the input format, the output will always be (..., H) for the integrated tensor.
        '''
        super(AttentionIntegrator, self).__init__()

        self.num_heads = rank
        self.head_dim  = hidden_dim

        self.proj_q = torch.nn.Linear(hidden_dim, rank * hidden_dim, bias=False)
        self.proj_k = torch.nn.Linear(hidden_dim, rank * hidden_dim, bias=False)

        self.seq_idx = seq_idx
        self.emb_idx = emb_idx
        self.hidden_dim = hidden_dim
        self.rank = rank
    
    def forward(self, x, return_weights=False, batch_size=256):
        '''
        This layer works using the following steps:
        1. Flatten the input so seq_idx is the second-last dimension and emb_idx is the last dimension
        2. Apply the attention layer to the input
        3. Reshape the output back to the original shape (ish)
        '''
        seq_len = x.shape[self.seq_idx]
        # 1. Flatten the input so seq_idx is the second-last dimension and emb_idx is the last dimension
        # 1.1 Transpose the input to get the seq_idx and emb_idx dimensions in the right place
        x = torch.movedim(x, (self.seq_idx, self.emb_idx), (-2, -1))
        orig_shape = x.shape
        # 1.2 Flatten other dimensions
        x = x.reshape(-1, seq_len, self.hidden_dim)

        # Calculate attention weights
        q = self.proj_q(x)
        k = self.proj_k(x)
        B, L, E = q.shape 
        H = self.num_heads
        D = E // H

        # Reshape to (B, H, L, D)
        q = q.view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)
        k = k.view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)

        if B <= batch_size:
            # Scaled dot-product attention
            attn_scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)  # (B, H, L, L)
            attn_weights = attn_scores.softmax(dim=-1)             # (B, H, L, L)
        else:
            # Slice through the batch dimension
            attn_weights = torch.zeros(B, H, L, L, device=x.device)
            for i in range(0, B, batch_size):
                end = min(i + batch_size, B)
                q_i = q[i:end]
                k_i = k[i:end]
                attn_scores = (q_i @ k_i.transpose(-1, -2)) / math.sqrt(D)
                attn_scores = attn_scores.softmax(dim=-1)             # (B, H, L, L)
                attn_weights[i:end] = attn_scores

        weights = attn_weights.mean(dim=1).mean(dim=1)                   # average heads & queries
        weights = weights.softmax(dim=-1)                     # re-normalise

        pooled = torch.einsum('bi,bij->bj', weights, x)  # (B*, H)
        pooled = pooled.reshape(*orig_shape[:-2], self.hidden_dim)             # (..., H)

        if return_weights:
            weights = weights.reshape(*orig_shape[:-2], seq_len)       # (..., L)
            return pooled, weights
        return pooled
    
class diffmodel(torch.nn.Module):
    def __init__(self, orig_model, hidden_dim=256, maxlen=512, k_neighbors=32, EXV_dim=384):
        super(diffmodel, self).__init__()
        self.orig_model = orig_model
        self.hidden_dim = orig_model.hidden_dim
        self.emb_hidden_dim = hidden_dim if hidden_dim is not None else orig_model.hidden_dim
        self.k_neighbors = k_neighbors
        self.EXV_dim = EXV_dim

        # Integration components remain the same
        self.proj_h_V = torch.nn.Linear(orig_model.hidden_dim, self.emb_hidden_dim)
        self.integ_chem_env = AttentionIntegrator(seq_idx=2, emb_idx=3, hidden_dim=self.EXV_dim, rank=16)
        self.proj_chem_env = torch.nn.Linear(EXV_dim, self.emb_hidden_dim)
        self.integ_state_h_V = AttentionIntegrator(seq_idx=0, emb_idx=2, hidden_dim=self.emb_hidden_dim, rank=16)
        self.integ_state_chem_env = AttentionIntegrator(seq_idx=0, emb_idx=2, hidden_dim=self.emb_hidden_dim, rank=16)

        self.seq_enc_layer = torch.nn.Embedding(TOTAL_DIM, self.emb_hidden_dim)

        self.edge_idx_embed = torch.nn.Embedding(maxlen, self.emb_hidden_dim)

        # Add the initial layer
        self.init_layer = InitLayer(self.emb_hidden_dim, rank=16, attn_impl="xformers")
        
        # Regular decoder layers
        self.decoder_layers = torch.nn.ModuleList(
            [DiffLayer(self.emb_hidden_dim, rank=16, attn_impl="xformers") for _ in range(8)]
        )
        
        self.W_out = torch.nn.Linear(self.emb_hidden_dim * 3, 21)
        self.t_embed = torch.nn.Embedding(NUM_DIFF_STEPS, self.emb_hidden_dim)
        self.pos_embed = torch.nn.Embedding(maxlen, self.emb_hidden_dim)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, feature_dict_list):
        # Get embeddings from the original model
        assert len(feature_dict_list) > 0
        raise NotImplementedError("This model is not implemented yet")
    
    def get_weights(self, h_V, h_EXV):
        # Combine the h_V embeddings
        h_V = self.proj_h_V(h_V)  # (num_states, L, H)
        h_V_int, h_V_weights = self.integ_state_h_V(h_V, return_weights=True)
        h_V_int = h_V_int.unsqueeze(0)  # (1, L, H)
        h_V_weights = h_V_weights.unsqueeze(0)  # (1, L, H)
        # Combine the chemical environment embeddings
        chemical_env, local_chemical_env_weights = self.integ_chem_env(h_EXV, return_weights=True)
        chemical_env = self.proj_chem_env(chemical_env)  # (num_states, L, H)
        
        chemical_env_int, chemical_env_weights = self.integ_state_chem_env(chemical_env, return_weights=True)
        chemical_env_int = chemical_env_int.unsqueeze(0)  # (1, L, H)
        chemical_env_weights = chemical_env_weights.unsqueeze(0)  # (1, L, H)
        return h_V_weights, local_chemical_env_weights, chemical_env_weights
    
    def integrate_embeddings(self, h_V, h_EXV):
        # Combine the h_V embeddings
        h_V = self.proj_h_V(h_V)  # (num_states, L, H)
        h_V_int = self.integ_state_h_V(h_V, return_weights=False).unsqueeze(0)  # (1, L, H)

        chemical_env = self.integ_chem_env(h_EXV, return_weights=False)  # (num_states, L, EXV)
        chemical_env = self.proj_chem_env(chemical_env)  # (num_states, L, H)
        # Combine the chemical environment embeddings
        chemical_env_int = self.integ_state_chem_env(chemical_env, return_weights=False).unsqueeze(0)  # (1, L, H)

        return h_V_int, chemical_env_int
    
    def trainable_parameters(self):
        trainable_params = []
        trainable_params += self.integ_chem_env.parameters()
        trainable_params += self.integ_state_h_V.parameters()
        trainable_params += self.integ_state_chem_env.parameters()
        trainable_params += self.seq_enc_layer.parameters()
        trainable_params += self.edge_idx_embed.parameters() # Add edge index embedding parameters
        trainable_params += self.init_layer.parameters()  # Add init layer parameters
        trainable_params += self.decoder_layers.parameters()
        trainable_params += self.W_out.parameters()
        trainable_params += self.t_embed.parameters()
        trainable_params += self.pos_embed.parameters()
        return trainable_params
    
def train_oneshot(model : diffmodel, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only=False):
    # Get common preprocessing
    common_data = train_oneshot_common(model, feature_dict_list, diff_levels, device)
    S_true = common_data["S_true"]
    h_V = common_data["h_V"]
    h_EXV_encoder_fw = common_data["h_EXV_encoder_fw"]
    E_idx = common_data["E_idx"]
    mask = common_data["mask"]
    L = S_true.shape[1]
    
    output_dict = {}
    
    if eval_only:
        # Score actual sequence with the original model
        orig_output = model.orig_model.score(feature_dict_list[0], use_sequence=True)
        orig_logits = orig_output["logits"]
        mean_orig_loss = torch.mean(loss_fn(orig_logits.squeeze(0), S_true[0, :].long()))
        output_dict["loss_orig"] = mean_orig_loss.detach().cpu().item()
    
    optimizer.zero_grad()
    
    # Get embeddings using h_V processed with h_EXV
    h_V, h_EXV = model.integrate_embeddings(h_V.detach(), h_EXV_encoder_fw.detach())
    
    # Run diffusion step
    # Repeat embeddings for batches
    h_V = h_V.repeat(len(diff_levels), 1, 1)
    h_EXV = h_EXV.repeat(len(diff_levels), 1, 1)
    
    # Now, we can start the diffusion process
    # First, encode the ground truth sequence
    true_seq = S_true.long()
    
    # Generate noise
    gen = torch.Generator()
    
    # Apply diffusion
    self_diff_levels = torch.tensor([99] * len(diff_levels), device="cpu")
    self_diff_levels = torch.clamp(self_diff_levels, 0, NUM_DIFF_STEPS - 1)
    x_true = true_seq.repeat(len(diff_levels), 1)
    x = torch.randint(
        0, 21, x_true.shape, generator=gen
    ).to(device)
    x_tm1 = x.clone()
    self_diff_levels = self_diff_levels.to(device)
    
    # Generate a new sequence from noise and feed it into the model
    x_self_emb = model.seq_enc_layer(x.long())
    pos = torch.arange(L, device=x.device)         # (L,)
    pos_emb = model.pos_embed(pos)                 # (L, H)
    t_self_vec = model.t_embed(self_diff_levels)

    # Get edge embeddings
    E_idx = E_idx.detach()
    E_idx_emb = model.edge_idx_embed(E_idx)         # (num_states, L, k, H)
    E_idx_emb = E_idx_emb.sum(dim=2)                # (num_states, L, H)
    E_idx_emb = E_idx_emb.sum(dim=0).unsqueeze(0)   # (1, L, H)

    # Set up initial values
    seq_tensor = x_self_emb + pos_emb
    node_tensor = h_V + E_idx_emb
    chemical_env_tensor = h_EXV
    
    # First apply the init layer to broadcast sequence info to node and chem env
    seq_tensor, node_tensor, chemical_env_tensor = model.init_layer(seq_tensor, node_tensor, chemical_env_tensor)
    
    # Process through decoder layers
    for layer in model.decoder_layers:
        seq_tensor = seq_tensor + t_self_vec.unsqueeze(1)
        seq_tensor, node_tensor, chemical_env_tensor = layer(seq_tensor, node_tensor, chemical_env_tensor)
    
    # Get logits
    logits = model.W_out(torch.cat([seq_tensor, node_tensor, chemical_env_tensor], dim=-1))
    
    # Calculate loss
    loss = torch.zeros(1, device=device)
    for i in range(len(diff_levels)):
        logits_i = logits[i]
        tar_i = S_true[0].long()
        ce_t = loss_fn(logits_i, tar_i)
        loss += ce_t.mean()
    
    if not eval_only:
        loss.backward()
        optimizer.step()
    
    output_dict["loss"] = loss.detach().cpu().item() / len(diff_levels)
    
    return output_dict
    
def train(model: diffmodel, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only=False):
    # Get common preprocessing
    common_data = train_oneshot_common(model, feature_dict_list, diff_levels, device)
    S_true = common_data["S_true"]
    h_V = common_data["h_V"]
    h_EXV_encoder_fw = common_data["h_EXV_encoder_fw"]
    E_idx = common_data["E_idx"]
    mask = common_data["mask"]
    L = S_true.shape[1]
    
    output_dict = {}
    
    if eval_only:
        # Score actual sequence with the original model
        orig_output = model.orig_model.score(feature_dict_list[0], use_sequence=True)
        orig_logits = orig_output["logits"]
        mean_orig_loss = torch.mean(loss_fn(orig_logits.squeeze(0), S_true[0, :].long()))
        output_dict["loss_orig"] = mean_orig_loss.detach().cpu().item()
    
    optimizer.zero_grad()
    
    # Get embeddings using h_V processed with h_EXV
    h_V, h_EXV = model.integrate_embeddings(h_V.detach(), h_EXV_encoder_fw.detach())
    
    # Repeat embeddings for batches
    h_V = h_V.repeat(len(diff_levels), 1, 1)
    h_EXV = h_EXV.repeat(len(diff_levels), 1, 1)
    
    # Now, we can start the diffusion process
    # First, encode the ground truth sequence
    true_seq = S_true.long()
    
    # Generate noise
    gen = torch.Generator()
    
    # Apply diffusion with masking
    self_diff_levels = torch.tensor(diff_levels, device="cpu")
    self_diff_levels = torch.clamp(self_diff_levels, 0, NUM_DIFF_STEPS - 1)
    x_true = true_seq.repeat(len(diff_levels), 1)
    
    # Use the new masking diffusion process
    x, mask_positions = categorical_diffuse_mask(x_true, self_diff_levels, gen, device)
    self_diff_levels = self_diff_levels.to(device)
    
    # Generate a new sequence from masked input and feed it into the model
    x_self_emb = model.seq_enc_layer(x.long())
    pos = torch.arange(L, device=x.device)         # (L,)
    pos_emb = model.pos_embed(pos)                 # (L, H)
    t_self_vec = model.t_embed(self_diff_levels)

    # Get edge embeddings
    E_idx = E_idx.detach()
    E_idx_emb = model.edge_idx_embed(E_idx)         # (num_states, L, k, H)
    E_idx_emb = E_idx_emb.sum(dim=2)                # (num_states, L, H)
    E_idx_emb = E_idx_emb.sum(dim=0).unsqueeze(0)   # (, L, H)

    # Set up initial values
    seq_tensor = x_self_emb + pos_emb
    node_tensor = h_V + E_idx_emb
    chemical_env_tensor = h_EXV
    
    # First apply the init layer to broadcast sequence info to node and chem env
    seq_tensor, node_tensor, chemical_env_tensor = model.init_layer(seq_tensor, node_tensor, chemical_env_tensor)
    
    # Process through decoder layers
    for layer in model.decoder_layers:
        seq_tensor = seq_tensor + t_self_vec.unsqueeze(1)
        seq_tensor, node_tensor, chemical_env_tensor = layer(seq_tensor, node_tensor, chemical_env_tensor)
    
    # Get logits
    logits = model.W_out(torch.cat([seq_tensor, node_tensor, chemical_env_tensor], dim=-1))
    
    # Calculate loss only on masked positions
    loss = torch.zeros(1, device=device)
    for i in range(len(diff_levels)):
        logits_i = logits[i]
        tar_i = S_true[0].long()
        
        # Apply cross entropy loss only on masked positions
        ce_loss = loss_fn(logits_i, tar_i)  # (L,) loss per position
        masked_loss = ce_loss * mask_positions[i].float()  # zero out non-masked positions
        
        # Normalize by number of masked positions
        num_masked = mask_positions[i].sum().float()
        if num_masked > 0:  # avoid division by zero
            loss += masked_loss.sum() / num_masked
        else:
            loss += 0.0
    
    # Average over batch
    loss = loss / len(diff_levels)
    
    if not eval_only:
        loss.backward()
        optimizer.step()
    
    output_dict["loss"] = loss.detach().cpu().item()
    
    return output_dict