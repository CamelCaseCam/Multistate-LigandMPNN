import torch
import math
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

from model_utils import cat_neighbors_nodes

from .common import TOTAL_DIM, NUM_DIFF_STEPS, categorical_diffuse

class DiffLayer(torch.nn.Module):
    def __init__(self, hidden_dim, rank=16, attn_impl="flash"):  # "flash" or "xformers" or "naive"
        super().__init__()
        self.h = hidden_dim
        self.r = rank
        self.attn_impl = attn_impl

        # local conv
        self.conv = torch.nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)

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
        # Q,K,V: (B,L,r)
        B, L, r = Q.shape
        Qh = Q.transpose(1,2)        # (B,r,L)
        Kh = K.transpose(1,2)
        Vh = V.transpose(1,2)

        ctx = torch.bmm(Qh, Kh.transpose(1,2)) / math.sqrt(r)
        ctx = torch.nn.functional.softmax(ctx, dim=-1)
        ctx = torch.bmm(ctx, Vh)
        ctx = ctx.transpose(1,2).reshape(B, L, r)   # (B,L,r)
        return ctx

    def forward(self, x, embeddings):        # x: (B,L,H)
        # If x is on CPU, change our attention implementation to naive
        if x.device.type == "cpu":
            self.attn_impl = "naive"
        
        # 1. local conv
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        x = self.norm1(torch.relu(x))
        x_orig = x.clone()

        # 3. low-rank attention (flash or xformers)
        Q, K, V = self.proj_q(x), self.proj_k(embeddings), self.proj_v(embeddings)
        if self.attn_impl == "flash":
            context = self._flash_attn(Q, K, V)
        elif self.attn_impl == "xformers":
            context = self._xformers_attn(Q, K, V)
        elif self.attn_impl == "naive":
            context = self._naive_attn(Q, K, V)

        if self.attn_impl is not None:
            extracted_emb = self.ln_attn(embeddings + self.proj_out(context))

        x = self.linear(torch.cat([x_orig, extracted_emb], dim=-1))
        x = self.norm2(torch.relu(x))
        x = self.drop(x)

        return x

class diffmodel(torch.nn.Module):
    def __init__(self, orig_model, maxlen=512, k_neighbors=32, EXV_dim=384):
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
        self.comb_weights = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # For encoding sequence information
        self.graph_integration_layer = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.seq_enc_layer = torch.nn.Embedding(TOTAL_DIM, self.hidden_dim)

        self.decoder_layers = torch.nn.ModuleList(
            [DiffLayer(self.hidden_dim, rank=16, attn_impl="xformers") for _ in range(4)]
        )
        self.W_out = torch.nn.Linear(self.hidden_dim, 21)
        self.t_embed = torch.nn.Embedding(NUM_DIFF_STEPS, self.hidden_dim)
        self.pos_embed = torch.nn.Embedding(maxlen, self.hidden_dim)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, feature_dict_list):
        # Get embeddings from the original model
        assert len(feature_dict_list) > 0
        raise NotImplementedError("This model is not implemented yet")
    
    def integrate_embeddings(self, h_V, h_EXV):
        # Combine the embeddings
        # First, stack the embeddings across nodes so (num_states, L, k, H) -> (num_states, L, H * k)
        h_EXV_stacked = h_EXV.view(h_EXV.shape[0], h_EXV.shape[1], -1)
        # There may be fewer than 32 neighbors for some nodes, so we need to pad the embeddings along the final dimension
        h_EXV_stacked = torch.nn.functional.pad(h_EXV_stacked, (0, self.k_neighbors * self.EXV_dim - h_EXV_stacked.shape[2]), value=0)
        # Then, apply the linear layer to get the combined weights
        h_EXV = self.collect_weights(h_EXV_stacked)

        h_EXVV = torch.cat([h_V, h_EXV], dim=-1)
        # Apply the graph integration layer
        h_EXVV = self.graph_integration_layer(h_EXVV)

        ratio = torch.zeros_like(h_EXVV)
        for state_idx in range(1, h_EXVV.shape[0]):
            last = h_EXVV[state_idx - 1]
            current = h_EXVV[state_idx]
            inp = torch.cat([last, current], dim=-1)
            ratio[state_idx] = self.comb_weights(inp)
        # Now, use the ratio to combine the embeddings (summing to 1)
        ratio = ratio / (torch.sum(ratio, dim=-1, keepdim=True) + 1e-8)
        # Summing time!
        embeddings = torch.sum(ratio * h_EXVV, dim=0, keepdim=True)
        return embeddings
    
    def trainable_parameters(self):
        trainable_params = []
        trainable_params += self.collect_weights.parameters()
        trainable_params += self.comb_weights.parameters()
        trainable_params += self.graph_integration_layer.parameters()
        trainable_params += self.seq_enc_layer.parameters()
        trainable_params += self.decoder_layers.parameters()
        trainable_params += self.W_out.parameters()
        trainable_params += self.t_embed.parameters()
        trainable_params += self.pos_embed.parameters()
        return trainable_params
    
def train_oneshot(model : diffmodel, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only = False):
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
        orig_output = model.orig_model.score(feature_dict_list[0], use_sequence=True)
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
    h_V = h_V.detach()

    embeddings = model.integrate_embeddings(h_V, h_EXV_encoder_fw)
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
    pos = torch.arange(L, device=x.device)         # (L,)
    pos_emb = model.pos_embed(pos)                 # (L, H)
    x_self_emb = x_self_emb + pos_emb              # broadcast over batch
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
    if not eval_only:
        loss.backward()
        optimizer.step()

    output_dict["loss"] = loss.detach().cpu().item() / len(diff_levels)

    return output_dict