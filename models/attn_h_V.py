import torch
import math
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

from .common import TOTAL_DIM, NUM_DIFF_STEPS, diffusion_step_common, train_oneshot_common

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
            Q4 = Q.view(B, L, self.num_heads, self.head_dim).to(torch.float16)
            K4 = K.view(B, L, self.num_heads, self.head_dim).to(torch.float16)
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
        self.comb_weights = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)

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
    
    def integrate_embeddings(self, h_V):
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
    
    def trainable_parameters(self):
        trainable_params = []
        trainable_params += self.comb_weights.parameters()
        trainable_params += self.seq_enc_layer.parameters()
        trainable_params += self.decoder_layers.parameters()
        trainable_params += self.W_out.parameters()
        trainable_params += self.t_embed.parameters()
        return trainable_params
    
def train_oneshot(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only=False):
    # Get common preprocessing
    common_data = train_oneshot_common(model, feature_dict_list, diff_levels, device)
    S_true = common_data["S_true"]
    h_V = common_data["h_V"]
    
    output_dict = {}
    
    if eval_only:
        # Score actual sequence with the original model
        orig_output = model.orig_model.score(feature_dict_list[0], use_sequence=True)
        orig_logits = orig_output["logits"]
        mean_orig_loss = torch.mean(loss_fn(orig_logits.squeeze(0), S_true[0, :].long()))
        output_dict["loss_orig"] = mean_orig_loss.detach().cpu().item()
    
    optimizer.zero_grad()
    
    # Get embeddings using h_V directly
    embeddings = model.integrate_embeddings(h_V)
    
    # Run diffusion step
    logits, x_tm1 = diffusion_step_common(model, embeddings, diff_levels, S_true, device)
    
    # Calculate loss
    loss = torch.zeros(1, device=device)
    for i in range(len(diff_levels)):
        logits_i = logits[i]
        tar_i = x_tm1[i].long()
        ce_t = loss_fn(logits_i, tar_i)
        loss += ce_t.mean()
    
    if not eval_only:
        loss.backward()
        optimizer.step()
    
    output_dict["loss"] = loss.detach().cpu().item() / len(diff_levels)
    
    return output_dict