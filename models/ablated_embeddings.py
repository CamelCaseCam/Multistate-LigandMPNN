import torch

from .common import TOTAL_DIM, NUM_DIFF_STEPS, train_oneshot_common, diffusion_step_common

class DiffLayer(torch.nn.Module):
    def __init__(self, hidden_dim, rank=16, attn_impl="flash"):  # "flash" or "xformers"
        super().__init__()
        self.h = hidden_dim
        self.r = rank
        self.attn_impl = attn_impl

        # local conv
        self.conv = torch.nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)

        # final projection
        self.linear = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.drop  = torch.nn.Dropout(0.1)

    def forward(self, x, embeddings):        # x: (B,L,H)
        # 1. local conv
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        x = self.norm1(torch.relu(x))

        # Note: we will make the embeddings zero once in the `integrate_embeddings` function
        x = self.linear(torch.cat([x, embeddings], dim=-1))
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
    
    def integrate_embeddings(self, L, device):
        return torch.zeros((1, L, self.hidden_dim), device=device)
    
    def trainable_parameters(self):
        trainable_params = []
        trainable_params += self.seq_enc_layer.parameters()
        trainable_params += self.decoder_layers.parameters()
        trainable_params += self.W_out.parameters()
        trainable_params += self.t_embed.parameters()
        return trainable_params

def train_oneshot(model, feature_dict_list, loss_fn, optimizer, diff_levels, device, eval_only=False):
    # Get common preprocessing
    common_data = train_oneshot_common(model, feature_dict_list, diff_levels, device)
    S_true = common_data["S_true"]
    h_EXV_encoder_fw = common_data["h_EXV_encoder_fw"]
    
    output_dict = {}
    
    if eval_only:
        # Score actual sequence with the original model
        orig_output = model.orig_model.score(feature_dict_list[0], use_sequence=True)
        orig_logits = orig_output["logits"]
        mean_orig_loss = torch.mean(loss_fn(orig_logits.squeeze(0), S_true[0, :].long()))
        output_dict["loss_orig"] = mean_orig_loss.detach().cpu().item()
    
    optimizer.zero_grad()
    
    # Get embeddings (zeros in this case)
    embeddings = model.integrate_embeddings(h_EXV_encoder_fw.shape[1], device)
    
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