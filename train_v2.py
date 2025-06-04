'''
Training script for Multistate LigandMPNN.
'''

from model_utils import ProteinMPNN, cat_neighbors_nodes
import torch
from load_nmr import get_dicts_from_pdb
import numpy as np
import tqdm
from data_utils import featurize, alphabet
import os

from download_data import download_one

MODEL_PATH = "model_params/ligandmpnn_v_32_030_25.pt"
EPOCHS = 200
MIN_STATES = 2
MAX_STATES = 6
NUM_TRAIN_PER_EPOCH = 10
MAX_TOTAL_SEQ_LEN = 1024
MAX_BROADCAST_VAL = 1e4
TEST_WITH_CPU = False
device = torch.device("cuda") if torch.cuda.is_available() and not TEST_WITH_CPU else torch.device("cpu")
LOAD_EXT_CKPT = "model_params/multistate_broadcast_w1.pt"

checkpoint = torch.load(MODEL_PATH)
atom_context_num = checkpoint["atom_context_num"]
ligand_mpnn_use_side_chain_context = True
k_neighbors = checkpoint["num_edges"]
LOAD_PREV = True

model = ProteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=k_neighbors,
    atom_context_num=atom_context_num,
    model_type="multistate_ligand_mpnn_v2",
    ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    device=device,
    debug=False,
)

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

model.load_from_ligand_mpnn_checkpoint(MODEL_PATH)
model.to(device)

orig_model.load_state_dict(checkpoint["model_state_dict"])
orig_model.to(device)

# Set up optimizer with limited weights for finetuning
trainable_params = []
trainable_params += model.broadcast_weights.parameters()
for dl in model.decoder_layers:
    # Only train W1 (message acceptor) layer
    trainable_params += dl.parameters()
trainable_params += model.W_out.parameters()

optimizer = torch.optim.Adam(
    trainable_params,
    lr=1e-4,
    weight_decay=0.0,
)
# Set up loss function
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
# Set up device

def train(model, feature_dict_list, loss_fn, eval_only = False):
    """Sample sequences compatible with multiple conformational states"""
    if model.model_type != "multistate_ligand_mpnn_v2":
        raise ValueError("multistate_sample is only for multistate_ligand_mpnn_v2")

    # Validate input
    if not feature_dict_list or not isinstance(feature_dict_list, list):
        raise ValueError("feature_dict_list must be a non-empty list")
    
    num_states = len(feature_dict_list)
    if num_states < 2:
        raise ValueError("At least two states must be provided")
    output_dict = {}

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
        h_V, h_E, E_idx = model.encode(feature_dict)
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

    all_probs = torch.zeros((B_decoder, L, 20), device=device)
    all_log_probs = torch.zeros((B_decoder, L, 21), device=device)
    S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=device)
    h_S = torch.zeros((B_decoder, L, model.hidden_dim), device=device)

    # Initialize separate h_V stacks for each state
    h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(model.num_decoder_layers)]
    h_E   = torch.cat(h_E_list,   dim=0)      # (num_states, L, k, H)
    E_idx = torch.cat(E_idx_list, dim=0)      # (num_states, L, k)

    # Pre-compute encoder embeddings
    h_EXV_encoder_fw_list = []
    for state_idx in range(num_states):
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S[state_idx:state_idx+1]), h_E[state_idx].unsqueeze(0), E_idx[state_idx].unsqueeze(0))
        h_EXV_encoder = cat_neighbors_nodes(h_V_stack[0][state_idx].unsqueeze(0), h_EX_encoder, E_idx[state_idx].unsqueeze(0))
        h_EXV_encoder_fw_list.append((mask_fw[state_idx:state_idx+1] * h_EXV_encoder).detach())
    h_EXV_encoder_fw = torch.cat(h_EXV_encoder_fw_list, dim=0).detach()

    mean_loss = 0
    # Decoding loop
    optimizer.zero_grad()
    for t_ in range(L):
        t = decoding_order[:, t_]
        mask_t = torch.gather(mask, 1, t[:, None])[:, 0]
        
        idx = t[:, None, None].repeat(1, 1, h_V.shape[-1])

        # Process each state        
        E_idx_t = torch.gather(
            E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
        )
        h_E_t = torch.gather(
            h_E,
            1,
            t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]),
        ).detach()
        h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
        h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1])).detach()

        mask_bw_t = torch.gather(mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2], mask_bw.shape[-1])).detach()
        for l, layer in enumerate(model.decoder_layers):
            next_buf = h_V_stack[l + 1].clone()
            # Vectorized implementation
            h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
            h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
            h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
            state_output = layer(h_V_t, h_ESV_t, mask_V=mask_t)
            full_src = next_buf.clone()      # (B, L, H)
            idx_expanded = idx.expand_as(state_output)
            next_buf = full_src.scatter(1, idx_expanded, state_output)

            # Broadcast to other states
            src_buf  = next_buf                     # read-only reference
            new_buf  = src_buf.clone()              # place to write the updated rows
            for state_idx in range(num_states):

                # first operand in the fold is that state's own row
                running = src_buf[state_idx, t].clone()        # (1, H)

                for other_idx in range(num_states):
                    if other_idx == state_idx:
                        continue

                    # if we have *already* finished 'other_idx' we want its **updated**
                    # value; otherwise we take the original one.
                    other_row = (
                        new_buf[other_idx, t]        # already updated earlier in this loop
                        if other_idx < state_idx     #  <–– sequential dependency
                        else src_buf[other_idx, t]   # not touched yet
                    )

                    inp  = torch.cat((running, other_row), dim=-1)   # (1, 2H)
                    weight = model.broadcast_weights[l](inp)  # (1, H)
                    weight = torch.nn.functional.sigmoid(weight)
                    running = weight * running + (1 - weight) * other_row
                    running = torch.clamp(running, -MAX_BROADCAST_VAL, MAX_BROADCAST_VAL)

                # write the folded result **once** for this state
                new_buf[state_idx, t] = running

            # after every state row has been processed, promote the buffer
            next_buf = new_buf
            # Update the stack
            h_V_stack[l + 1] = next_buf
        
        # Average final h_V_t across states
        h_V_t = torch.gather(
            h_V_stack[-1],
            1,
            t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]),
        )[:, 0]
        h_V_t = torch.mean(h_V_t, dim=0, keepdim=True)
        logits = model.W_out(h_V_t)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Compute the loss
        target = S_true[:, t[0]].long()  # All states have the same sequence, so we pick the first one
        loss = loss_fn(logits, target)
        loss = torch.mean(loss * mask_t)
        if not eval_only:
            loss.backward(retain_graph=False)
            # Check all gradients for NaN
            for param in trainable_params:
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    raise ValueError("Gradient is NaN")
        mean_loss += loss.detach().cpu().item()
        if mean_loss != mean_loss:
            raise ValueError("Loss is NaN")

        S_true_t = torch.gather(S_true, 1, t[0, None].unsqueeze(0))[:, 0].repeat(num_states)
        hsup = h_S.scatter(1, t[:, None, None].repeat(1, 1, h_S.shape[-1]), model.W_s(S_true_t)[:, None, :])
        h_S = hsup.detach()
        # Detach stacks to avoid backpropagation through time
        for l in range(model.num_decoder_layers + 1):
            h_V_stack[l] = h_V_stack[l].detach()
        # Cleanup (this is veeeeeery leaky)
        del loss, logits, log_probs, target, mask_t, h_ESV_decoder_t, h_V_t, h_ESV_t
        del h_E_t, E_idx_t, h_EXV_encoder_t, mask_bw_t
    # Do the update once at the end
    if not eval_only:
        optimizer.step()
    if False and not eval_only and L > 0:
        for param in trainable_params:
            if param.grad is not None:
                param.grad.mul_(1.0 / L)
    # Explicitly free up memory
    
    del h_S, h_E_list, E_idx_list, h_V_list
    output_dict["loss"] = mean_loss / L if L > 0 else 0.0
    output_dict["h_V_stacks"] = h_V_stack
    return output_dict


DATA_PATH = "training/train_multistate.json"
PDB_PATH = "training/pdbs/"

import json

with open(DATA_PATH, "r") as f:
    data = json.load(f)

'''
We train on two types of data: 
 - `NMR`: NMR data with multiple conformations per PDB
 - `Alt`: Collections of PDBs (possibly with multiple conformations per PDB) that represent the same protein with 
    different ligands
'''

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


def old_train_pipeline():
    gen = np.random.default_rng()
    nmr_indices = list(range(len(data["NMR"])))
    for epoch in range(EPOCHS):
        # NMR data
        epoch_loss = 0
        # Shuffle the indices (objectively terrible way of doing this)
        gen.shuffle(nmr_indices)
        num=0
        for i in (pbar := tqdm.tqdm(nmr_indices)):
            num+=1
            pdb_path = PDB_PATH + data["NMR"][i].lower() + ".pdb"
            feature_dict_list = get_dicts_from_pdb(pdb_path, device)
            
            # We know there are at least two states in all NMR data. Pick random states to train on
            total_num_states = len(feature_dict_list)
            if total_num_states < MIN_STATES:
                raise ValueError("Not enough states in NMR data")
            num_states = gen.integers(MIN_STATES, MAX_STATES + 1)
            selected_states = gen.choice(total_num_states, num_states, replace=False) if num_states < total_num_states else np.arange(total_num_states)
            selected_states = [feature_dict_list[i] for i in selected_states]
            selected_states = featurize_dicts(selected_states, device)
            # Train on the selected states
            state_loss = train(model, selected_states, optimizer, loss_fn)
            epoch_loss += state_loss
            pbar.set_description(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss/(num+1):.4f}")

        # Todo: Alt data. testing without it for now

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"training/multistate_checkpoints/model_epoch_{epoch + 1}.pt")

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

if LOAD_PREV:
    # Get checkpoints
    checkpoint_files = os.listdir("training/multistate_checkpoints/")
    checkpoint_files = [f for f in checkpoint_files if f.endswith(".pt")]
    checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[2].split(".")[0]))
    if len(checkpoint_files) == 0:
        print("No checkpoints found, starting from scratch")
        LOAD_PREV = False
    else:
        # Load the latest checkpoint
        latest_checkpoint = checkpoint_files[-1]
        model.load_state_dict(torch.load(f"training/multistate_checkpoints/{latest_checkpoint}"))
        optimizer.load_state_dict(torch.load(f"training/multistate_checkpoints/optimizer_epoch_{latest_checkpoint.split('_')[2]}"))
if LOAD_EXT_CKPT:
    # Load the extended checkpoint
    model.load_state_dict(torch.load(LOAD_EXT_CKPT, weights_only=True))
    print("Loaded existing checkpoint")

def train_pipeline_v2():
    '''
    This training pipeline is a bit different - it uses NMR data as above, but from a much larger dataset
    '''
    global LOAD_PREV

    gen = np.random.default_rng()
    with open(NMR_V2_DATA_PATH, "r") as f:
        trainlines = f.readlines()
    trainlines = [line.strip().lower() for line in trainlines]
    # Overfit test
    trainlines = trainlines[:10]

    with open(NMR_V2_TEST_DATA_PATH, "r") as f:
        testlines = f.readlines()
    testlines = [line.strip().lower() for line in testlines]
    for p in trainable_params:
        p._prev = p.clone()  # snapshot of the weights

    subset_val = gen.choice(len(testlines), 5, replace=False)
    feature_dict_list_train = [[load_pdb_id(testlines[i].strip(), gen), testlines[i].strip()] for i in subset_val]

    for epoch in range(EPOCHS):
        try:
            # NMR data
            epoch_loss = 0
            # Shuffle the indices (objectively terrible way of doing this)
            subset = gen.choice(len(trainlines), NUM_TRAIN_PER_EPOCH, replace=False)
            num=0
            optimizer.zero_grad()
            for i in (pbar := tqdm.tqdm(subset)):
                num+=1
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
                
                # Train on the selected states
                try:
                    out_dict = train(model, feature_dict_list, loss_fn)
                    state_loss = out_dict["loss"]
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

            # Save the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                # If we loaded a checkpoint, delete older checkpoints
                if LOAD_PREV:
                    for f in checkpoint_files:
                        os.remove(f"training/multistate_checkpoints/{f}")
                    LOAD_PREV = False
                torch.save(model.state_dict(), f"training/multistate_checkpoints/model_epoch_{epoch + 1}.pt")
                torch.save(optimizer.state_dict(), f"training/multistate_checkpoints/optimizer_epoch_{epoch + 1}.pt")

            # Testing
            test_loss = 0
            orig_test_loss = 0
            # Pick random PDBs from the test set
            num=0
            for feature_dict_list, pdb_name in feature_dict_list_train:
                num+=1
                if feature_dict_list is None:
                    num -= 1
                    continue
                
                # Train on the selected states
                try:
                    out_dict = train(model, feature_dict_list, loss_fn, eval_only=True)
                    state_loss = out_dict["loss"]
                    orig_loss = out_dict["loss_orig"]
                except torch.OutOfMemoryError:
                    print(f"Out of memory for {pdb_name}, please remove from test set")
                    num -= 1
                    continue
                test_loss += state_loss
                orig_test_loss += orig_loss
            print(f"Test Loss: {test_loss/num:.4f} Orig Loss: {orig_test_loss/num:.4f}")
        except AttributeError:
            print(f"Protein {pdb_name} failed, please remove from training or test set")

if __name__ == "__main__":
    train_pipeline_v2()