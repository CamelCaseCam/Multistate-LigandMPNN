"""
Diffusion-based protein folding script.
Uses trained models to iteratively fold a protein from an unfolded/tangled state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
from Bio.PDB.vectors import Vector
import random
from copy import deepcopy
import tqdm
import PeptideBuilder
from PeptideBuilder import Geometry
import protein_structure as ps

# Model path constants (to be updated when training is complete)
ENCODER_PATH = 'encoder_8l.pth'
DIFFUSION_PATH = 'bidirectional_diffusion_8l.pth'
DECODER_PATH = 'sequence_decoder.pth'
ATOMIC_AUTOENCODER_PATH = 'autoregressive_atomic_autoencoder.pth'

# Configuration constants
COMPLETELY_UNFOLDED = True  # True for linear, False for tangled loops
NUM_DIFFUSION_STEPS = 100
PHYSICS_TIMESTEPS_PER_DIFFUSION = 50
DT = 0.01  # Physics timestep

# Spring constants (will be scheduled)
BACKBONE_SPRING_CONSTANT = 0.0
SIDECHAIN_SPRING_CONSTANT = 0.0
BOND_SPRING_MAX = 2.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import necessary components from existing files
from load_nmr import element_list_stripped
from encoder import AtomicAutoencoder
from train_with_enc import Encoder, BidirectionalDiffusion, SequenceDecoder, amino_acid_tokens

def generate_initial_peptide(sequence, unfolded=True):
    return ps.ProteinStructure.from_sequence(sequence, unfolded=unfolded)

def get_sequence_from_structure(structure):
    return ''.join(structure.residue_types)

def update_structure_sequence(structure, new_sequence):
    structure.update_residues(new_sequence)

def apply_physics_update(structure, target_environments, diffusion_step, total_steps):
    """
    Apply physics-based updates to move atoms toward target positions.
    """
    # Schedule spring constants
    progress = diffusion_step / total_steps
    #bond_spring = BOND_SPRING_CONSTANT_START + (BOND_SPRING_CONSTANT_END - BOND_SPRING_CONSTANT_START) * (progress ** 2)
    # Parabolic scheduling for bond spring constant where vertex is (total_steps / 2, BOND_SPRING_MAX)
    # f(x, a, b) = -x(x-a)(4b/a^2) where a=1
    bond_spring = -progress * (progress - 1) * (4 * BOND_SPRING_MAX)
    
    # Get current atom positions and prepare data
    positions = structure.positions.copy()
    velocities = np.zeros_like(positions)
    
    # Bond pairs from structure
    bond_pairs = structure.bond_pairs
    
    # Prepare CA indices
    ca_mask = np.array(structure.atom_names) == 'CA'
    ca_indices = []
    for r in range(len(structure.residue_types)):
        res_mask = (structure.residue_indices == r) & ca_mask
        if any(res_mask):
            ca_indices.append(np.where(res_mask)[0][0])
    
    # Physics simulation
    for timestep in range(PHYSICS_TIMESTEPS_PER_DIFFUSION):
        forces = np.zeros_like(positions)
        num_contributions = np.zeros_like(positions)
        
        # 1. Forces from target environments
        for i, (ca_idx, target_env) in enumerate(zip(ca_indices, target_environments)):
            ca_pos = positions[ca_idx]
            rel_pos = positions - ca_pos
            distances = np.linalg.norm(rel_pos, axis=1)
            mask = distances <= 5.0
            neighbor_indices = np.where(mask)[0]
            if len(neighbor_indices) > 0:
                sorted_indices = neighbor_indices[np.argsort(distances[neighbor_indices])]
                target_coords = target_env['coords'].numpy()
                num_targets = min(len(sorted_indices), len(target_coords))
                for j in range(num_targets):
                    atom_idx = sorted_indices[j]
                    current_rel_pos = rel_pos[atom_idx]
                    target_rel_pos = target_coords[j]
                    displacement = target_rel_pos - current_rel_pos
                    atom_name = structure.atom_names[atom_idx]
                    spring_k = BACKBONE_SPRING_CONSTANT if atom_name in ['N', 'CA', 'C', 'O'] else SIDECHAIN_SPRING_CONSTANT
                    force = spring_k * displacement
                    forces[atom_idx] += force
                    num_contributions[atom_idx, :] += np.ones(3)
        
        forces = np.divide(forces, num_contributions, where=(num_contributions > 0))
        forces = np.where(num_contributions > 0, forces, 0)
        
        # 2. Bond forces
        if bond_pairs:
            atom1_idxs = np.array([p[0] for p in bond_pairs])
            atom2_idxs = np.array([p[1] for p in bond_pairs])
            pos1 = positions[atom1_idxs]
            pos2 = positions[atom2_idxs]
            vec = pos2 - pos1
            lengths = np.linalg.norm(vec, axis=1)
            mask = lengths > 0
            directions = np.zeros_like(vec)
            directions[mask] = vec[mask] / lengths[mask, np.newaxis]
            force_magnitudes = bond_spring * (1.5 - lengths)
            forces_on_2 = force_magnitudes[:, np.newaxis] * directions
            forces_on_1 = -forces_on_2
            forces[atom1_idxs] += forces_on_1
            forces[atom2_idxs] += forces_on_2
        
        # 3. Update positions using Verlet integration
        velocities += forces * DT
        positions += velocities * DT
        
        # Apply damping
        velocities *= 0.99
    
    # Update positions in structure
    structure.update_positions(positions)

def load_models():
    """
    Load all trained models.
    """
    # Load atomic autoencoder
    autoencoder = AtomicAutoencoder(len(element_list_stripped) + 1, 256, 8, 4, bottleneck_tokens=8).to(device)
    if os.path.exists(ATOMIC_AUTOENCODER_PATH):
        autoencoder.load_state_dict(torch.load(ATOMIC_AUTOENCODER_PATH, map_location=device))
    autoencoder.eval()
    
    # Load encoder
    encoder = Encoder(
        autoencoder,
        vocab_size=len(amino_acid_tokens),
        embed_dim=128,
        num_heads=8,
        num_layers=3,
        max_len=512
    ).to(device)
    if os.path.exists(ENCODER_PATH):
        encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    encoder.eval()
    
    # Load diffusion model
    diffusion = BidirectionalDiffusion(
        embed_dim=128,
        num_layers=8,
        num_heads=8,
        num_timesteps=1000
    ).to(device)
    if os.path.exists(DIFFUSION_PATH):
        diffusion.load_state_dict(torch.load(DIFFUSION_PATH, map_location=device))
    diffusion.eval()
    
    # Load decoder
    decoder = SequenceDecoder(128, len(amino_acid_tokens)).to(device)
    if os.path.exists(DECODER_PATH):
        decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
    decoder.eval()
    
    return autoencoder, encoder, diffusion, decoder

def diffusion_folding(initial_sequence, output_path="folded_structure.pdb"):
    """
    Main diffusion folding pipeline.
    """
    print(f"Starting diffusion folding for sequence: {initial_sequence}")
    
    # Load models
    autoencoder, encoder, diffusion, decoder = load_models()
    
    # Generate initial structure
    structure = generate_initial_peptide(initial_sequence, unfolded=COMPLETELY_UNFOLDED)
    trajectory = [deepcopy(structure)]
    
    # Diffusion timesteps
    timesteps = torch.linspace(1000, 1, NUM_DIFFUSION_STEPS, dtype=torch.long, device=device)
    
    for step, t in (pbar := tqdm.tqdm(list(enumerate(timesteps)))):
        pbar.set_description(f"Timestep {t}")
        
        # 1. Extract current environments and sequence
        current_envs = structure.get_calpha_envs()
        current_sequence = get_sequence_from_structure(structure)
        
        # 2. Prepare data for model
        if not current_envs:
            print("No environments found, skipping step")
            continue
            
        # Convert to model format
        max_atoms = 80
        coords_batch = []
        types_batch = []
        sequence_tokens = []
        
        for env in current_envs:
            coords = env['coords']
            types = env['types']
            
            # Pad/truncate to max_atoms
            if len(coords) > max_atoms:
                coords = coords[:max_atoms]
                types = types[:max_atoms]
            else:
                pad_coords = torch.zeros((max_atoms - len(coords), 3))
                coords = torch.cat([coords, pad_coords], dim=0)
                pad_types = torch.full((max_atoms - len(types),), len(element_list_stripped), dtype=torch.long)
                types = torch.cat([types, pad_types], dim=0)
            
            coords_batch.append(coords)
            types_batch.append(types)
            
            # Convert residue to token
            residue = env['residue']
            aa_map = {
                'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
            }
            aa = aa_map.get(residue, 'X')
            sequence_tokens.append(amino_acid_tokens.index(aa))
        
        # Convert to tensors
        coords_tensor = torch.stack(coords_batch).unsqueeze(0).to(device)  # (1, L, max_atoms, 3)
        types_tensor = torch.stack(types_batch).unsqueeze(0).to(device)    # (1, L, max_atoms)
        sequence_tensor = torch.tensor(sequence_tokens).unsqueeze(0).to(device)  # (1, L)
        
        # 3. Run model inference
        with torch.no_grad():
            # Get embeddings
            seq_emb, atomic_emb = encoder(sequence_tensor, coords_tensor, types_tensor)
            
            # Add noise for current timestep
            t_batch = t.unsqueeze(0).to(device)
            
            # Generate noise
            noise_seq = torch.randn_like(seq_emb)
            noise_env = torch.randn_like(atomic_emb)
            
            # Add noise according to diffusion schedule
            alpha_bar = 1.0 - (t.float() / 1000.0)  # Simple linear schedule
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
            
            noisy_seq = sqrt_alpha_bar * seq_emb + sqrt_one_minus_alpha_bar * noise_seq
            noisy_env = sqrt_alpha_bar * atomic_emb + sqrt_one_minus_alpha_bar * noise_env
            
            # Predict noise
            pred_noise_seq, pred_noise_env = diffusion(noisy_seq, noisy_env, t_batch, t_batch)
            
            # Denoise (simplified DDPM step)
            denoised_seq = (noisy_seq - sqrt_one_minus_alpha_bar * pred_noise_seq) / sqrt_alpha_bar
            denoised_env = (noisy_env - sqrt_one_minus_alpha_bar * pred_noise_env) / sqrt_alpha_bar
            
            # Decode sequence
            new_sequence_logits = decoder(denoised_seq)
            new_sequence_tokens = torch.argmax(new_sequence_logits, dim=-1).squeeze(0)
            new_sequence = ''.join([amino_acid_tokens[token.item()] for token in new_sequence_tokens])
        
        # 4. Update structure
        # Update sequence
        update_structure_sequence(structure, new_sequence)
        
        # Generate new target environments from denoised embeddings
        # For simplicity, we'll use the current environments as targets
        # In a full implementation, you'd decode the atomic embeddings back to coordinates
        target_environments = current_envs
        
        # Apply physics updates
        apply_physics_update(structure, target_environments, step, NUM_DIFFUSION_STEPS)
        
        # Add to trajectory
        trajectory.append(deepcopy(structure))
    
    # Save final structure
    io = PDBIO()
    io.set_structure(structure.to_bio_structure())
    io.save(output_path)
    
    # Save trajectory as multi-state PDB
    trajectory_structure = Structure.Structure('trajectory')
    for i, struct in enumerate(trajectory):
        model = struct.to_bio_structure()[0]
        model.id = i
        trajectory_structure.add(model)
    
    io.set_structure(trajectory_structure)
    trajectory_path = output_path.replace('.pdb', '_trajectory.pdb')
    io.save(trajectory_path)
    
    print(f"Folding complete. Final structure saved to {output_path}")
    print(f"Trajectory saved to {trajectory_path}")
    return structure, trajectory

if __name__ == "__main__":
    # Example usage with a shorter test sequence
    test_sequence = "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
    structure, trajectory = diffusion_folding(test_sequence)
    print("Diffusion folding completed successfully!")
