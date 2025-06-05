import numpy as np
from scipy.sparse import csr_matrix
import io
from Bio.PDB import PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem
from copy import deepcopy
from Bio.PDB import Structure, Model, Chain, Residue, Atom
import Bio.PDB.internal_coords
from collections import defaultdict
import torch
from copy_residue_coords import copy_all_geometry

# Assuming PeptideBuilder and Geometry are available as in the original script
import PeptideBuilder
from PeptideBuilder import Geometry
from load_nmr import element_list_stripped

upper_elem_stripped = [e.upper() for e in element_list_stripped]

ATOM_NUMBERING_HIERARCHY = ["CA", "CB", "CG", "CD", "CE"]   # All other heavy atoms will be rotated absolutely, hydrogens will be moved relative to the closest heavy atom

def _ensure_pdb_info(atom):
    _name_ledger = defaultdict(lambda: defaultdict(int))
    info = atom.GetPDBResidueInfo()
    if info is None:                                     # ← hydrogens w/out metadata
        info = Chem.AtomPDBResidueInfo()
        atom.SetMonomerInfo(info)

    # ───── fill in missing residue information ─────────
    if info.GetResidueName() == "":
        # copy from heavy neighbour when possible
        heavy = next((n for n in atom.GetNeighbors() if n.GetSymbol() != "H"), None)
        if heavy and heavy.GetPDBResidueInfo():
            hinfo = heavy.GetPDBResidueInfo()
            info.SetResidueNumber(hinfo.GetResidueNumber())
            info.SetResidueName(hinfo.GetResidueName())
            info.SetChainId(hinfo.GetChainId())
            info.SetInsertionCode(hinfo.GetInsertionCode())
        else:  # fallback
            info.SetResidueNumber(1)
            info.SetResidueName("UNK")

    # ───── propose an atom label if missing/blank ──────
    name = info.GetName().strip()
    if not name or name == atom.GetSymbol():             # common when it was "H   "
        if atom.GetSymbol() == "H":
            # base label from bonded heavy atom if any
            heavy = next((n for n in atom.GetNeighbors() if n.GetSymbol() != "H"), None)
            if heavy and heavy.GetPDBResidueInfo():
                base = heavy.GetPDBResidueInfo().GetName().strip()[-1]  # "A", "B", …
                name = f"H{base}"
            else:
                name = "H"                               # worst-case fallback
        else:
            name = atom.GetSymbol()

    # ───── ensure uniqueness inside the residue ────────
    resid_key = (info.GetResidueNumber(), info.GetChainId())
    used = _name_ledger[resid_key]
    if name in used:
        used[name] += 1
        # make room for a digit suffix by trimming to 3 chars, then append count
        name = (name[:3] + str(used[name]))[:4]
    else:
        used[name] = 1
    info.SetName(name.ljust(4))                          # pad to width 4

    return info

class ProteinStructure:
    def __init__(self):
        self.positions = None  # np.array (N, 3)
        self.elements = None  # list of str
        self.atom_names = None  # list of str
        self.residue_indices = None  # np.array of int
        self.residue_types = None  # list of str
        self.bond_pairs = None  # list of (int, int)
        self.adjacency = None  # csr_matrix
        self.atom_to_residue = None  # dict or list
        self.sequence = None  # amino acid sequence as a string

    @classmethod
    def from_sequence(cls, sequence, unfolded=True):
        # Generate initial structure using PeptideBuilder
        if len(sequence) == 0:
            raise ValueError("Sequence is empty")
        
        # Initialize first residue
        geo = Geometry.geometry(sequence[0])
        if unfolded:
            geo.phi = -180.0
            geo.psi_im1 = 180.0
            geo.omega = 180.0
        else:
            geo.phi = np.random.uniform(-180, 180)
            geo.psi_im1 = np.random.uniform(-180, 180)
            geo.omega = np.random.uniform(160, 200)
        bio_structure = PeptideBuilder.initialize_res(geo)
        
        # Add remaining residues
        for aa in sequence[1:]:
            geo = Geometry.geometry(aa)
            if unfolded:
                geo.phi = -180.0
                geo.psi_im1 = 180.0
                geo.omega = 180.0
            else:
                geo.phi = np.random.uniform(-180, 180)
                geo.psi_im1 = np.random.uniform(-180, 180)
                geo.omega = np.random.uniform(160, 200)
            bio_structure = PeptideBuilder.add_residue(bio_structure, geo)
        # Add terminal oxygen
        bio_structure = PeptideBuilder.add_terminal_OXT(bio_structure)
        
        # Convert to RDKit to add hydrogens and get bonds
        # First, save Biopython structure to PDB string
        pdb_io = PDBIO()
        pdb_io.set_structure(bio_structure)
        pdb_string = io.StringIO()
        pdb_io.save(pdb_string)
        pdb_string = pdb_string.getvalue()
        
        # Load into RDKit
        mol = Chem.MolFromPDBBlock(pdb_string, sanitize=False, removeHs=False)
        if mol is None:
            raise ValueError("Failed to load structure into RDKit")
        
        # Get conformer
        conf = mol.GetConformer()
        
        # Extract data
        positions = []
        elements = []
        atom_names = []
        residue_indices = []
        residue_types = []
        bond_pairs = []
        
        atom_residue_map = {}
        current_residue = -1
        residue_map = {}
        
        for atom in mol.GetAtoms():
            info = _ensure_pdb_info(atom)
            if info:
                res_name = info.GetResidueName().strip()
                res_num = info.GetResidueNumber()
                if res_num not in residue_map:
                    residue_map[res_num] = len(residue_types)
                    residue_types.append(res_name)
                    current_residue = len(residue_types) - 1
                atom_residue = residue_map[res_num]
            else:
                atom_residue = current_residue
            
            positions.append(conf.GetAtomPosition(atom.GetIdx()))
            elements.append(atom.GetSymbol())
            atom_names.append(info.GetName().strip())
            residue_indices.append(atom_residue)
        
        # Convert positions to numpy
        positions = np.array([[p.x, p.y, p.z] for p in positions])
        
        # Get bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_pairs.append((i, j))
        
        # Create instance
        ps = cls()
        ps.positions = positions
        ps.elements = elements
        ps.atom_names = atom_names
        ps.residue_indices = np.array(residue_indices)
        ps.residue_types = residue_types
        ps.bond_pairs = bond_pairs
        ps.sequence = sequence
        
        # Create adjacency
        N = len(positions)
        row = [b[0] for b in bond_pairs] + [b[1] for b in bond_pairs]
        col = [b[1] for b in bond_pairs] + [b[0] for b in bond_pairs]
        data = [1] * len(row)
        ps.adjacency = csr_matrix((data, (row, col)), shape=(N, N), dtype=bool)
        
        return ps
    
    def calculate_angles(self):
        '''
        Returns the backbone angles (phi, psi, omega) for each residue.
        '''
        bio_structure = self.to_bio_structure()
        angles = []
        # Assume 1 model, 1 chain
        model = bio_structure[0]
        chain = model["A"]
        chain.atom_to_internal_coordinates()
        for res in chain.get_residues():
            ric = res.internal_coord
            ang = (
                ric.get_angle("psi"),
                ric.get_angle("phi"),
                ric.get_angle("omg")
            )
            angles.append(ang)
        return np.array(angles), bio_structure

    def update_residues(self, residue_str):
        '''
        Update residues to match the new string. Backbone atoms are kept, sidechain atoms are altered. 
        '''
        # Get angles
        angles, biostruct = self.calculate_angles()
        # Regenerate the structure from the new residue string
        geo = Geometry.geometry(residue_str[0])
        # If not same, copy only backbone
        geo.phi = angles[0][0]
        geo.phi = geo.phi if geo.phi is not None else -180.0
        geo.psi_im1 = angles[0][1]
        geo.psi_im1 = geo.psi_im1 if geo.psi_im1 is not None else 180.0
        geo.omega = angles[0][2]
        geo.omega = geo.omega if geo.omega is not None else 180.0
        if residue_str[0] == self.sequence[0]:
            # Copy *everything* from the first residue in the existing structure
            copy_all_geometry(biostruct[0]["A"][1], geo)

        bio_structure = PeptideBuilder.initialize_res(geo)

        for i, aa in enumerate(residue_str[1:]):
            geo = Geometry.geometry(aa)
            geo.phi = angles[i + 1][0]
            geo.phi = geo.phi if geo.phi is not None else -180.0
            geo.psi_im1 = angles[i + 1][1]
            geo.psi_im1 = geo.psi_im1 if geo.psi_im1 is not None else 180.0
            geo.omega = angles[i + 1][2]
            geo.omega = geo.omega if geo.omega is not None else 180.0
            if aa == self.sequence[i + 1]:
                # Copy *everything* from the existing residue
                copy_all_geometry(biostruct[0]["A"][i + 2], geo)
            bio_structure = PeptideBuilder.add_residue(bio_structure, geo)

        # Add terminal oxygen
        bio_structure = PeptideBuilder.add_terminal_OXT(bio_structure)

        pdb_io = PDBIO()
        pdb_io.set_structure(bio_structure)
        pdb_string = io.StringIO()
        pdb_io.save(pdb_string)
        pdb_string = pdb_string.getvalue()
        
        # TODO: stop duplicating code, pull into own function
        # Load into RDKit
        mol = Chem.MolFromPDBBlock(pdb_string, sanitize=False, removeHs=False)
        if mol is None:
            raise ValueError("Failed to load structure into RDKit")
        
        # Get conformer
        conf = mol.GetConformer()
        
        # Extract data
        positions = []
        elements = []
        atom_names = []
        residue_indices = []
        residue_types = []
        bond_pairs = []
        
        atom_residue_map = {}
        current_residue = -1
        residue_map = {}
        
        for atom in mol.GetAtoms():
            info = _ensure_pdb_info(atom)
            if info:
                res_name = info.GetResidueName().strip()
                res_num = info.GetResidueNumber()
                if res_num not in residue_map:
                    residue_map[res_num] = len(residue_types)
                    residue_types.append(res_name)
                    current_residue = len(residue_types) - 1
                atom_residue = residue_map[res_num]
            else:
                atom_residue = current_residue
            
            positions.append(conf.GetAtomPosition(atom.GetIdx()))
            elements.append(atom.GetSymbol())
            atom_names.append(info.GetName().strip())
            residue_indices.append(atom_residue)
        
        # Convert positions to numpy
        positions = np.array([[p.x, p.y, p.z] for p in positions])
        
        # Get bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_pairs.append((i, j))

        self.positions = positions
        self.elements = elements
        self.atom_names = atom_names
        self.residue_indices = np.array(residue_indices)
        self.residue_types = residue_types
        self.bond_pairs = bond_pairs
        
        # Create adjacency
        N = len(positions)
        row = [b[0] for b in bond_pairs] + [b[1] for b in bond_pairs]
        col = [b[1] for b in bond_pairs] + [b[0] for b in bond_pairs]
        data = [1] * len(row)
        self.adjacency = csr_matrix((data, (row, col)), shape=(N, N), dtype=bool)

    def get_local_environments(self, ca_index):
        # ca_index is the residue index
        ca_mask = (self.atom_names == 'CA') & (self.residue_indices == ca_index)
        if not any(ca_mask):
            raise ValueError("No CA for the residue")
        ca_atom = np.where(ca_mask)[0][0]
        ca_pos = self.positions[ca_atom]
        
        # Compute distances
        distances = np.linalg.norm(self.positions - ca_pos, axis=1)
        
        # Sort by distance
        sorted_indices = np.argsort(distances)
        
        # Return positions, but perhaps relative
        # The task says "atoms sorted by distance", probably positions and indices
        return self.positions[sorted_indices], sorted_indices

    # Other methods as needed, e.g. to update positions from physics

    def update_positions(self, new_positions):
        self.positions = new_positions

    def to_bio_structure(self):
        structure = Structure.Structure("protein")
        model     = Model.Model(0)
        chain     = Chain.Chain("A")

        # cache already-created residues: {res_idx: Residue}
        residue_cache = {}

        for i, pos in enumerate(self.positions):
            res_idx = int(self.residue_indices[i])

            # ─── get or build the Residue ─────────────────────
            if res_idx not in residue_cache:
                res_name = self.residue_types[res_idx]
                res_id   = (" ", res_idx + 1, " ")         # (hetero flag, number, icode)
                residue  = Residue.Residue(res_id, res_name, " ")
                chain.add(residue)
                residue_cache[res_idx] = residue
            else:
                residue = residue_cache[res_idx]

            # ─── create the Atom ─────────────────────────────
            atom_name = self.atom_names[i]
            element   = self.elements[i]
            atom      = Atom.Atom(
                atom_name,               # name (string ≤4 cols)
                pos,                     # coordinates (numpy array or list)
                0.0,                     # B-factor
                1.0,                     # occupancy
                " ",                     # altLoc
                atom_name,               # fullname (again ≤4 cols)
                i,                       # serial number
                element                  # element symbol
            )
            residue.add(atom)

        model.add(chain)
        structure.add(model)
        return structure

    def get_calpha_envs(self, cutoff=5.0, max_neighbours=80) -> torch.Tensor:
        '''
        Returns the local environments of all CA atoms in the structure.

        Args:
            cutoff (float): Distance cutoff for neighbors in angstroms. 
        Returns:
            torch.Tensor: Array of shape (N, M, 3) where N is the number of CA atoms and M is the number of neighbors within the cutoff, 
            padded to max_neighbors. Alpha carbon is the first atom in each environment, others are sorted by distance. Distances are
            relative to the alpha carbon position.
        '''
        ca_indices = np.where([atom[:2] == 'CA' for atom in self.atom_names])[0]

        ca_distances = np.linalg.norm(self.positions[ca_indices] - self.positions[:, np.newaxis], axis=2).T

        ca_envs = []
        for i, ca_index in enumerate(ca_indices):
            distances = ca_distances[i]
            neighbours = np.where(distances < cutoff)[0]

            # Sort neighbours by distance
            sorted_neighbour_ind = np.argsort(distances[neighbours])
            sorted_neighbours = neighbours[sorted_neighbour_ind]
            sorted_neighbours = sorted_neighbours[:max_neighbours]
            # Pad with zeros if fewer than max_neighbours
            if len(sorted_neighbours) < max_neighbours:
                padding = np.zeros((max_neighbours - len(sorted_neighbours), 3))
                env_positions = np.vstack((self.positions[sorted_neighbours], padding))
            else:
                env_positions = self.positions[sorted_neighbours]

            # Get positions relative to CA
            ca_pos = self.positions[ca_index]
            env_positions = self.positions[sorted_neighbours] - ca_pos
            
            # Get types
            env_types = [self.elements[i] for i in sorted_neighbours]
            env_types = [upper_elem_stripped.index(e.upper()) if e.upper() in upper_elem_stripped else len(upper_elem_stripped) for e in env_types]  # Convert to indices

            # Get residue type
            residue = self.residue_types[self.residue_indices[ca_index]]

            ca_envs.append({
                'coords': torch.tensor(env_positions, dtype=torch.float32),
                'types': torch.tensor(env_types, dtype=torch.long),
                'residue': residue
            })
        return ca_envs