import torch
import os
from prody import parsePDB
import numpy as np
from load_nmr import element_list_stripped
import tqdm

elem = [e.upper() for e in element_list_stripped]

resname_to_single_letter = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X"
}

def get_atom_env(atoms, ca_atoms, idx, neighbor_distance=5.0):
    assert 0 <= idx < len(ca_atoms), "Index out of range for CA atoms."
    ca_atom = ca_atoms[idx]
    ca_pos = ca_atom.getCoords()
    neighbours = atoms.select(f'within {neighbor_distance} of target', target=ca_pos)
    coords = neighbours.getCoords()
    types = neighbours.getElements()
    types = [elem.index(element.upper()) if element.upper() in elem else len(elem) for element in types]
    coords = coords - ca_pos
    r = np.linalg.norm(coords, axis=1)
    indices = np.argsort(r)
    coords = coords[indices]
    types = [types[i] for i in indices]
    return coords, types

MAX_STORAGE = 5e9  # 5GB
COMMON = {'H', 'C', 'N', 'O', 'S'}
PDB_DIR = 'training/pdbs'
DATA_DIR = 'training/data'
os.makedirs(os.path.join(DATA_DIR, 'common'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'uncommon'), exist_ok=True)

all_elements = set()
current_size = 0

for f in tqdm.tqdm(os.listdir(PDB_DIR)):
    if not f.endswith('.pdb'):
        continue
    file_path = os.path.join(PDB_DIR, f)
    atoms = parsePDB(file_path)
    elements = set(atoms.getElements())
    all_elements.update(e.upper() for e in elements)
    is_uncommon = any(e.upper() not in COMMON for e in elements)
    folder = 'uncommon' if is_uncommon else 'common'
    ca_atoms = atoms.select('calpha')
    residues = ca_atoms.getResnames() if ca_atoms is not None else []
    if ca_atoms is None or len(ca_atoms) == 0:
        continue
    envs = []
    for i in range(len(ca_atoms)):
        coords, types = get_atom_env(atoms, ca_atoms, i)
        coords = torch.tensor(coords, dtype=torch.float32)
        types = torch.tensor(types, dtype=torch.long)
        res = residues[i]
        res = resname_to_single_letter.get(res, 'X')

        envs.append({'coords': coords, 'types': types, 'residue': res})
    save_path = os.path.join(DATA_DIR, folder, f.replace('.pdb', '.pt'))
    torch.save({'envs': envs}, save_path)
    current_size += os.path.getsize(save_path)
    if current_size > MAX_STORAGE:
        print(f'Exceeded storage budget at {current_size} bytes')
        break

with open('training/all_elements.txt', 'w') as out:
    out.write(','.join(sorted(all_elements)))

print("Dataset generation complete.")
