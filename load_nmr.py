from prody import parsePDB
import numpy as np
import torch
from data_utils import get_aligned_coordinates

element_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mb",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]

element_list_stripped = [
    "H",
    "Li",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Mb",
    "Pd",
    "Ag",
]

def get_dicts_from_pdb(path, device):
    element_list = [item.upper() for item in element_list]
    element_dict = dict(zip(element_list, range(1, len(element_list))))
    restype_3to1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    restype_STRtoINT = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
        "X": 20,
    }

    atom_order = {
        "N": 0,
        "CA": 1,
        "C": 2,
        "CB": 3,
        "O": 4,
        "CG": 5,
        "CG1": 6,
        "CG2": 7,
        "OG": 8,
        "OG1": 9,
        "SG": 10,
        "CD": 11,
        "CD1": 12,
        "CD2": 13,
        "ND1": 14,
        "ND2": 15,
        "OD1": 16,
        "OD2": 17,
        "SD": 18,
        "CE": 19,
        "CE1": 20,
        "CE2": 21,
        "CE3": 22,
        "NE": 23,
        "NE1": 24,
        "NE2": 25,
        "OE1": 26,
        "OE2": 27,
        "CH2": 28,
        "NH1": 29,
        "NH2": 30,
        "OH": 31,
        "CZ": 32,
        "CZ2": 33,
        "CZ3": 34,
        "NZ": 35,
        "OXT": 36,
    }
    atom_types = [
        "N",
        "CA",
        "C",
        "CB",
        "O",
        "CG",
        "CG1",
        "CG2",
        "OG",
        "OG1",
        "SG",
        "CD",
        "CD1",
        "CD2",
        "ND1",
        "ND2",
        "OD1",
        "OD2",
        "SD",
        "CE",
        "CE1",
        "CE2",
        "CE3",
        "NE",
        "NE1",
        "NE2",
        "OE1",
        "OE2",
        "CH2",
        "NH1",
        "NH2",
        "OH",
        "CZ",
        "CZ2",
        "CZ3",
        "NZ",
    ]
    atoms = parsePDB(path)
    num_models = atoms._n_csets

    outputs = []
    for model_idx in range(num_models):
        atoms.setACSIndex(model_idx)
        protein_atoms = atoms.select("protein")
        backbone = protein_atoms.select("backbone")
        other_atoms = atoms.select("not protein and not water")
        water_atoms = atoms.select("water")

        CA_atoms = protein_atoms.select("name CA")
        CA_resnums = CA_atoms.getResnums()
        CA_chain_ids = CA_atoms.getChids()
        CA_icodes = CA_atoms.getIcodes()

        CA_dict = {}
        for i in range(len(CA_resnums)):
            code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]
            CA_dict[code] = i

        xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
        xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)
        for atom_name in atom_types:
            xyz, xyz_m = get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
            xyz_37[:, atom_order[atom_name], :] = xyz
            xyz_37_m[:, atom_order[atom_name]] = xyz_m

        N = xyz_37[:, atom_order["N"], :]
        CA = xyz_37[:, atom_order["CA"], :]
        C = xyz_37[:, atom_order["C"], :]
        O = xyz_37[:, atom_order["O"], :]

        N_m = xyz_37_m[:, atom_order["N"]]
        CA_m = xyz_37_m[:, atom_order["CA"]]
        C_m = xyz_37_m[:, atom_order["C"]]
        O_m = xyz_37_m[:, atom_order["O"]]

        mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

        b = CA - N
        c = C - CA
        a = np.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

        chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
        R_idx = np.array(CA_resnums, dtype=np.int32)
        S = CA_atoms.getResnames()
        S = [restype_3to1[AA] if AA in list(restype_3to1) else "X" for AA in list(S)]
        S = np.array([restype_STRtoINT[AA] for AA in list(S)], np.int32)
        X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

        try:
            Y = np.array(other_atoms.getCoords(), dtype=np.float32)
            Y_t = list(other_atoms.getElements())
            Y_t = np.array(
                [
                    element_dict[y_t.upper()] if y_t.upper() in element_list else 0
                    for y_t in Y_t
                ],
                dtype=np.int32,
            )
            Y_m = (Y_t != 1) * (Y_t != 0)

            Y = Y[Y_m, :]
            Y_t = Y_t[Y_m]
            Y_m = Y_m[Y_m]
        except:
            Y = np.zeros([1, 3], np.float32)
            Y_t = np.zeros([1], np.int32)
            Y_m = np.zeros([1], np.int32)

        output_dict = {}
        output_dict["X"] = torch.tensor(X, device=device, dtype=torch.float32)
        output_dict["mask"] = torch.tensor(mask, device=device, dtype=torch.int32)
        output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
        output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
        output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)

        output_dict["R_idx"] = torch.tensor(R_idx, device=device, dtype=torch.int32)
        output_dict["chain_labels"] = torch.tensor(
            chain_labels, device=device, dtype=torch.int32
        )

        output_dict["chain_letters"] = CA_chain_ids

        mask_c = []
        chain_list = list(set(output_dict["chain_letters"]))
        chain_list.sort()
        for chain in chain_list:
            mask_c.append(
                torch.tensor(
                    [chain == item for item in output_dict["chain_letters"]],
                    device=device,
                    dtype=bool,
                )
            )

        output_dict["mask_c"] = mask_c
        output_dict["chain_list"] = chain_list

        output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)

        output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
        output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)

        outputs.append((output_dict, backbone, other_atoms, CA_icodes, water_atoms))
    return outputs