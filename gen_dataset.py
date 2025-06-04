DATA_PATH = "training/train_multistate.json"
PDB_PATH = "training/pdbs/"

import json

with open(DATA_PATH, "r") as f:
    data = json.load(f)

