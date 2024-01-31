import pickle
import torch
import os
import io

mypath = "experiments/pickles"

onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

for i, pkl_file in enumerate(onlyfiles):
    print(f"===== {i, pkl_file} =====")

    with open(os.path.join(mypath, pkl_file), "rb") as f:
        obj = CPU_Unpickler(f).load()

    name = pkl_file.split(".")[0]
    torch.save(obj, f"{mypath}/{name}.pt")

