from ase.optimize import BFGS
from fairchem.core import OCPCalculator 
from ase.io import read
import torch

def setup_calculator(checkpoint_path:str, rank)->OCPCalculator:
    a = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = a["config"]
    config["local_rank"] = rank
    calc = OCPCalculator(
        config_yml=config,
        cpu=False,
        seed=42
    )
    return calc

def calculate_energy_of_slab(adsorbate_slab, path:str, log_path, calc):
    adsorbate_slab.calc = calc
    
    dyn = BFGS(adsorbate_slab, trajectory=path, logfile=log_path)
    dyn.run(0.05, 100)
    traj = read(path, index=-1)
    traj.get_potential_energy()
    return traj