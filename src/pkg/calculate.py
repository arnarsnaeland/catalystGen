from ase.optimize import BFGS
from fairchem.core import OCPCalculator 
from ase.io import read

def setup_calculator(checkpoint_path:str)->OCPCalculator:
    calc = OCPCalculator(
        checkpoint_path=checkpoint_path,
        cpu=False
    )
    return calc

def calculate_energy_of_slab(adsorbate_slab, path:str, calc):
    adsorbate_slab.calc = calc
    
    dyn = BFGS(adsorbate_slab, trajectory=path)
    dyn.run(0.2, 100)
    traj = read(path, index=-1)
    traj.get_potential_energy()
    traj.not_relaxed_id = adsorbate_slab.db_id
    return traj