from ase.optimize import BFGS
from fairchem.core import OCPCalculator 
from ase.io import read

#Given a path to ML model checkpoint, creates a calculator object
def setup_calculator(checkpoint_path:str)->OCPCalculator:
    calc = OCPCalculator(
        checkpoint_path=checkpoint_path,
        cpu=False,
        seed=42
    )
    return calc

# Relaxes given adsorbate slab using BFGS, storing the generated trajectory file in the path
# Returns the final relaxed adsorbate slab, with energy calculated
def calculate_energy_of_slab(adsorbate_slab, path:str, log_path, calc):
    adsorbate_slab.calc = calc
    
    dyn = BFGS(adsorbate_slab, trajectory=path, logfile=log_path)
    dyn.run(0.05, 100)
    traj = read(path, index=-1) #Read final relaxed system from trajectory file
    traj.get_potential_energy() #Calculate energy of the final atoms object
    return traj