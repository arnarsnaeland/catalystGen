from ase.optimize import LBFGS
from fairchem.core import OCPCalculator 

def setup_calculator(checkpoint_path:str)->OCPCalculator:
    calc = OCPCalculator(
        checkpoint_path=checkpoint_path,
        cpu=False
    )
    return calc

def calculate_energy_of_slab(adsorbate_slab, calc, path:str):
    adsorbate_slab.calc = calc

    dyn = LBFGS(adsorbate_slab, trajectory=path)
    dyn.run(0.2, 100)