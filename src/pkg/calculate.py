from ase.optimize import LBFGS
from fairchem.core import OCPCalculator

def setup_calculator(checkpoint_path:str)->OCPCalculator:
    calc = OCPCalculator(
        checkpoint_path=checkpoint_path,
        cpu=False
    )
    return calc

def calculate_energy_of_slab(adsorbate_slab, calc):
    adsorbate_slab.calc = calc

    dyn = LBFGS(adsorbate_slab, trajectory=f"{adsorbate_slab.get_chemical_formula()}.traj")
    return dyn.run(0.05, 100)