from ocdata.core import Bulk, Adsorbate, Slab, AdsorbateSlabConfig
from ase import Atoms
from ase.io import read
from modelPrompter import prompt_llm
import os

# Define the Miller index for the surface (just for testing, llm will generate this)
MILLER_IDX = (1, 1, 1)
MODEL_NAME = "7b"
MODEL_PATH = "/checkpoint-80000"

cifs = prompt_llm(MODEL_NAME, MODEL_PATH, instruction_prompt="The material is Ruthenium (Ru)")


for cif in cifs:
    bulk = Bulk(read(cif, ':', 'cif')[0])
    print(bulk)
    adsorbate = Adsorbate()
    print(adsorbate)
    slabs = Slab.from_bulk_get_specific_millers(MILLER_IDX, bulk)
    print(slabs)
    random_adslabs = AdsorbateSlabConfig(slabs[0], adsorbate, mode="random_site_heuristic_placement", num_sites=100)
    print("Random Adsorbate Slab Config:")
    print(random_adslabs)




