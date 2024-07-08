from ocdata.core import Bulk, Adsorbate
from ase import Atoms
from ase.io import read
from modelPrompter import prompt_llm


cifs = prompt_llm("7b", "exp/7b-test-run/checkpoint-84000")


for cif in cifs:
    bulk = Bulk(read(cif.cif, ':', 'cif')[0])
    print(bulk)

