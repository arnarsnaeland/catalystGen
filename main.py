from ocdata.core import Bulk, Adsorbate, Slab, AdsorbateSlabConfig
from ase.io import read, write
from ase import Atoms
from modelPrompter import prompt_llm
import io
import argparse
import pandas as pd
import os

# Define the Miller index for the surface (just for testing, llm will generate this)
#MODEL_NAME = "7b"
#MODEL_PATH = "checkpoint-80000"
#"The material is Ruthenium (Ru)."
#Creates samples using the llm and saves them to a csv file, out_path
def create_llm_samples(args):
    if ".csv" not in args.out_path:
        i = os.environ.get("SLURM_JOB_ID", 0)
        args.out_path = os.path.join(args.out_path, f"samples_{i}.csv") 
    prompt_llm(args)

#Reads cif files from a csv file, returns a list of cifs
def read_llm_samples(out_path)->list:
    return pd.read_csv(out_path, usecols=['cif'])['cif'].tolist()

def cif_to_bulk(cif:str)->Bulk:
    return Bulk(read(io.StringIO(cif), ':', 'cif')[0])

def create_adsorbate(adsorbate:str)->Adsorbate:
    #TODO: implement specific adsorbate creation
    return Adsorbate()

def bulk_to_slab(bulk:Bulk)->list[Slab]:
    return Slab.from_bulk_get_all_slabs(bulk) #By default gets all slabs up to miller index 2

def slab_to_adsorbate_slab_config(slab:Slab, adsorbate:Adsorbate)->AdsorbateSlabConfig:
    return AdsorbateSlabConfig(slab, adsorbate, mode="random_site_heuristic_placement", num_sites=100)

def run(args):
    create_llm_samples(args)
    cifs = read_llm_samples(args.out_path)
    bulks = [cif_to_bulk(cif) for cif in cifs]
    slabs = [bulk_to_slab(bulk) for bulk in bulks]
    adsorbate = create_adsorbate(args.adsorbate)
    adsorbate_slab_configs = [slab_to_adsorbate_slab_config(slab, adsorbate) for slab in slabs]
    print(adsorbate_slab_configs[0].get_metadata_dict())
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--instruction_prompt", type=str, default="")
    parser.add_argument("--adsorbate", type=str, default="")
    args = parser.parse_args()

    run(args)


