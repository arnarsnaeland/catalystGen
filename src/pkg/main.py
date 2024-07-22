import io
import os
import argparse
import pandas as pd

from ocdata.core import Bulk, Adsorbate, Slab, AdsorbateSlabConfig
from ase.io import read, write
from ase import Atoms
from ase.collections import g2

from pkg.modelPrompter import prompt_llm
from pkg.calculate import setup_calculator, calculate_energy_of_slab

#Creates samples using the llm and saves them to a csv file, out_path
def create_llm_samples(args):
    if ".csv" not in args.out_path:
        i = os.environ.get("SLURM_JOB_ID", 0)
        args.out_path = os.path.join(args.out_path, f"samples_{i}.csv") 
    prompt_llm(args)

#Reads cif files from a csv file, returns a list of cifs
def read_llm_samples(out_path)->list:
    samples = pd.read_csv(out_path, usecols=['cif'])['cif'].tolist()
    atom_obj_list = []
    for sample in samples:
        atom_obj_list.append(read(io.StringIO(sample), ':', 'cif')[0])
    write(f"sample.traj", atom_obj_list, format="traj") #Write to traj file, TODO: remove this line
    return atom_obj_list

def create_bulk(atoms:Atoms)->Bulk:
    return Bulk(bulk_atoms=atoms)

def create_adsorbate(adsorbate:str)->Adsorbate:
    ads = g2[adsorbate]
    return Adsorbate(ads)

def bulk_to_slab(bulk:Bulk)->list[Slab]:
    #return Slab.from_bulk_get_all_slabs(bulk) #By default gets all slabs up to miller index 2
    #TODO: use commented line above, but for now just get the (1,1,1) slab
    return Slab.from_bulk_get_specific_millers((1,1,1), bulk)

def slab_to_adsorbate_slab_config(slab:Slab, adsorbate:Adsorbate)->AdsorbateSlabConfig:
    return AdsorbateSlabConfig(slab, adsorbate, mode="random_site_heuristic_placement", num_sites=100)

def write_to_cif(adsorbate_slab_configs:list[AdsorbateSlabConfig], directory:str):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, adsorbate_slab_config in enumerate(adsorbate_slab_configs):
        write(os.path.join(directory, f"adsorbate_slab_{i}.cif"), adsorbate_slab_config.get_metadata_dict(0)["adsorbed_slab_atomsobject"])

def main(args):
    create_llm_samples(args)
    cifs = read_llm_samples(args.out_path) 
    bulks = [create_bulk(cif) for cif in cifs]
    slabs = [bulk_to_slab(bulk) for bulk in bulks] #returns a list of lists of slabs
    adsorbate = create_adsorbate(args.adsorbate)
    adsorbate_slab_configs = [slab_to_adsorbate_slab_config(slab, adsorbate) for list_of_slabs in slabs for slab in list_of_slabs] #Just a nested for loop
    write_to_cif(adsorbate_slab_configs, args.cif_dir)
    calc = setup_calculator(args.ml_model_checkpoint)
    calculate_energy_of_slab(adsorbate_slab_configs[0].atoms_list[0], calc)
    print("Done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adsorbate", type=str, required=True)
    parser.add_argument("--surface_site_sampling_mode", type=str, default="random_site_heuristic_placement")
    parser.add_argument("--ml-model-checkpoint", type=str, default="eq2_153M_ec4_allmd.pt")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="")
    parser.add_argument("--cif_dir", type=str, default="cif_files")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--instruction_prompt", type=str, default="")
    args = parser.parse_args()

    main(args)