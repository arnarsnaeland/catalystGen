import io
import os
import argparse
import pandas as pd

from fairchem.data.oc.core import Bulk, Adsorbate, Slab, AdsorbateSlabConfig
from ase.io import read, write
from ase import Atoms
from ase.collections import g2
from ase.db import connect

from modelPrompter import prompt_llm
from calculate import setup_calculator, calculate_energy_of_slab
import database_utils
from catalyst_system import CatalystSystem

#Creates samples using the llm and saves them to a csv file, out_path
def create_llm_samples(args):
    if ".csv" not in args.out_path:
        i = os.environ.get("SLURM_JOB_ID", 0)
        args.out_path = os.path.join(args.out_path, f"samples_{i}.csv") 
    prompt_llm(args)
    return args

#Reads cif files from a csv file, returns a list of cifs
def read_llm_samples(out_path)->list:
    samples = pd.read_csv(out_path, usecols=['cif'])['cif'].tolist()
    atom_obj_list = []
    for sample in samples:
        atom_obj = read(io.StringIO(sample), ':', 'cif')[0]
        atom_obj_list.append(atom_obj)
    #write(f"sample.traj", atom_obj_list, format="traj") #Write to traj file, TODO: remove this line
    return atom_obj_list

def create_bulk(atoms:Atoms)->Bulk:
    return Bulk(bulk_atoms=atoms)

def create_adsorbate(adsorbate:str)->Adsorbate:
    ads = g2[adsorbate]
    return Adsorbate(ads)

def bulk_to_slabs(bulk:Bulk)->list[Slab]:
    #return Slab.from_bulk_get_all_slabs(bulk) #By default gets all slabs up to miller index 2
    #TODO: use commented line above, but for now just get the (1,1,1) slab
    try:
        return Slab.from_bulk_get_specific_millers((1,1,1), bulk)
    except Exception as e:
        print("Error creating slab from bulk")
        print(e)
        return None

def slab_to_adsorbate_slab_config(slab:Slab, adsorbate:Adsorbate)->AdsorbateSlabConfig:
    return AdsorbateSlabConfig(slab, adsorbate, mode="random_site_heuristic_placement", num_sites=4)

def write_to_cif(adsorbate_slab_configs:list[AdsorbateSlabConfig], directory:str):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, adsorbate_slab_config in enumerate(adsorbate_slab_configs):
        write(os.path.join(directory, f"adsorbate_slab_{i}.cif"), adsorbate_slab_config.get_metadata_dict(0)["adsorbed_slab_atomsobject"])

def main(args):
    if args.out_path == "":
        create_llm_samples(args)
    atom_obj_list = read_llm_samples(args.out_path) 
    adsorbate = create_adsorbate(args.adsorbate)
    calc = setup_calculator(args.ml_model_checkpoint)
    # bulks = [create_bulk(atom_obj) for atom_obj in atom_obj_list]
    cs = [CatalystSystem(atom_obj, adsorbate, args.surface_site_sampling_mode, 4) for atom_obj in atom_obj_list]
        
    
    
    bulk_db = connect("bulk.db")
    slab_db = connect("slab.db")
    adsorbate_slab_db = connect("adsorbate_slab.db")
    
    for system in cs:
        system.write_to_db(bulk_db, slab_db, adsorbate_slab_db)
        system.relax_adsorbate_slabs(calc, args.cif_dir)
        system.write_relaxed_adsorbate_slabs_to_db(adsorbate_slab_db)

    # database_utils.write_bulks_to_db(bulks, bulk_db)
    
    # for bulk in bulks:
    #     bulk.slabs = bulk_to_slabs(bulk)
    #     for slab in bulk.slabs:
    #         slab.adsorbate_slab_configs = slab_to_adsorbate_slab_config(slab, adsorbate)
    
    # slabs = [bulk.slabs for bulk in bulks]
    # database_utils.write_slabs_to_db(slabs, slab_db)
    
    # adsorbate_slab_configs = [slab_to_adsorbate_slab_config(slab, adsorbate) for list_of_slabs in slabs for slab in list_of_slabs] #Just a nested for loop
    
    # database_utils.write_adsorbate_slab_configs_to_db(adsorbate_slab_configs, adsorbate_slab_db)
    
    # relaxed_adslabs = []
    
    # for adsorbate_slab_config in adsorbate_slab_configs:
    #     bulk_id = adsorbate_slab_config.slab.bulk.db_id
    #     slab_id = adsorbate_slab_config.slab.db_id
    #     os.makedirs(os.path.join(args.cif_dir, f"bulk{bulk_id}_slab{slab_id}"), exist_ok=True)
    #     for atom_obj in adsorbate_slab_config.atoms_list:
    #         traj_path = os.path.join(args.cif_dir, f"bulk{bulk_id}_slab{slab_id}/adslab{atom_obj.db_id}.traj")
    #         relaxed_adslab = calculate_energy_of_slab(atom_obj, calc, traj_path)
    #         relaxed_adslab.bulk_id = bulk_id
    #         relaxed_adslab.slab_id = slab_id
    #         relaxed_adslab.adslab_id = atom_obj.db_id
    #         relaxed_adslabs.append(relaxed_adslab)
    
    # database_utils.write_adsorbate_slabs_to_db(relaxed_adslabs, adsorbate_slab_db)
    
    print("Done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adsorbate", type=str, required=True)
    parser.add_argument("--surface_site_sampling_mode", type=str, default="random_site_heuristic_placement")
    parser.add_argument("--ml_model_checkpoint", type=str, default="eq2_153M_ec4_allmd.pt")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="")
    parser.add_argument("--cif_dir", type=str, default="cif_files")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--instruction_prompt", type=str, default="")
    args = parser.parse_args()

    main(args)