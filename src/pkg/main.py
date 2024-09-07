import io
import os
import argparse
import pandas as pd
import torch.multiprocessing as multiprocessing
from queue import Empty
import numpy as np

from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig
from ase.io import read, write
from ase.collections import g2
from ase.db import connect

from modelPrompter import prompt_llm
from calculate import setup_calculator
from catalyst_system import CatalystSystem

#Creates samples using the llm and saves them to a csv file, out_path
def create_llm_samples(args):
    i = os.environ.get("SLURM_JOB_ID", 0)
    args.samples_file = os.path.join(args.out_path, f"samples_{i}.csv") 
    prompt_llm(args)
    return args

#Reads cif files from a csv file, returns a list of atom objects
def read_llm_samples(args)->list:
    samples = pd.read_csv(args.samples_file, usecols=['cif'])['cif'].tolist()
    atom_obj_list = []
    for i, sample in enumerate(samples):
        atom_obj = read(io.StringIO(sample), ':', 'cif')[0]
        atom_obj_list.append(atom_obj)
        write(os.path.join(args.out_path, f"sample{i}.cif"), atom_obj, format="cif") #Write each generated cif to a cif file TODO: remove this line
    return atom_obj_list

def create_adsorbate(adsorbate:str)->Adsorbate:
    ads = g2[adsorbate]
    return Adsorbate(ads)

#Write a list of adsorbate slab configs to a directory as cif files
def write_to_cif(adsorbate_slab_configs:list[AdsorbateSlabConfig], directory:str):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, adsorbate_slab_config in enumerate(adsorbate_slab_configs):
        write(os.path.join(directory, f"adsorbate_slab_{i}.cif"), adsorbate_slab_config.get_metadata_dict(0)["adsorbed_slab_atomsobject"])

def main(args):
    os.makedirs(args.out_path, exist_ok=False)
    if args.samples_file == "":
        create_llm_samples(args)
    atom_obj_list = read_llm_samples(args)
    adsorbate_list = args.adsorbate.split(",") 
    adsorbates = [create_adsorbate(adsorbate) for adsorbate in adsorbate_list]
    cs = []
    for adsorbate in adsorbates:
        cs.extend([CatalystSystem(atom_obj, adsorbate, args.surface_site_sampling_mode) for atom_obj in atom_obj_list])
    #If any of the CatalystSystems did not manage to create valid slabs, remove them from the list
    cs = [i for i in cs if i is not None]
    
    
    bulk_db = connect(os.path.join(args.out_path, "bulk.db"))
    slab_db = connect(os.path.join(args.out_path, "slab.db"))

    calc = setup_calculator(args.ml_model_checkpoint)

    
    for system in cs:
        system.write_to_db(bulk_db, slab_db)
        system.set_path(args.out_path)
        system.set_calculator(calc)
    
    return cs
  
def compute_energy(catalyst_system, db):
    catalyst_system.relax_adsorbate_slabs(db)
    return catalyst_system


#Split a list into n batches as evenly as possible    
def batched(lst, num_batches):
    return np.array_split(lst, num_batches)



class Worker(multiprocessing.Process):
    def __init__(self, queue, id, db):
        super().__init__()
        self.id = id
        self.queue = queue
        self.db = db
        
    def run(self):
        while True:
            try:
                system = self.queue.get(timeout=10)
                compute_energy(system, self.db)
                del system
            except Empty:
                print(f"Worker {self.id} found empty queue")
                break
        print(f"Worker {self.id} finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adsorbate", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--surface_site_sampling_mode", type=str, default="random_site_heuristic_placement")
    parser.add_argument("--ml_model_checkpoint", type=str, default="eq2_153M_ec4_allmd.pt")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--samples_file", type=str, default="") #If you want to provide a file with bulk structures to use, otherwise will generate samples
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--instruction_prompt", type=str, default="")
    parser.add_argument("--distributed", type=str, default="False")
    parser.add_argument("--num_processes", type=int, default=1)
    
    args = parser.parse_args()
    
    adsorbate_slab_db = connect(os.path.join(args.out_path, "adsorbate_slab.db"))
    
    cs = main(args)
    
    if args.distributed == "True":
        multiprocessing.set_start_method("spawn")
        # Create a queue to hold the systems
        queue = multiprocessing.Queue()

        # Add the systems to the queue
        for system in cs:
            queue.put(system)

        # Create a list to hold the worker processes
        workers = []

        # Create and start the worker processes
        for id in range(args.num_processes):
            worker = Worker(queue, id, adsorbate_slab_db)
            worker.start()
            workers.append(worker)

        # Wait for all worker processes to finish
        for worker in workers:
            worker.join()

        print("All workers finished")
        
    else: #Run a single process
        for system in cs:
            compute_energy(system, adsorbate_slab_db)