import io
import os
import argparse
import pandas as pd
import torch.multiprocessing as multiprocessing
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from queue import Empty
import cupy
from cupy import cuda
import numpy as np
import torch
import submitit

from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig
from ase.io import read, write
from ase.collections import g2
from ase.db import connect

from modelPrompter import prompt_llm
from calculate import setup_calculator
from catalyst_system import CatalystSystem

#Creates samples using the llm and saves them to a csv file, out_path
def create_llm_samples(args):
    if ".csv" not in args.out_path:
        i = os.environ.get("SLURM_JOB_ID", 0)
        args.out_path = os.path.join(args.out_path, f"samples_{i}.csv") 
    prompt_llm(args)
    return args

#Reads cif files from a csv file, returns a list of atom objects
def read_llm_samples(out_path)->list:
    samples = pd.read_csv(out_path, usecols=['cif'])['cif'].tolist()
    atom_obj_list = []
    for i, sample in enumerate(samples):
        atom_obj = read(io.StringIO(sample), ':', 'cif')[0]
        atom_obj_list.append(atom_obj)
        write(f"sample{i}.cif", atom_obj, format="cif") #Write each generated cif to a cif file TODO: remove this line
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
    if args.out_path == "":
        create_llm_samples(args)
    atom_obj_list = read_llm_samples(args.out_path) 
    adsorbate = create_adsorbate(args.adsorbate)
    cs = [CatalystSystem(atom_obj, adsorbate, args.surface_site_sampling_mode) for atom_obj in atom_obj_list]
    
    #If any of the CatalystSystems did not manage to create valid slabs, remove them from the list
    cs = [i for i in cs if i is not None]
    

    
    bulk_db = connect("bulk-dist.db")
    slab_db = connect("slab-dist.db")
    
    calc = setup_calculator(args.ml_model_checkpoint)

    
    for system in cs:
        system.write_to_db(bulk_db, slab_db)
        system.set_path(args.traj_dir)
        system.set_calculator(calc)
    
    return cs
  
def compute_energy(catalyst_system):
    catalyst_system.relax_adsorbate_slabs()
    return catalyst_system


#Split a list into n batches as evenly as possible    
def batched(lst, num_batches):
    return np.array_split(lst, num_batches)



class Worker(multiprocessing.Process):
    def __init__(self, queue, gpu_id):
        super().__init__()
        self.gpu_id = gpu_id
        self.queue = queue
        self.calc = setup_calculator("eq2_153M_ec4_allmd.pt")
        
    def run(self):
        print(f"Running on GPU {self.gpu_id} with {cuda.Device(self.gpu_id).pci_bus_id}")
        output = []
        while True:
            try:
                system = self.queue.get(timeout=10)
                print(f"Running on GPU {self.gpu_id}, computing for bulk{system.adsorbate_slab_configs[0].slab.bulk.db_id}, slab{system.adsorbate_slab_configs[0].slab.db_id}")
                system.set_calculator(self.calc)
                output.append(compute_energy(system))
                del system
            except Empty:
                print(f"Worker {self.gpu_id} found empty queue")
                break
        print(f"Worker {self.gpu_id} finished")
        return output

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
    parser.add_argument("--traj_dir", type=str, default="traj_files")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--instruction_prompt", type=str, default="")
    parser.add_argument("--slurm_partition", type=str)
    parser.add_argument("--distributed", type=str, default="False")
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()
    
    adsorbate_slab_db = connect("adsorbate_slab-dist.db")
    
    cs = main(args)
    
    if args.distributed == "True":

        # Create a queue to hold the systems
        queue = multiprocessing.Queue()

        # Add the systems to the queue
        for system in cs:
            queue.put(system)

        # Create a list to hold the worker processes
        workers = []

        # Create and start the worker processes
        for gpu_id in range(args.num_gpus):
            worker = Worker(queue, gpu_id)
            worker.start()
            workers.append(worker)

        # Wait for all worker processes to finish
        for worker in workers:
            worker.join()

        # Collect the results from the worker processes
        results = []
        for worker in workers:
            results.extend(worker.run())

        # Write the relaxed adsorbate slabs to the database
        for result in results:
            result.write_relaxed_adsorbate_slabs_to_db(adsorbate_slab_db)
    else: #Run on single gpu
        for system in cs:
            compute_energy(system)
            system.write_relaxed_adsorbate_slabs_to_db(adsorbate_slab_db)
