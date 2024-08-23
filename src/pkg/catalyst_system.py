import os

from fairchem.data.oc.core import Bulk, Adsorbate, Slab, AdsorbateSlabConfig
from ase import Atoms

import database_utils, calculate

class CatalystSystem:
    """
    Takes in a bulk material and an adsorbate.
    Initializes slabs of the bulk material, places the adsorbate on the slab to create adsorbate slab configurations.
    
    Arguments:
    bulk_atoms: Atoms object used to create the bulk material
    adsorbate: Adsorbate molecule
    mode: Method of placing the adsorbate on the slab. Default is "random_site_heuristic_placement".
    num_sites: Number of sites to place the adsorbate on the slab. Default is 100
    num_augmentations_per_site: int
        Number of augmentations of the adsorbate per site. Total number of
        generated structures will be `num_sites` * `num_augmentations_per_site`.
    """
    
    def __init__(
        self,
        bulk_atoms: Atoms,
        adsorbate: Adsorbate,
        mode: str = "random_site_heuristic_placement",
        num_sites: int = 100,
        num_augmentations_per_site: int = 1,
    ):
        self.path = None
        self.calc = None
        self.bulk = Bulk(bulk_atoms=bulk_atoms)
        self.slabs = self.bulk_to_slabs(self.bulk)
        if self.slabs is None:
            return None
        self.adsorbate_slab_configs = [self.slab_to_adsorbate_slab_config(slab, adsorbate, mode, num_sites, num_augmentations_per_site) for slab in self.slabs]
   
    def set_calculator(self, calc):
        self.calc = calc
        
    def set_path(self, path:str):
        self.path = path

    def bulk_to_slabs(self, bulk:Bulk)->list[Slab]:
        # Slab.from_bulk_get_specific_millers((1,1,1), bulk)
        #TODO: use commented line above, but for now just get the (1,1,1) slab
        try:
            return Slab.from_bulk_get_all_slabs(bulk) #By default gets all slabs up to miller index 2
        except Exception as e:
            print("Error creating slab from bulk")
            print(e)
            return None     
        
        
    def slab_to_adsorbate_slab_config(self, slab:Slab, adsorbate:Adsorbate, mode, num_sites, num_augmentations_per_site )->AdsorbateSlabConfig:
        return AdsorbateSlabConfig(slab, adsorbate, num_sites, num_augmentations_per_site, mode = mode)
    
    
    def relax_adsorbate_slabs(self, db):
        for adsorbate_slab_config in self.adsorbate_slab_configs:
            bulk_id = adsorbate_slab_config.slab.bulk.db_id
            slab_id = adsorbate_slab_config.slab.db_id
            os.makedirs(os.path.join(self.path, f"bulk{bulk_id}_slab{slab_id}"), exist_ok=True)
            relaxed_adslabs = []
            for i, atom_obj in enumerate(adsorbate_slab_config.atoms_list):
                log_path = os.path.join(self.path, f"bulk{bulk_id}_slab{slab_id}/adslab{i}.txt")
                traj_path = os.path.join(self.path, f"bulk{bulk_id}_slab{slab_id}/adslab{i}.traj")
                try:
                    relaxed_adslab = calculate.calculate_energy_of_slab(atom_obj, traj_path, log_path, self.calc)
                except Exception as e:
                    print(f"Error relaxing adsorbate slab bulk{bulk_id},slab{slab_id},adslab{i}: {e}")
                    continue
                relaxed_adslab.bulk_id = bulk_id
                relaxed_adslab.slab_id = slab_id
                relaxed_adslab.adslab_id = i
                relaxed_adslabs.append(relaxed_adslab)
            database_utils.write_adsorbate_slabs_to_db(relaxed_adslabs, db)
    
    def write_relaxed_adsorbate_slabs_to_db(self, db):
        database_utils.write_adsorbate_slabs_to_db(self.relaxed_adslabs, db)            
    
    def write_to_db(self, bulk_db, slab_db):
        database_utils.write_bulk_to_db(self.bulk, bulk_db)
        database_utils.write_slabs_to_db(self.slabs, slab_db) 