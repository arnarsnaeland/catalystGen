from ase.db import connect

def write_bulks_to_db(bulks, db):
    with db as db:
        for bulk in bulks:
            id = db.write(bulk.atoms)
            bulk.db_id = id
            
def write_slabs_to_db(slabs, db):
    with db as db:
        for slab_list in slabs:
            for slab in slab_list:
                id = db.write(slab.atoms, bulk_id=slab.bulk.db_id)
                slab.db_id = id
            
def write_adsorbate_slab_configs_to_db(adsorbate_slab_configs, db, relaxed):
    with db as db:
        if relaxed:
            for adsorbate_slab_config in adsorbate_slab_configs:
                for atom_obj in adsorbate_slab_config.atoms_list:
                    id = db.write(atom_obj, bulk_id=adsorbate_slab_config.slab.bulk.db_id, slab_id=adsorbate_slab_config.slab.db_id, relaxed=relaxed, relaxed_id=adsorbate_slab_config.not_relaxed_id)
                    atom_obj.db_id = id
        else:    
            for adsorbate_slab_config in adsorbate_slab_configs:
                for atom_obj in adsorbate_slab_config.atoms_list:
                    id = db.write(atom_obj, bulk_id=adsorbate_slab_config.slab.bulk.db_id, slab_id=adsorbate_slab_config.slab.db_id, relaxed=relaxed)
                    atom_obj.db_id = id