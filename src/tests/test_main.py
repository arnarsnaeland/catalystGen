from pkg.main import read_llm_samples, cif_to_bulk, create_adsorbate, bulk_to_slab, slab_to_adsorbate_slab_config, write_to_cif, main

from ocdata.core import Bulk, Adsorbate, Slab, AdsorbateSlabConfig

def test_read_llm_samples():
    out_path = "samples_7484885.csv"
    cifs = read_llm_samples(out_path)
    assert len(cifs) == 3
    assert cifs[0] == '''# generated using pymatgen
data_CdSnRu2
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.00000000
_cell_length_b   5.00000000
_cell_length_c   5.00000000
_cell_angle_alpha   59.00000000
_cell_angle_beta   59.00000000
_cell_angle_gamma   59.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   CdSnRu2
_chemical_formula_sum   'Cd1 Sn1 Ru2'
_cell_volume   86.37216793
_cell_formula_units_Z   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Ru  Ru0  1  0.23000000  0.70000000  0.68000000  1
  Ru  Ru1  1  0.73000000  0.20000000  0.18000000  1
  Cd  Cd2  1  0.48000000  0.95000000  0.93000000  1
  Sn  Sn3  1  0.98000000  0.45000000  0.43000000  1
'''

def test_cif_to_bulk():
    cif = '''# generated using pymatgen
data_CdSnRu2
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.00000000
_cell_length_b   5.00000000
_cell_length_c   5.00000000
_cell_angle_alpha   59.00000000
_cell_angle_beta   59.00000000
_cell_angle_gamma   59.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   CdSnRu2
_chemical_formula_sum   'Cd1 Sn1 Ru2'
_cell_volume   86.37216793
_cell_formula_units_Z   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Ru  Ru0  1  0.23000000  0.70000000  0.68000000  1
  Ru  Ru1  1  0.73000000  0.20000000  0.18000000  1
  Cd  Cd2  1  0.48000000  0.95000000  0.93000000  1
  Sn  Sn3  1  0.98000000  0.45000000  0.43000000  1
'''
    bulk = cif_to_bulk(cif)
    assert bulk.atoms.get_chemical_formula() == "CdSnRu2"
    