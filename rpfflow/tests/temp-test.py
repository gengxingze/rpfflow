from rpfflow.core.structure import generate_adsorption_structures, optimize_structure, create_mol
from rpfflow.utils.convert import rdkit_to_nx, rdkit_to_ase
from ase.io import read



mol_react = create_mol('[CH3]', add_h=True)                 # CO2 (或简化占位)
mol_prod  = create_mol("C", add_h=True)

slab = read("./Cu.xyz")

ads_candidates, _ = generate_adsorption_structures(
                    adsorbate_ase=rdkit_to_ase(mol_react),
                    slab_ase=slab)
aa = []
for atom in ads_candidates:
    aa.append(optimize_structure(atom))
    print("[CH3]",aa[-1].get_potential_energy())


bb = []
ads_candidates, _ = generate_adsorption_structures(
                    adsorbate_ase=rdkit_to_ase(mol_prod),
                    slab_ase=slab)
for atom in ads_candidates:
    bb.append(optimize_structure(atom))
    print("CH4", bb[-1].get_potential_energy())


print("done")
