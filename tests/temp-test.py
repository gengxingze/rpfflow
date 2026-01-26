def test_generate_adsorption_structures():
    from ase.build import molecule, fcc100
    co = molecule("CO")
    # co.center(vacuum=10.0)
    slab = fcc100('Cu',
                  size=(3, 3, 3),  # 表面尺寸: 4x4, 厚度4层
                  a=3.615,  # Cu 晶格常数 (Å)
                  vacuum=12.0)  # 真空层厚度 (Å)
    from pathways.compute import generate_adsorption_structures
    structure, _ = generate_adsorption_structures(adsorbate_ase=co, slab_ase=slab)
    from ase.io import write
    write("../test_generate_adsorption_structures.extxyz", structure)

if __name__ == '__main__':
    test_generate_adsorption_structures()
    print("Done")