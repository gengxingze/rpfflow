from ase.optimize import BFGS
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from graph_mm.molgraph import nx_to_ase
from pathways.gp_data import StructureItem

def optimize_structure(atoms, device="cpu",
                       fmax=0.05, steps=200, fix_mask=None):
    """
    对 atoms 对象执行结构优化
    - atoms: ASE Atoms
    - model: MACE 势模型大小，例如 small / medium / large
    - device: cpu 或 cuda
    - fmax: 最大力阈值 (eV/Å)
    - steps: 最大优化步数
    - fix_mask: 用于固定部分原子的布尔列表 (可选)
    """
    # 固定原子
    if fix_mask is not None:
        atoms.set_constraint(FixAtoms(mask=fix_mask))

    # 构造 MACE 势计算器
    calc = mace_mp(model="small", device='cpu')
    atoms.calc = calc

    # BFGS 优化器
    dyn = BFGS(atoms, logfile="opt.log")
    dyn.run(fmax=fmax, steps=steps)

    return atoms


def generate_adsorption_structures(adsorbate_ase, slab_ase, repeat=(1, 1, 1),min_lw=4.0):
    """
    Generate adsorption structures using pymatgen, input/output in ASE Atoms.

    Parameters
    ----------
    slab_ase : ase.Atoms
        Slab structure (with vacuum).
    adsorbate_ase : ase.Atoms
        Adsorbate molecule.
    repeat : tuple
        Supercell for adsorption search.
    min_lw : float
        Minimum vacuum thickness.

    Returns
    -------
    ads_structures_ase : list of ase.Atoms
        Generated adsorption structures.
    ads_sites : dict
        Adsorption sites from pymatgen (ontop/bridge/hollow/all).
    """

    adaptor = AseAtomsAdaptor()

    # ASE -> pymatgen
    slab_pm = adaptor.get_structure(slab_ase)
    from pymatgen.core import Molecule
    ads_species = adsorbate_ase.get_chemical_symbols()
    ads_coords = adsorbate_ase.get_positions()
    ads_pm = Molecule(species=ads_species, coords=ads_coords)

    # 构建吸附位点搜索器
    asf = AdsorbateSiteFinder(slab_pm)

    # 找位点
    ads_sites = asf.find_adsorption_sites()


    ads_mols = [ads_pm]

    # 生成吸附结构
    ads_structures_pm = []
    for mol in ads_mols:
        ads_structures_pm += asf.generate_adsorption_structures(
            mol,
            repeat=list(repeat),
            min_lw=min_lw,
        )

    def pmg_to_ase_drop_tags(struct):
        struct = struct.copy()

        # 删 structure 级别
        struct.site_properties.pop("tags", None)

        # 删 site 级别（关键）
        for site in struct.sites:
            if "tags" in site.properties:
                site.properties.pop("tags", None)
            if "tag" in site.properties:
                site.properties.pop("tag", None)

        return adaptor.get_atoms(struct)
    # pymatgen -> ASE
    ads_structures_ase = [pmg_to_ase_drop_tags(s) for s in ads_structures_pm]


    return ads_structures_ase, ads_sites


from rules.struct import rotate_F
class HandleStructure:
    def __init__(self, reaction_path, ads_sites=None):
        self.ads_sites = ads_sites
        self.steps = []
        self._build_steps(reaction_path[:-2])

    def _build_steps(self, reaction_path):
        for i, state in enumerate(reaction_path):
            ase_atoms = nx_to_ase(state["state"].graph[0])
            ase_atoms = rotate_F(ase_atoms)
            bb = ase_atoms.copy()
            # 找到所有 F 原子的索引
            f_indices = [i for i, s in enumerate(bb.get_chemical_symbols()) if s == "F"]
            # 删除（支持一次删多个）
            bb.pop(f_indices[0])

            from ase.build import fcc100
            slab = fcc100('Cu',
                          size=(3, 3, 3),  # 表面尺寸: 4x4, 厚度4层
                          a=3.615,  # Cu 晶格常数 (Å)
                          vacuum=12.0)  # 真空层厚度 (Å)
            cc = generate_adsorption_structures(adsorbate_ase=bb, slab_ase=slab)
            pp = []
            for i, s in enumerate(cc[0]):
                s = optimize_structure(s, device="cpu")

            step = StructureItem(
                graph=None,
                ase=ase_atoms,
                ads_ase=s,
                aligned=False,
                meta={}
            )
            self.steps.append(s)
