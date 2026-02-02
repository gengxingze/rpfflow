import numpy as np
import networkx as nx
from copy import deepcopy
from typing import Dict, Optional, List
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.io import write
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from rpfflow.utils.convert import nx_to_ase, rdkit_to_nx



# ============================================================
# RDKit molecule construction
# ============================================================
def create_mol(
    smiles: Optional[str] = None,
    structure: Optional[str] = None,
    add_h: bool = False,
    optimize: bool = False
) -> Chem.Mol:
    """
    Create an RDKit molecule from SMILES or structure file.

    Args:
        smiles: SMILES string
        structure: mol file path
        add_h: whether to add hydrogens
        optimize: whether to run UFF optimization

    Returns:
        RDKit Mol object with 3D conformer.
    """
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
    elif structure is not None:
        mol = Chem.MolFromMolFile(structure)
    else:
        raise ValueError("Either smiles or structure must be provided.")

    if mol is None:
        raise ValueError("Failed to create RDKit Mol.")

    if add_h:
        mol = Chem.AddHs(mol)

    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL, catchErrors=True)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if optimize:
        AllChem.UFFOptimizeMolecule(mol)

    return mol


def generate_coordinate(mol: Chem.Mol) -> Chem.Mol:
    """
    Safely generate 3D coordinates without enforcing full sanitization.
    Suitable for TS or broken structures.
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError("Input must be an RDKit Mol object.")

    mol = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SANITIZE_SETAROMATICITY |
                        Chem.SANITIZE_FINDRADICALS |
                        Chem.SANITIZE_SETCONJUGATION
        )
    except Exception as e:
        print("[WARN] Partial sanitize failed (allowed):", e)
        mol.UpdatePropertyCache(strict=False)

    if mol.GetNumConformers() == 0:
        params = AllChem.ETKDG()
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("3D embedding failed.")

    return mol


# ============================================================
# Common molecular graph templates
# ============================================================
def create_common_molecules() -> Dict[str, nx.Graph]:
    """
    Build common molecular graph templates.
    """
    molecules = {}

    smiles_dict = {
        "OH_": "[OH-]",
        "H": "[H]",
        "H2O": "O",
        "CO2": "O=C=O",
    }

    for name, smi in smiles_dict.items():
        mol = create_mol(smi, add_h=True, optimize=True)
        molecules[name] = rdkit_to_nx(mol)
        print(f"[✓] {name} generated.")

    from rpfflow.rules.basica import update_valence

    # O=C-O-R
    G1 = nx.Graph()
    G1.add_nodes_from([
        (0, {"symbol": "O"}),
        (1, {"symbol": "C"}),
        (2, {"symbol": "O"}),
        (3, {"symbol": "F"}),  # active site
    ])
    G1.add_edge(0, 1, bond_order=2.0)
    G1.add_edge(1, 2, bond_order=1.0)
    G1.add_edge(2, 3, bond_order=1.0)
    update_valence(G1)
    molecules["OCOR"] = G1

    # O=C(R)-O
    G2 = nx.Graph()
    G2.add_nodes_from([
        (0, {"symbol": "O"}),
        (1, {"symbol": "C"}),
        (2, {"symbol": "O"}),
        (3, {"symbol": "F"}),
    ])
    G2.add_edge(0, 1, bond_order=2.0)
    G2.add_edge(1, 2, bond_order=1.0)
    G2.add_edge(1, 3, bond_order=1.0)
    update_valence(G2)
    molecules["OCRO"] = G2

    R = deepcopy(molecules["H"])
    R.nodes[0]["symbol"] = "F"
    update_valence(R)
    molecules["F"] = R

    return molecules



def rotate_F(atoms: Atoms):
    pos = atoms.get_positions()

    # === 1. 找 F 原子 ===
    symbols = atoms.get_chemical_symbols()
    if "F" not in symbols:
        raise ValueError("结构中没有 F 原子")
    iF = symbols.index("F")
    rF = pos[iF]

    # === 2. 找最近相连原子 ===
    dists = np.linalg.norm(pos - rF, axis=1)
    dists[iF] = 1e10
    iN = np.argmin(dists)
    rN = pos[iN]

    v = rN - rF
    v = v / np.linalg.norm(v)

    z = np.array([0.0, 0.0, 1.0])

    # === 3. 计算旋转矩阵 (Rodrigues) ===
    axis = np.cross(v, z)
    sin_theta = np.linalg.norm(axis)
    cos_theta = np.dot(v, z)

    if sin_theta < 1e-8:
        R = np.eye(3)
    else:
        axis = axis / sin_theta
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + K * sin_theta + K @ K * (1 - cos_theta)

    # === 4. 以 F 为中心旋转 ===
    pos_rot = (pos - rF) @ R.T + rF
    atoms.set_positions(pos_rot)

    # === 5. 保证 F 的 Z 最小 ===
    z_coords = atoms.get_positions()[:, 2]
    if atoms.positions[iF, 2] > np.min(z_coords):
        atoms.positions[:, 2] *= -1

    return atoms


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
    from mace.calculators import mace_mp
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


def get_reference_structure(slab_ase):
    from ase.build import molecule
    """
    创建参考分子（H, H2O）并进行几何优化。
    返回一个包含优化后 ASE 对象和原始 Slab 的字典。
    """
    # 1. 创建初始结构
    # 'H' 原子的 potential_energy 建议直接设为参考值，或者创建孤立原子
    h_atom = molecule('H')
    # h_atom.set_cell([10, 10, 10])  # 给孤立原子一个足够大的真空层
    h_atom.center()

    h2o_mol = molecule('H2O')
    # h2o_mol.set_cell([12, 12, 12])
    h2o_mol.center()

    # 2. 几何优化
    # 注意：孤立原子的 optimize_structure 可能会因为没有受力而跳过
    print("--- 优化参考结构 H ---")
    opt_h = optimize_structure(h_atom, device="cpu", fmax=0.05, steps=200)

    print("--- 优化参考结构 H2O ---")
    opt_h2o = optimize_structure(h2o_mol, device="cpu", fmax=0.05, steps=200)

    opt_f = optimize_structure(slab_ase, device="cpu", fmax=0.05, steps=200)
    # 3. 构建结果字典
    reference_dict = {
        "H": opt_h,
        "H2O": opt_h2o,
        "F": opt_f  # 将传入的 slab 作为 F 的参考（对应表面）
    }

    return reference_dict


def save_reaction_path(history, filename="reaction_path.extxyz"):
    from ase.calculators.singlepoint import SinglePointCalculator
    path_frames = []

    for entry in history:
        structures = entry["stable_structures"]  # 这是一个 [Atoms, Atoms, ...] 列表

        if not structures:
            continue

        for i, atoms in enumerate(structures):
            # 1. 创建副本，防止修改原始数据
            temp_atoms = atoms.copy()

            # 2. 提取并注入能量和受力
            # 这样 write 时，每个碎片都会带有它自己的物理信息
            try:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                temp_atoms.calc = SinglePointCalculator(
                    temp_atoms, energy=energy, forces=forces
                )
            except Exception as e:
                # 如果没有计算器信息，则跳过计算器设置
                print(f"Step {entry['step']} fragment {i} has no energy/force: {e}")

            # 3. 注入元数据
            temp_atoms.info['step'] = entry['step']
            temp_atoms.info['action'] = entry['action']
            temp_atoms.info['fragment_id'] = i  # 标记这是该步骤中的第几个碎片

            path_frames.append(temp_atoms)

    # 4. 写入文件
    write(filename, path_frames)
    print(f"成功导出 {len(path_frames)} 个独立碎片结构至 {filename}")

