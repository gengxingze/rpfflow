from copy import deepcopy
from ase import Atoms
from rpfflow.core.state import RxnState, SearchNode


def replace_slab(end_node: SearchNode, slab: Atoms) -> SearchNode:
    """
    复制一条完整路径，但将所有 state.slab 替换为新的 slab
    返回新的 end_node（结构完全独立）
    """

    new_parent = None
    new_end_node = None

    # 2. 逐步重建链
    for node in end_node.iter_path():
        state = node.state
        # ⚡ 轻量复制（graphs 直接复用，不 deepcopy）
        new_state = RxnState(
            graphs=state.graphs,  # 共享（假设不被修改）
            stage=state.stage,
            penalty=state.penalty,
            slab=slab  # ✅ 替换
        )
        new_node = SearchNode(
            state=new_state,
            parent=new_parent,
            action=node.action
        )
        new_parent = new_node
        new_end_node = new_node

    return new_end_node

