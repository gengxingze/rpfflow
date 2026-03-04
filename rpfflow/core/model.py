import logging
import time
from collections import deque
from rpfflow.core.action import HydrogenationAction, DissociationAction, CCCouplingAction
from rpfflow.core.state import RxnState, SearchNode
from rpfflow.rules.matchs import is_duplicate

logger = logging.getLogger(__name__)


def bfs_search(initial_state: RxnState, target_graph, n_hydrogen=8, rules=None, max_paths=5):

    if rules is None:
        rules = [HydrogenationAction(), DissociationAction(), CCCouplingAction()]

    root_node = SearchNode(state=initial_state)
    open_queue = deque([root_node])

    found_paths = []

    # ---------------------------
    # 🔥 统计变量
    # ---------------------------
    iteration_count = 0
    generated_states = 0
    pruned_by_h = 0
    start_time = time.time()

    logger.info("BFS 搜索开始")

    while open_queue:
        iteration_count += 1
        current_node = open_queue.popleft()
        state = current_node.state

        # 每 1000 步打印一次进度（避免刷屏）
        if iteration_count % 1000 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"迭代: {iteration_count}, "
                f"队列长度: {len(open_queue)}, "
                f"当前深度: {current_node.depth}, "
                f"已生成状态: {generated_states}, "
                f"剪枝(H超限): {pruned_by_h}, "
                f"耗时: {elapsed:.2f}s"
            )

        # ---------------------------
        # 1️⃣ 目标检查
        # ---------------------------
        if is_duplicate(target_graph, state.graphs):
            found_paths.append(current_node)

            logger.info(
                f"!!! 找到生成物路径！"
                f"已找到: {len(found_paths)}/{max_paths}, "
                f"深度: {current_node.depth}, "
                f"累计迭代: {iteration_count}"
            )

            if len(found_paths) >= max_paths:
                elapsed = time.time() - start_time
                logger.info(
                    f">>>>达到最大路径数 {max_paths}，停止搜索。\n"
                    f"总迭代: {iteration_count}, "
                    f"总生成状态: {generated_states}, "
                    f"总剪枝: {pruned_by_h}, "
                    f"总耗时: {elapsed:.2f}s"
                )
                return found_paths

            continue

        # ---------------------------
        # 2️⃣ 规则驱动演化
        # ---------------------------
        for rule in rules:
            for next_state, action_desc, h_cost in rule.apply(state):

                total_h_cost = current_node.cumulative_h_cost + h_cost

                if total_h_cost > n_hydrogen:
                    pruned_by_h += 1
                    continue

                if next_state:
                    generated_states += 1

                    child_node = SearchNode(
                        state=next_state,
                        parent=current_node,
                        action=action_desc,
                        step_h_cost=h_cost
                    )

                    open_queue.append(child_node)

    # ---------------------------
    # 搜索结束
    # ---------------------------
    elapsed = time.time() - start_time

    if not found_paths:
        logger.warning(
            f"XXX 搜索结束，未找到路径。\n"
            f"总迭代: {iteration_count}, "
            f"总生成状态: {generated_states}, "
            f"总剪枝: {pruned_by_h}, "
            f"总耗时: {elapsed:.2f}s"
        )
    else:
        logger.info(
            f"搜索结束。\n"
            f"总迭代: {iteration_count}, "
            f"总生成状态: {generated_states}, "
            f"总剪枝: {pruned_by_h}, "
            f"总耗时: {elapsed:.2f}s"
        )

    return found_paths

