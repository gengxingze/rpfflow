import logging
import time
from collections import deque
from rpfflow.core.action import HydrogenationAction, DissociationAction, CouplingAction, AssociationAction
from rpfflow.core.state import RxnState, SearchNode
from rpfflow.rules.matchs import is_duplicate

logger = logging.getLogger(__name__)


def bfs_search(initial_state: RxnState, target_graph, n_hydrogen=8, rules=None, max_paths=100):

    if rules is None:
        rules = [HydrogenationAction(), DissociationAction(), CouplingAction(), AssociationAction()]

    root_node = SearchNode(state=initial_state)
    open_queue = deque([root_node])

    found_paths = []

    iteration_count = 0
    generated_states = 0
    pruned_by_h = 0

    start_time = time.time()

    # --------------------------------
    # rule 统计
    # --------------------------------
    rule_stats = {
        type(rule).__name__: {
            "calls": 0,
            "generated": 0,
            "none_state": 0,
            "pruned_h": 0,
            "exceptions": 0
        }
        for rule in rules
    }

    logger.info("BFS 搜索开始")

    while open_queue:

        iteration_count += 1
        current_node = open_queue.popleft()
        state = current_node.state
        queue_size = len(open_queue)

        if iteration_count % 1 == 0:
            elapsed = time.time() - start_time

            logger.info(
                f"迭代:{iteration_count} | "
                f"队列:{len(open_queue)} | "
                f"深度:{current_node.depth} | "
                f"生成:{generated_states} | "
                f"剪枝(H):{pruned_by_h} | "
                f"耗时:{elapsed:.2f}s"
            )

        # -------------------------
        # 目标检测
        # -------------------------
        if is_duplicate(target_graph, state.graphs):

            found_paths.append(current_node)

            logger.info(
                f"找到路径 {len(found_paths)}/{max_paths} "
                f"| depth={current_node.depth}"
            )

            if len(found_paths) >= max_paths:
                break

            continue

        # -------------------------
        # 规则应用
        # -------------------------
        for rule in rules:

            rule_name = type(rule).__name__
            rule_stats[rule_name]["calls"] += 1

            try:

                generated_by_rule = 0

                for next_state, action_desc, h_cost in rule.apply(state):

                    if next_state is None:
                        rule_stats[rule_name]["none_state"] += 1
                        logger.debug(f"{rule_name} 生成 None state")
                        continue

                    total_h_cost = current_node.cumulative_h_cost + h_cost

                    if total_h_cost > n_hydrogen:
                        pruned_by_h += 1
                        rule_stats[rule_name]["pruned_h"] += 1
                        continue

                    generated_states += 1
                    generated_by_rule += 1

                    child_node = SearchNode(
                        state=next_state,
                        parent=current_node,
                        action=action_desc,
                        step_h_cost=h_cost
                    )
                    logger.info(f"{rule_name}  {child_node}")

                    open_queue.append(child_node)

                rule_stats[rule_name]["generated"] += generated_by_rule

                # 如果 rule 完全没产生状态
                if generated_by_rule == 0:
                    logger.debug(f"{rule_name} 未产生新状态")

            except Exception as e:

                rule_stats[rule_name]["exceptions"] += 1

                logger.error(
                    f"Rule {rule_name} 执行异常: {e}",
                    exc_info=True
                )

    # -------------------------
    # 搜索结束
    # -------------------------

    elapsed = time.time() - start_time

    logger.info(
        f"\n搜索结束\n"
        f"迭代:{iteration_count}\n"
        f"生成状态:{generated_states}\n"
        f"剪枝(H):{pruned_by_h}\n"
        f"耗时:{elapsed:.2f}s"
    )

    # -------------------------
    # Rule统计报告
    # -------------------------

    logger.info("Rule 统计:")

    for rule, stat in rule_stats.items():

        logger.info(
            f"{rule:20s} | "
            f"calls={stat['calls']} | "
            f"gen={stat['generated']} | "
            f"none={stat['none_state']} | "
            f"pruneH={stat['pruned_h']} | "
            f"err={stat['exceptions']}"
        )

    return found_paths

