import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Iterable

# 假设这些类已经定义好
from rpfflow.core.action import HydrogenationAction, DissociationAction, CouplingAction, AssociationAction
from rpfflow.core.state import RxnState, SearchNode
from rpfflow.rules.matchs import is_duplicate
from rpfflow.core.cutter import transition_probability

logger = logging.getLogger(__name__)


@dataclass
class SearchStats:
    """负责存储和报告搜索统计数据"""
    iteration: int = 0
    generated: int = 0
    pruned_h: int = 0
    start_time: float = field(default_factory=time.time)
    rule_metrics: dict = field(default_factory=dict)

    def log_progress(self, queue_size: int, depth: int):
        elapsed = time.time() - self.start_time
        logger.info(
            f"迭代:{self.iteration} | 队列:{queue_size} | 深度:{depth} | "
            f"生成:{self.generated} | 剪枝(H):{self.pruned_h} | 耗时:{elapsed:.2f}s\n"
        )

    def report_final(self):
        elapsed = time.time() - self.start_time
        logger.info(f"\n搜索结束 - 总耗时:{elapsed:.2f}s\n{self}")
        for name, m in self.rule_metrics.items():
            logger.info(f"{name:20s} | calls={m['calls']} | gen={m['gen']} | err={m['err']}")


def bfs_search(initial_state: RxnState, target_graph, n_hydrogen=8, rules=None, max_paths=10, max_depth=5):
    # 1. 初始化配置
    rules = rules or [HydrogenationAction(), DissociationAction(), CouplingAction(), AssociationAction()]
    stats = SearchStats()
    stats.rule_metrics = {type(r).__name__: {"calls": 0, "gen": 0, "err": 0} for r in rules}

    found_paths = []
    queue = deque([SearchNode(state=initial_state)])

    logger.info("BFS 搜索开始")
    visited_set = set()
    _MAX_VISITED = 20

    # 2. 主循环
    while queue:
        current_node = queue.popleft()
        stats.iteration += 1

        if stats.iteration % 1 == 0:  # 降低日志频率提高性能
            stats.log_progress(len(queue), current_node.depth)
            logger.info(f"当前状态： {current_node}")

        # 目标检测
        if is_duplicate(target_graph, current_node.state.graphs):
            found_paths.append(current_node)
            logger.info(f"找到路径 {len(found_paths)}/{max_paths} | depth={current_node.depth}")
            continue

        if len(found_paths) >= max_paths:
            logger.info(f"已经发现设定路径 {len(found_paths)}/{max_paths} ，搜索结束")
            break

        # 剪枝逻辑， 重复路径剪枝
        if current_node.path_signature_tuple in visited_set:
            continue
        else:
            visited_set.add(current_node.path_signature_tuple)

        if len(visited_set) > _MAX_VISITED:
            visited_set.clear()
            continue


        # 利用深度进行剪枝
        if current_node.depth > max_depth:
            continue

        # 剪枝逻辑，对于多碳耦合反应
        if current_node.cumulative_h_cost < n_hydrogen - 1:
            indices = set(current_node.state.element_indices({"C", "N"}))
            f_indices = set(current_node.state.element_indices({"F"}))
            # 取交集：只有既含 C/N 又含 F 的片段才能参与耦合
            available_indices = list(indices.intersection(f_indices))
            # 需谨慎
            if len(available_indices) != len(indices):  # C/F 已不在一起
                continue


        # 扩展节点
        expand_node = []
        for rule in rules:
            rule_name = type(rule).__name__
            stats.rule_metrics[rule_name]["calls"] += 1

            try:
                for next_state, action_desc, h_cost in rule.apply(current_node.state):
                    if next_state is None: continue
                    expand_node.append((next_state, action_desc, h_cost))
                    stats.generated += 1
                    stats.rule_metrics[rule_name]["gen"] += 1

            except Exception as e:
                stats.rule_metrics[rule_name]["err"] += 1
                logger.error(f"Rule {rule_name} 异常: {e}", exc_info=True)

            # 1. 预选池：存放所有通过基础剪枝的候选子节点
            candidates = []

            for i, (next_state, action_desc, h_cost) in enumerate(expand_node):
                # 基础剪枝：氢平衡
                total_h_cost = current_node.cumulative_h_cost + h_cost
                if total_h_cost > n_hydrogen:
                    stats.pruned_h += 1
                    continue

                # 创建临时节点对象（先不进队列）
                child_node = SearchNode(
                        state=next_state, parent=current_node,
                        action=action_desc, step_h_cost=h_cost
                    )
                # 计算转移概率 delta_G = E_next - E_current
                # 注意：这里是用当前步的增量来评估“这一步”的优劣
                delta_g = child_node.adsorption_energy
                prob = transition_probability(delta_g)


                # 将概率存入节点或元组中，方便排序
                candidates.append((prob, child_node))
            use_beam_search, beam_width = True, 1
            # 2. 排序与筛选逻辑
            if use_beam_search and beam_width is not None:
                # 按概率从大到小排序
                candidates.sort(key=lambda x: x[0], reverse=True)
                # 只取前 N 个
                targets = candidates[:beam_width]
            else:
                # 开关关闭，全量放入
                targets = candidates

            # 3. 正式放进队列
            for prob, node in targets:
                queue.append(node)


    stats.report_final()
    return found_paths