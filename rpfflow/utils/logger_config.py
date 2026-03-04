"""
统一日志配置模块

使用方法：
from logger_config import setup_logger
setup_logger(log_file="train.log", level="INFO")
"""

import logging
import sys


def setup_logger(log_file=None, level="INFO"):
    """
    初始化全局日志系统

    Parameters
    ----------
    log_file : str or None
        日志文件路径（None 表示不写入文件）
    level : str
        日志等级: DEBUG / INFO / WARNING / ERROR
    """

    # 字符串转 logging 常量
    level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有 handler（防止重复打印）
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ----------------------
    # 控制台输出
    # ----------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    # 🔥 关键
    try:
        console_handler.stream.reconfigure(encoding='utf-8')
    except Exception:
        pass
    root_logger.addHandler(console_handler)

    # ----------------------
    # 文件输出（可选）
    # ----------------------
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 防止 logging 影响 tqdm
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    root_logger.info("Logger initialized successfully.")

