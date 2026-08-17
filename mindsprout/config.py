
# -*- coding: utf-8 -*-
"""MindSprout 配置: 数据目录可用环境变量 MINSPROUT_HOME 覆盖"""
import os
from pathlib import Path

BASE = Path(os.environ.get("MINSPROUT_HOME", str(Path(__file__).resolve().parents[1] / "data")))
BASE.mkdir(parents=True, exist_ok=True)

STATE_DIR = BASE / "state"
MEMORY_DIR = BASE / "memory"
WORLD_DIR = BASE / "world"
PLATFORM_DIR = Path(__file__).resolve().parents[1] / "platform"
