"""
humanize-ai: AI文本去AI化引擎

黑箱函数：AI文本进 → 人类文本出
本地运行，CPU/GPU自动检测，零成本，零配置

安装: pip install humanize-ai
使用: from humanize_ai import humanize
      result = humanize("AI写的文本...")
"""

from mindsprout.config import BASE

# humanize_ai/__init__.py
from .engine import Humanizer, humanize, stream_humanize, get_model_info

__version__ = "0.1.0"
__all__ = ["Humanizer", "humanize", "stream_humanize", "get_model_info"]
