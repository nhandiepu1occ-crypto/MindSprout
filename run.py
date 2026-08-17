# -*- coding: utf-8 -*-
"""
MindSprout - 启动入口
用法:
  python run.py                 # 启动平台 (默认 Qwen2.5-3B)
  MINSPROUT_MODEL=... python run.py   # 指定模型路径
环境:
  MINSPROUT_HOME  数据目录 (默认 ./data)
  MINSPROUT_MODEL 基座模型路径 (默认 Qwen2.5-3B-Instruct)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "platform"))

def main():
    model = os.environ.get("MINSPROUT_MODEL", "")
    if not model:
        print("⚠️ 未设置 MINSPROUT_MODEL，请指定 Qwen2.5 模型路径")
        print("   示例: MINSPROUT_MODEL=/models/qwen2.5-3b-instruct python run.py")
        print("   或设置环境变量后重试。")
        return 1
    os.environ.setdefault("MINSPROUT_HOME", str(ROOT / "data"))
    # 平台 app 内嵌模型路径从环境变量读取
    os.environ["LUOLUO_MODEL"] = model
    sys.argv = ["app.py"]
    import app
    app.main()
    return 0

if __name__ == "__main__":
    sys.exit(main())
