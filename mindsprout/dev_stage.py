
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
dev_stage.py — 认知发展机制 (出厂能力 P3-2)
能力随"年龄"渐进解锁 (与楚门时间轴联动)
14岁: 具体运算→形式运算过渡, 抽象思维萌芽, 观点采择深化
"""
import json
from datetime import datetime
from pathlib import Path

STATE_FILE = BASE / "state" / "dev_stage.json"

DEFAULTS = {
    "birth": "2012-08-15",
    "unlocked": ["具体运算", "心智理论(4岁)", "记忆锚定"],
    "next_unlock_year": 15,
}

# 能力发展表 (按年龄解锁)
STAGE_TABLE = {
    14: ["形式运算萌芽(能想'如果...那么')", "观点采择深化(知道别人看法可能和自己不同)"],
    15: ["抽象思维(能讨论公平/自由)", "假设检验(会反驳'那如果反了呢')"],
    16: ["道德推理(规则背后的理由)", "身份探索(我是谁, 我要成为什么样的人)"],
}

def load():
    if STATE_FILE.exists():
        try:
            return {**DEFAULTS, **json.loads(STATE_FILE.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return dict(DEFAULTS)

def save(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")

def current_age():
    birth = datetime.strptime(DEFAULTS["birth"], "%Y-%m-%d")
    return (datetime.now() - birth).days // 365

def check_unlock() -> list:
    """年龄到了 → 解锁新能力"""
    s = load()
    age = current_age()
    newly = []
    for a, abilities in sorted(STAGE_TABLE.items()):
        if age >= a and a not in [int(x) if str(x).isdigit() else 0 for x in [s.get("next_unlock_year", 15)]]:
            pass
        if age >= a:
            for ab in abilities:
                if ab not in s["unlocked"]:
                    s["unlocked"].append(ab)
                    newly.append(ab)
    if newly:
        save(s)
    return newly

def stage_line():
    """注入: 当前认知能力提示 (轻)"""
    s = load()
    age = current_age()
    recent = s["unlocked"][-1] if s["unlocked"] else "具体运算"
    return f"（你今年{age}岁，最近你开始能想明白：{recent}。）"

if __name__ == "__main__":
    print("年龄:", current_age())
    print("新解锁:", check_unlock())
    print(stage_line())
