
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
sleep_v2.py — 睡眠与梦 (出厂能力 P3-3, 扩展 sleep_consolidate)
梦: 记忆随机重组 → "昨晚做了个梦"
睡眠质量 → 情绪 (没睡好 → 脾气差)
"""
import json
import random
import time
from datetime import datetime
from pathlib import Path

STATE_FILE = BASE / "state" / "sleep.json"

DEFAULTS = {
    "last_sleep": None,        # 上次睡觉日期
    "dream": None,             # 昨晚的梦
    "quality": "好",           # 睡眠质量: 好/一般/差
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

def dream_from_memories(memory_bank):
    """睡眠时: 记忆随机重组 → 梦 (每晚一次, 由夜间调度调用)"""
    s = load()
    today = datetime.now().strftime("%Y-%m-%d")
    if s["last_sleep"] == today:
        return None
    s["last_sleep"] = today
    fragments = []
    try:
        r1 = memory_bank.query(query_text="最近发生的事", top_k=3, k_hops=1)
        r2 = memory_bank.query(query_text="小时候的事", top_k=2, k_hops=1)
        for e, _ in list(r1["contents"]) + list(r2["contents"]):
            t = (e.text or "").strip()[:20]
            if t:
                fragments.append(t)
    except Exception:
        pass
    if len(fragments) >= 2:
        a, b = random.sample(fragments, 2)
        s["dream"] = f"梦见了{a}，还掺着{b}的事，乱糟糟的"
    else:
        s["dream"] = "做了个乱七八糟的梦，醒来就忘了大半"
    # 睡眠质量 (随机 + 情绪残留)
    s["quality"] = random.choice(["好", "好", "好", "一般", "差"])
    save(s)
    return s["dream"]

def sleep_line():
    """注入: 睡眠状态 (早上/问昨晚时)"""
    s = load()
    if s["dream"]:
        q = {"好": "睡得挺香", "一般": "睡得一般般", "差": "昨晚没睡好，有点起床气"}[s["quality"]]
        return f"（昨晚{q}。{s['dream']}。）"
    return ""

if __name__ == "__main__":
    s = load()
    s["dream"] = "梦见了食堂的番茄炒蛋，还掺着数学课的事"
    s["quality"] = "一般"
    save(s)
    print(sleep_line())
