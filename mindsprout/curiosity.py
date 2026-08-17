
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
curiosity.py — 好奇心 (出厂能力 P0-4)
对未知主动发问 — 她不是只回应的机器
"""
import json
import random
from pathlib import Path

STATE_FILE = BASE / "state" / "curiosity.json"

DEFAULTS = {
    "open_questions": [      # 她心里悬而未决的问题
        "那个人到底是谁",
        "他为什么总来找我说话",
    ],
    "asked_recently": {},    # {question: timestamp} 防止重复问
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

def add_question(q: str):
    """遇到未知 → 心里记下一个问题"""
    s = load()
    if q not in s["open_questions"]:
        s["open_questions"].append(q)
        save(s)

def maybe_ask(curiosity_level: float, cooldown_h: float = 2.0) -> str:
    """好奇度高时, 主动问出一个问题 (冷却防烦)"""
    import time
    s = load()
    if curiosity_level < 70:
        return ""
    now = time.time()
    qs = [q for q in s["open_questions"]
          if now - s["asked_recently"].get(q, 0) > cooldown_h * 3600]
    if not qs:
        return ""
    q = random.choice(qs)
    s["asked_recently"][q] = now
    save(s)
    return q

def resolve_question(q: str):
    """问题得到回答 → 从悬而未决移除"""
    s = load()
    if q in s["open_questions"]:
        s["open_questions"].remove(q)
        save(s)

def curiosity_line(curiosity_level: float) -> str:
    if curiosity_level >= 80:
        s = load()
        if s["open_questions"]:
            return f"（你心里一直惦记着：{s['open_questions'][0]}。总想找机会问清楚。）"
    return ""

if __name__ == "__main__":
    print("主动问:", maybe_ask(85))
    add_question("数学老师为什么总点我名")
    print("问题清单:", load()["open_questions"])
    print("注入:", curiosity_line(85))
