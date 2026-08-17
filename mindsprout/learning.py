
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
learning.py — 在线学习系统 (出厂能力 P0-2)
①在线事实学习: 对话新事实 → 记忆库
②条件联结: 人物-感官-情绪联结随共现强化
③习惯化: 重复问题反应递减
④强化: 被回应 → 表达权重+
"""
import json
import time
from pathlib import Path

STATE_FILE = BASE / "state" / "learning.json"

DEFAULTS = {
    "conditioning": {},    # {key: strength} 条件联结 (如 "him:green": 0.7)
    "habituation": {},     # {question_text: {times, last}} 习惯化
    "reinforced": {},      # {expression_pattern: weight} 强化
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

def learn_fact(memory_bank, text: str, exp_id=None):
    """对话中的新事实 → 记忆库 (auto_memory 的升级: 结构化学习)"""
    # 简单启发式: 含"我是/我叫/我喜欢/我不喜欢/我家/我在"的陈述 → store
    import re
    if not re.search(r"(我是|我叫|我喜欢|我不喜欢|我家|我在|我是你)", text or ""):
        return None
    import uuid
    eid = exp_id or ("exp_" + uuid.uuid4().hex[:12])
    try:
        memory_bank.store(
            exp_id=eid, text=text[:100], source_url="learned",
            source_year=2026, scene_graph={"time": "life"},
            emotion_vector={"valence": 0.1, "arousal": 0.2, "dominant": "neutral"},
        )
        memory_bank.save(str(BASE / "memory"))
        return eid
    except Exception:
        return None

def condition(person_id: str, sense_key: str, amount=0.1):
    """条件联结: 他出现+绿色安心 → 联结加强 (反复共现 → 自动联想)"""
    s = load()
    key = f"{person_id}:{sense_key}"
    s["conditioning"][key] = min(1.0, s["conditioning"].get(key, 0.2) + amount)
    save(s)
    return s["conditioning"][key]

def habituate(question_text: str) -> float:
    """习惯化: 同一问题反复问 → 新鲜感递减 (回答变简洁/不耐烦)"""
    s = load()
    q = (question_text or "")[:30]
    h = s["habituation"].setdefault(q, {"times": 0, "last": 0})
    h["times"] += 1
    h["last"] = time.time()
    save(s)
    # 新鲜度: 第1次 1.0, 第3次 0.6, 第6次 0.3
    return max(0.2, 1.0 - 0.13 * (h["times"] - 1))

def reinforce(expression_key: str, amount=0.05):
    """强化: 表达被回应 → 权重+ (她学会什么表达有效)"""
    s = load()
    s["reinforced"][expression_key] = min(1.0, s["reinforced"].get(expression_key, 0.1) + amount)
    save(s)
    return s["reinforced"][expression_key]

def freshness(question_text: str) -> float:
    s = load()
    h = s["habituation"].get((question_text or "")[:30])
    if not h:
        return 1.0
    return max(0.2, 1.0 - 0.13 * (h["times"] - 1))

if __name__ == "__main__":
    for i in range(4):
        print("第", i + 1, "次问同样问题 → 新鲜度", round(habituate("你作业写完了吗"), 2))
    print("条件联结:", condition("him", "green"))
    print("强化:", reinforce("嘴硬"))
