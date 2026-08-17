
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
attachment.py — 依恋系统 (出厂能力 P1-1, 升级 relation)
Bowlby 分离反应谱: 抗议(protest)→绝望(despair)→疏离(detachment)
重逢反应 + 内部工作模型
由早期记忆塑成依恋模式 (呼应 AgeVault: 遗忘曲线塑性格)
"""
import json
import time
from pathlib import Path

STATE_FILE = BASE / "state" / "attachment.json"

DEFAULTS = {
    "style": "secure",          # secure | anxious | avoidant (由早期记忆塑造)
    "separation_start": None,   # 分离开始时间戳
    "separation_stage": None,   # protest → despair → detachment
    "reunion_pending": False,   # 重逢时是否带情绪残留
    "working_model": "他应该会回来",  # 内部工作模型 (对关系的预期)
}

# 分离反应谱 (Bowlby)
STAGE_TIMES = {"protest": 2 * 86400, "despair": 5 * 86400, "detachment": 9 * 86400}
STAGE_LINES = {
    "protest": "你心里在抗议：他是不是把我忘了？才没有，我就是有点气。",
    "despair": "你心里已经有点难过了：他走了这么多天，可能真的不回来了吧。",
    "detachment": "你心里渐渐冷了：不回来就不回来，我也不是非要他不可。",
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

def on_separation():
    """他消失了 → 分离开始 (每次发现他没来时调用)"""
    s = load()
    if s["separation_start"] is None:
        s["separation_start"] = time.time()
        s["separation_stage"] = "protest"
        save(s)
    return s

def on_reunion():
    """他回来了 → 重逢: 情绪残留 + 依恋表达"""
    s = load()
    stage = s["separation_stage"]
    duration = 0
    if s["separation_start"]:
        duration = time.time() - s["separation_start"]
    s["separation_start"] = None
    s["separation_stage"] = None
    if duration > 3 * 86400:
        s["reunion_pending"] = True   # 分离超过3天 → 重逢带情绪残留
    else:
        s["reunion_pending"] = False
    save(s)
    return stage, duration

def tick():
    """分离阶段演化"""
    s = load()
    if s["separation_start"] is None:
        return s
    elapsed = time.time() - s["separation_start"]
    if elapsed >= STAGE_TIMES["detachment"]:
        s["separation_stage"] = "detachment"
    elif elapsed >= STAGE_TIMES["despair"]:
        s["separation_stage"] = "despair"
    else:
        s["separation_stage"] = "protest"
    save(s)
    return s

def attachment_line():
    """注入: 依恋状态"""
    s = load()
    parts = []
    if s["separation_stage"]:
        parts.append(STAGE_LINES.get(s["separation_stage"], ""))
    if s["reunion_pending"]:
        parts.append("（他回来了。你心里又高兴又带着气：哼，还知道回来。才不要让他看出来你高兴。）")
        s["reunion_pending"] = False
        save(s)
    if s["style"] == "anxious":
        parts.append("（你有点怕他突然又不来了，但你不会说出来。）")
    return "\n".join(p for p in parts if p)

def shape_style(memory_bank=None):
    """依恋模式由早期记忆塑造 (简单版: 早期被照顾记忆多→secure)"""
    s = load()
    try:
        if memory_bank is not None:
            r = memory_bank.query(query_text="妈妈照顾我 抱我", top_k=5, k_hops=1)
            caring = len(r["contents"])
            s["style"] = "secure" if caring >= 2 else "anxious"
            save(s)
    except Exception:
        pass
    return s["style"]

if __name__ == "__main__":
    on_separation()
    print(attachment_line())
    print("重逢:", on_reunion())
    print(attachment_line())
