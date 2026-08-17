
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
body.py — 身体内感受 (出厂能力 P0-3)
心跳/胃/手汗/体温/肌肉 — 与情绪+驱力映射
"心里一紧"从身体自然出, 不是文本套话
"""
import json
import time
import random
from pathlib import Path

STATE_FILE = BASE / "state" / "body.json"

DEFAULTS = {
    "heart": 70,      # 心率 60-140
    "stomach": 40,    # 胃感 0=空 100=饱
    "sweat": 20,      # 手汗 0-100
    "temp": 36.5,     # 体温感 (主观: 冷/热/正常)
    "muscle": 20,     # 肌肉疲惫 0-100
    "shiver": 0,      # 起鸡皮疙瘩
}

# 情绪 → 身体映射 (James-Lange 方向: 情绪伴随身体)
EMOTION_BODY = {
    "joy":      {"heart": +15, "stomach": -10, "sweat": +5,  "feels": "心口暖洋洋的，嘴角自己就往上翘"},
    "sad":      {"heart": -10, "stomach": +15, "sweat": 0,   "feels": "胸口闷闷的，喉咙有点发紧"},
    "anger":    {"heart": +30, "stomach": +10, "sweat": +25, "feels": "一股气堵在胸口，手有点抖"},
    "anxiety":  {"heart": +25, "stomach": +20, "sweat": +35, "feels": "心怦怦跳，手心出汗，胃缩成一团"},
    "calm":     {"heart": -10, "stomach": -5,  "sweat": -10, "feels": "呼吸又轻又慢，整个人松下来"},
    "sulky":    {"heart": +10, "stomach": +10, "sweat": 0,   "feels": "心里堵着一团气，鼓鼓的"},
    "warmth":   {"heart": +8,  "stomach": -15, "sweat": 0,   "feels": "心里暖烘烘的，像被晒过的被子裹着"},
    "curiosity": {"heart": +5, "stomach": 0,   "sweat": +5,  "feels": "心里痒痒的，坐不住"},
    "fear":     {"heart": +35, "stomach": +15, "sweat": +40, "feels": "心快跳出来了，腿有点软"},
    "neutral":  {"heart": 0,   "stomach": 0,   "sweat": 0,   "feels": "身体没什么特别的感觉"},
}

# 驱力 → 身体
DRIVE_BODY = {
    "hunger": {"stomach": -25, "feels": "胃空空的，叫了两声"},
    "thirst": {"feels": "嗓子发干"},
    "sleep":  {"muscle": +15, "feels": "眼皮沉得抬不起来，胳膊软绵绵的"},
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

def apply_emotion(emotion_dom: str):
    """情绪 → 身体 (每次情绪变化调用)"""
    s = load()
    eb = EMOTION_BODY.get(emotion_dom, EMOTION_BODY["neutral"])
    for k in ("heart", "stomach", "sweat"):
        if k in eb:
            s[k] = max(10, min(150, s[k] + eb[k]))
    s["feels"] = eb["feels"]
    # 身体回落到基线
    s["heart"] = max(60, s["heart"] - 2)
    save(s)
    return s

def apply_drive(drive: str, level: float):
    """驱力 → 身体"""
    if level < 65:
        return load()
    s = load()
    db = DRIVE_BODY.get(drive)
    if db:
        if "stomach" in db and drive == "hunger":
            s["stomach"] = max(0, s["stomach"] + db["stomach"])
        if "muscle" in db:
            s["muscle"] = min(100, s["muscle"] + db["muscle"])
    save(s)
    return s

def body_line(emotion_dom="neutral"):
    """注入: 身体流 (只在有感觉时注)"""
    s = load()
    parts = []
    if s["heart"] >= 95:
        parts.append("心怦怦跳")
    if s["sweat"] >= 50:
        parts.append("手心有点出汗")
    if s["stomach"] <= 15:
        parts.append("胃空空的")
    if s["muscle"] >= 60:
        parts.append("身上有点乏")
    if s.get("feels") and s["feels"] not in ("身体没什么特别的感觉",):
        parts.append(s["feels"])
    if parts:
        return "（你的身体：" + "；".join(parts[:3]) + "。）"
    return ""

if __name__ == "__main__":
    for e in ("joy", "anxiety", "anger", "sad"):
        s = apply_emotion(e)
        print(e, "→ 心率", s["heart"], "手汗", s["sweat"], "|", body_line(e)[:50])
