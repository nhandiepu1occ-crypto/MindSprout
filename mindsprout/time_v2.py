
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
time_v2.py — 主观时间 (出厂能力 P2-3)
无聊时时间慢 / 开心时时间快 / 期待未来
在 timeline 现实时钟之上叠加主观体验层
"""
import json
import time
from pathlib import Path

STATE_FILE = BASE / "state" / "time_v2.json"

DEFAULTS = {
    "last_interact": 0.0,     # 上次有人说话
    "subjective_speed": 1.0,  # 主观时间速度 (无聊 0.4慢 / 开心 1.5快)
    "anticipations": [],      # 期待的事 ["明天要交美术作业"]
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

def update(emotion_dom="neutral", is_bored=False):
    """主观时间速度: 与情绪/无聊耦合"""
    s = load()
    if emotion_dom == "joy":
        s["subjective_speed"] = 1.5
    elif emotion_dom in ("sad", "anxiety"):
        s["subjective_speed"] = 0.5
    elif is_bored:
        s["subjective_speed"] = 0.4
    else:
        s["subjective_speed"] = max(0.6, min(1.2, s["subjective_speed"] * 0.9 + 0.1))
    save(s)
    return s

def note_interaction():
    s = load()
    s["last_interact"] = time.time()
    save(s)

def alone_minutes() -> float:
    s = load()
    if not s["last_interact"]:
        return 0
    return (time.time() - s["last_interact"]) / 60

def anticipate(thing: str):
    s = load()
    if thing not in s["anticipations"]:
        s["anticipations"].append(thing)
        save(s)

def time_line():
    """注入: 主观时间感"""
    s = load()
    parts = []
    alone = alone_minutes()
    if alone > 120:  # 独自超过2小时
        parts.append("觉得时间过得好慢，有点无聊")
    if s["subjective_speed"] >= 1.4:
        parts.append("时间过得好快，还没玩够")
    if s["anticipations"]:
        parts.append(f"心里惦记着：{s['anticipations'][0]}")
    if parts:
        return "（你此刻：" + "；".join(parts) + "。）"
    return ""

if __name__ == "__main__":
    update("joy")
    anticipate("明天美术课要交画")
    print(time_line())
