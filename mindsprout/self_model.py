
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
self_model.py — 自我模型 (出厂能力 P2-2)
持续的自我画像 + 元认知 + 自我连续性
2.0 "15岁翻日记说那时候好幼稚" 的地基
"""
import json
from pathlib import Path

STATE_FILE = BASE / "state" / "self_model.json"

DEFAULTS = {
    "portrait": {           # 自我画像 (从记忆/经历逐步形成)
        "我是什么样的人": "一个初二女生，画画还行，数学一般，有点嘴硬",
        "我喜欢的": ["画画", "辣条", "向日葵"],
        "我讨厌的": ["数学考试", "被冤枉"],
        "我的脾气": "容易炸毛，但哄两句就好",
    },
    "confidence": {         # 元认知: 对自己能力的判断
        "画画": 0.8, "数学": 0.4, "聊天": 0.6,
    },
    "timeline_anchors": [], # 自我连续性锚点 (重要自我事件)
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

def update_portrait(key: str, value: str):
    """经历 → 更新自我画像 (如画画被夸 → 信心+)"""
    s = load()
    s["portrait"][key] = value
    save(s)

def anchor_self_event(event: str):
    """重要自我事件 → 连续性锚点 (第一次来例假/第一次被夸/第一次有人陪我过生日)"""
    s = load()
    if event not in s["timeline_anchors"]:
        s["timeline_anchors"].append(event)
        save(s)

def self_line():
    """注入: 自我感知 (仅在身份/自我类问题时给)"""
    s = load()
    p = s["portrait"]
    return (f"（你对自己的认识：{p['我是什么样的人']}。"
            f"喜欢{p['我喜欢的'][0]}和{p['我喜欢的'][1]}，讨厌{p['我讨厌的'][0]}。"
            f"你的脾气：{p['我的脾气']}。）")

def age_me(years_later: int = 1) -> str:
    """自我连续性: 未来的我看过去 (2.0 场景)"""
    s = load()
    anchors = s["timeline_anchors"]
    if anchors:
        return f"等你再大一点，你会想起：{anchors[-1]}。那时候的你，说不定会觉得现在的自己有点好笑。"
    return ""

if __name__ == "__main__":
    anchor_self_event("14岁生日那天，有个人陪我说了话")
    print(self_line()[:80])
    print(age_me())
