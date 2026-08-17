
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
social.py — 社会世界 (出厂能力 P3-1)
班级社会结构 + 面子/羞耻 + 群体认同
她不是孤岛, 有社会位置
"""
import json
from pathlib import Path

STATE_FILE = BASE / "state" / "social.json"

DEFAULTS = {
    "class_pos": "普通学生, 成绩中游, 美术课代表",  # 她在班里的位置
    "group": "和小雨、小美一个圈子",               # 群体归属
    "face_events": [],     # 丢面子事件记录
    "rival": "隔壁班那个说我画丑的女生",           # 对头
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

def lose_face(event: str):
    """丢面子 → 脸红/想钻地缝 (下次相关场景会被想起)"""
    s = load()
    if event not in s["face_events"]:
        s["face_events"].append(event)
        save(s)
    return f"（你想起那件事还是脸红：{event}。真想找个地缝钻进去。）"

def social_line():
    s = load()
    return (f"（你在班里：{s['class_pos']}。你常和{s['group']}一起。"
            f"你最烦{s['rival']}。）")

if __name__ == "__main__":
    print(lose_face("上次全班面前读作文读错了字"))
    print(social_line())
