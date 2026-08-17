
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
attention.py — 注意机制 (出厂能力 P2-1)
注意焦点(当前人物/场景/话题) → 检索偏置 + 注意转移(新刺激捕获)
"""
import json
from pathlib import Path

STATE_FILE = BASE / "state" / "attention.json"

DEFAULTS = {
    "person_id": None,       # 当前注意人物
    "scene": None,           # 当前场景
    "topic_words": [],       # 当前话题焦点词 (最近3轮累积)
    "last_shift": 0.0,       # 上次注意转移时间
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

def focus(person_id=None, scene=None, topic_words=None):
    """更新注意焦点 (对话开始时调用; person_id=None 清除焦点)"""
    s = load()
    s["person_id"] = person_id  # None 时清除 (楚门 v2.0: 防上一轮残留)
    if scene:
        s["scene"] = scene
    if topic_words:
        # 话题词累积 (最多5个, 新词在前)
        merged = list(topic_words) + [w for w in s["topic_words"] if w not in topic_words]
        s["topic_words"] = merged[:5]
    if person_id is None and topic_words is None:
        s["topic_words"] = []
    save(s)
    return s

def shift(new_person_id=None, new_scene=None):
    """注意转移 (人物/场景变了)"""
    s = load()
    old = (s["person_id"], s["scene"])
    if new_person_id and new_person_id != s["person_id"]:
        s["person_id"] = new_person_id
        s["topic_words"] = []   # 换人清话题
    if new_scene and new_scene != s["scene"]:
        s["scene"] = new_scene
    import time
    s["last_shift"] = time.time()
    save(s)
    return old, (s["person_id"], s["scene"])

def boost_query(query_text: str) -> str:
    """检索时: 注意焦点叠加到查询 (焦点相关记忆更容易浮现)"""
    s = load()
    if s["person_id"]:
        try:
            from humanize_ai.persona import PERSONS
            p = PERSONS.get(s["person_id"])
            if p:
                return query_text + " " + " ".join(p.get("topic_words", [])[:3])
        except Exception:
            pass
    return query_text

def attention_line():
    """注入: 注意焦点 (不强注 — 只在换人/换场景时给个轻微提示)"""
    s = load()
    parts = []
    if s["person_id"]:
        try:
            from humanize_ai.persona import PERSONS
            p = PERSONS.get(s["person_id"])
            if p:
                parts.append(f"你正跟{p['name']}说话")
        except Exception:
            pass
    if s["scene"]:
        from humanize_ai.senses import SCENES
        sc = SCENES.get(s["scene"])
        if sc:
            parts.append(f"你在{sc['name']}")
    if parts:
        return "（" + "，".join(parts) + "。）"
    return ""

if __name__ == "__main__":
    focus(person_id="mom", scene="home", topic_words=["吃饭", "作业"])
    print(attention_line())
    print("检索boost:", boost_query("你今天怎么样"))
    old, new = shift(new_person_id="him")
    print("注意转移:", old, "→", new)
