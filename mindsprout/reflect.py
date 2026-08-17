
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
reflect.py — 情绪反思 (1.8 第四块): 她知道为什么烦
- 负面情绪时, 从近期记忆检索原因 → 归因自述 (她的话)
- 耦合: emotion_state(触发) / memory_bank(近期记忆) / role_registry(可能涉及谁) / drives(生理影响情绪)
- 注入: 情绪不好时一句"（你心里知道为什么…）"
"""
import json
import time
import re
from pathlib import Path

STATE_FILE = BASE / "state" / "reflect.json"

DEFAULTS = {
    "last_reflect": 0.0,   # 上次反思时间
    "reflection": "",      # 归因文本
    "emotion": "",         # 当时的情绪
    "count_day": "",       # 每日计数
    "count": 0,
}

NEGATIVE = ("难过", "伤心", "委屈", "生气", "愤怒", "赌气", "冷淡", "烦", "焦虑", "害怕", "哭")


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


def is_negative(emotion_text):
    return any(w in emotion_text for w in NEGATIVE)


def should_reflect(emotion_text):
    """负面情绪 + 每天≤2次 + 冷却3h"""
    s = load()
    if not is_negative(emotion_text):
        return False
    today = time.strftime("%Y-%m-%d")
    if s.get("count_day") != today:
        s["count_day"] = today
        s["count"] = 0
        save(s)
    if s.get("count", 0) >= 2:
        return False
    if time.time() - s.get("last_reflect", 0) < 3600 * 3:
        return False
    return True


def _recent_events(memory_bank, top=6):
    """近期记忆: 今天/昨天的生活 (楚门生活记忆优先)"""
    items = []
    try:
        r = memory_bank.query(query_text="今天发生的事", top_k=top, k_hops=1)
        for e, _ in r["contents"][:top]:
            t = (e.text or "").strip()
            if t:
                items.append(t[:60])
    except Exception:
        pass
    if len(items) < 3:
        try:
            r2 = memory_bank.query(query_text="最近的事", top_k=top, k_hops=1)
            for e, _ in r2["contents"][:top]:
                t = (e.text or "").strip()
                if t and t not in items:
                    items.append(t[:60])
        except Exception:
            pass
    return items


def make_reflection(memory_bank, generate, emotion_text):
    """归因: 近期记忆 + 当前情绪 → 她的话"""
    s = load()
    events = _recent_events(memory_bank)
    # 生理因素 (困/饿会放大负面情绪)
    phys = []
    try:
        from humanize_ai.drives import load as _dl
        d = _dl()
        if d.get("sleep", 0) >= 70:
            phys.append("有点困")
        if d.get("hunger", 0) >= 75:
            phys.append("肚子饿")
    except Exception:
        pass
    ctx = "\n".join(f"- {t}" for t in events[:4]) or "- （今天没什么特别的）"
    phys_line = "；".join(phys) if phys else ""
    prompt = (f"你现在的情绪：{emotion_text}。{('身体上' + phys_line + '。') if phys_line else ''}\n"
              f"最近经历了这些：\n{ctx}\n\n"
              f"你为什么会是这种情绪？用你的话说说原因（第一人称，40-70字，"
              f"像自己跟自己嘀咕，别说'因为所以'这种作文话，别用括号）。")
    try:
        text = generate(prompt, max_tokens=110, temperature=0.9).strip()
        if len(text) < 15:
            raise ValueError("too short")
    except Exception:
        text = ""
    if not text:
        return s
    today = time.strftime("%Y-%m-%d")
    s["reflection"] = text
    s["emotion"] = emotion_text
    s["last_reflect"] = time.time()
    if s.get("count_day") != today:
        s["count_day"] = today
        s["count"] = 0
    s["count"] = s["count"] + 1
    save(s)
    return s


def reflect_line():
    """注入: 情绪不好时一句归因 (短)"""
    s = load()
    if s.get("reflection") and is_negative(s.get("emotion", "")):
        return f"（你心里知道今天为什么不太对劲：{s['reflection'][:45]}）"
    return ""


def reflect_data():
    s = load()
    return {"reflection": s.get("reflection", ""), "emotion": s.get("emotion", ""),
            "ts": s.get("last_reflect", 0)}


if __name__ == "__main__":
    print(reflect_data())
