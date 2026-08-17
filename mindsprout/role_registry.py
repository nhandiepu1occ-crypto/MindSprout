
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
role_registry.py — 关系系统 V2: 虚拟社会角色注册表
- 每个角色独立状态: 亲密度/最近互动/语气基调/最近事件/想念值
- 对话提及角色 → 关系更新; 角色语气 → 注入 system 影响对话
- 虚拟社会扩展: type=subai 的子AI 与人类角色同一状态机 (未来子AI 直接接入)
"""
import json
import time
import re
from pathlib import Path

ROLE_FILE = BASE / "state" / "roles.json"

# 角色初始注册表 (从她的生活记忆推断)
DEFAULTS = {
    "主人":   {"type": "human", "intimacy": 100.0, "last": "他今天让我喝水", "tone": "warm", "events": [], "birthday": "", "emoji": "🧑"},
    "小雨":   {"type": "human", "intimacy": 72.0, "last": "中午一起喝了冬瓜排骨汤", "tone": "warm", "events": [], "birthday": "2026-11-03", "emoji": "👧"},
    "小敏":   {"type": "human", "intimacy": 55.0, "last": "她夸我向日葵画得好", "tone": "warm", "events": [], "birthday": "", "emoji": "👩‍🎨"},
    "妈妈":   {"type": "family", "intimacy": 90.0, "last": "早上做了番茄鸡蛋面", "tone": "warm", "events": [], "birthday": "", "emoji": "👩"},
    "爸爸":   {"type": "family", "intimacy": 80.0, "last": "晚上加班还没回来", "tone": "warm", "events": [], "birthday": "", "emoji": "👨"},
    "李老师": {"type": "teacher", "intimacy": 45.0, "last": "上课讲了分数乘法", "tone": "neutral", "events": [], "birthday": "", "emoji": "👓"},
    # 虚拟社会预留: 子AI 也是社会成员 (同一状态机)
    "future-1": {"type": "subai", "intimacy": 10.0, "last": "还没苏醒", "tone": "sleeping", "events": [], "birthday": "", "emoji": "🌙"},
}

# 提及关键词 → 角色
MENTION_PATTERNS = [
    ("主人", [r"主人", r"哥哥", r"第家看"]),
    ("小雨", [r"小雨"]),
    ("小敏", [r"小敏"]),
    ("妈妈", [r"妈妈", r"妈"]),
    ("爸爸", [r"爸爸", r"爸"]),
    ("李老师", [r"老师", r"李老师"]),
]

# 角色情绪词 → 语气基调 (吵架/委屈等)
TONE_WORDS = {
    "sulky": ["吵架", "不理我", "生气", "气死", "闹别扭"],
    "hurt": ["委屈", "哭了", "伤心", "难过", "不理"],
    "warm": ["和好", "一起", "开心", "夸", "送"],
}


def load():
    if ROLE_FILE.exists():
        try:
            s = json.loads(ROLE_FILE.read_text(encoding="utf-8"))
            merged = dict(DEFAULTS)
            merged.update(s)  # 保留新角色, 补默认字段
            return merged
        except Exception:
            pass
    return dict(DEFAULTS)


def save(s):
    ROLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ROLE_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def _norm(name):
    """主人称呼归一化"""
    if name in ("第家看", "哥哥", "主人"):
        return "主人"
    return name


def mention(text):
    """识别对话中提及的角色 → 更新最近互动"""
    s = load()
    hit = None
    for role, pats in MENTION_PATTERNS:
        for p in pats:
            if re.search(p, text):
                hit = role
                break
        if hit:
            break
    if hit:
        s[hit]["last"] = text.strip()[:60]
        s[hit]["tone"] = _tone_from(text, s[hit].get("tone", "warm"))
        save(s)
    return hit


def _tone_from(text, cur):
    for tone, words in TONE_WORDS.items():
        for w in words:
            if w in text:
                return tone
    return cur


def interact(role, weight=1.0, mood="warm", text=""):
    """显式互动 (点赞她的朋友圈/评论/聊天提到)"""
    role = _norm(role)
    s = load()
    if role not in s:
        s[role] = {"type": "human", "intimacy": 30.0, "last": "", "tone": "neutral",
                   "events": [], "birthday": "", "emoji": "👤"}
    s[role]["intimacy"] = max(0, min(100, s[role]["intimacy"] + 2.0 * weight))
    if mood == "warm":
        s[role]["tone"] = "warm"
    if text:
        s[role]["last"] = text.strip()[:60]
    save(s)
    return s[role]


def tick():
    """每日: 亲密度缓慢衰减 + 语气回归"""
    s = load()
    for r, d in s.items():
        if d.get("type") == "subai":
            continue  # 未苏醒的子AI不衰减
        d["intimacy"] = max(5.0, d["intimacy"] - 0.5)
        if d.get("tone") in ("sulky", "hurt") and d.get("tone_changed_ts", 0) and \
                time.time() - d["tone_changed_ts"] > 86400 * 2:
            d["tone"] = "warm"  # 2天后气消了
    save(s)
    return s


def registry_line():
    """注入: 最高情绪/最亲近的 2 个角色状态 (影响对话语气)"""
    s = load()
    parts = []
    # 语气异常的角色优先 (吵架影响)
    for r, d in s.items():
        if d.get("tone") in ("sulky", "hurt") and d.get("type") != "subai":
            feel = {"sulky": "心里还有点气", "hurt": "心里有点委屈"}[d["tone"]]
            parts.append(f"（提到{r}你{feel}，{d.get('last', '')[:20]}）")
    # 最亲近的角色
    ranked = sorted([(r, d) for r, d in s.items()
                     if d.get("type") != "subai" and d.get("tone") not in ("sulky", "hurt")],
                    key=lambda x: -x[1]["intimacy"])[:1]
    for r, d in ranked:
        if d["intimacy"] >= 70:
            parts.append(f"（你心里挺亲近{r}的，{d.get('last', '')[:20]}）")
    return "\n".join(parts[:2])


def role_list():
    """UI 用: 全部角色状态"""
    s = load()
    return [{"name": r, **d, "intimacy": round(d.get("intimacy", 0))}
            for r, d in s.items()]


if __name__ == "__main__":
    print(json.dumps(role_list(), ensure_ascii=False, indent=1))
    print("注入:", registry_line())
