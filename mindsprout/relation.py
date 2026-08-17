
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
relation.py — 她与主人的关系状态机 v1 (2.0 场景2: 双向关系)
- 亲密值: 聊天/点赞加分, 时间衰减
- 被忽视: 3天→赌气, 5天→委屈, 7天→冷淡
- 情绪惯性: 最近一次互动的情绪基调延续
注入: system prompt 一段关系感知 (不是约束, 是她的内心状态)
"""
import json
import time
from pathlib import Path

STATE_FILE = BASE / "state" / "relation.json"

DEFAULTS = {
    "intimacy": 50.0,        # 0-100
    "last_interact_ts": 0.0,  # 上次互动(对话/点赞/评论)
    "last_mood": "normal",    # normal | sulky(赌气) | hurt(委屈) | cold(冷淡) | warm
    "mood_until": 0.0,        # 情绪惯性到期时间
    "sulky_count": 0,         # 累计被忽视次数
    "caller_name": "",        # 他让珞珞怎么称呼他 (楚门 v1.8: 对话者身份锚)
}

# 称呼提取: "叫我XX" / "以后叫我XX" / "你可以叫我XX" / "我叫XX"
CALLER_PATTERNS = [
    r"(?:以后|以后就|你可以|可以|你就|就)?叫我([\u4e00-\u9fa5A-Za-z0-9]{1,6})(?:吧|就行|就行吧|好了|就好)?",
]

def extract_caller_name(text: str) -> str:
    """从对方话语中提取称呼 (他让我怎么称呼他)"""
    import re
    # 先去句尾标点, 再剥离语气词/后缀, 最后匹配名字
    t = re.sub(r"[。.!！~～？?]", "", text.strip())
    t = re.sub(r"(就行吧|就行|吧|好了|就好|就行啦|哦|呀|嘛)$", "", t)
    for pat in (r"叫我([\u4e00-\u9fa5A-Za-z0-9]{1,6})",):
        m = re.search(pat, t)
        if m:
            name = m.group(1).strip()
            # 排除常见非称呼词
            if name and name not in ("小雨", "小美", "珞珞", "小珞珞", "老师", "妈妈", "爸爸"):
                return name
    return ""

def load():
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return {**DEFAULTS, **s}
        except Exception:
            pass
    return dict(DEFAULTS)

def save(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")

def _days(s):
    if not s["last_interact_ts"]:
        return 0.0
    return (time.time() - s["last_interact_ts"]) / 86400

def interact(weight=1.0, mood="warm", text=""):
    """每次互动(对话/点赞/评论): 亲密+2, 重置被忽视, 情绪基调; 提取称呼"""
    s = load()
    s["intimacy"] = min(100.0, s["intimacy"] + 2.0 * weight)
    s["last_interact_ts"] = time.time()
    s["last_mood"] = mood
    s["mood_until"] = time.time() + 7200  # 情绪基调延续2小时(对话中)
    # 和好后赌气清零
    if mood == "warm":
        s["sulky_count"] = max(0, s["sulky_count"] - 1)
    # 称呼记忆 (楚门 v1.8): 他说"叫我XX" → 记住
    if text:
        name = extract_caller_name(text)
        if name:
            s["caller_name"] = name
    save(s)
    return s

def tick():
    """每日衰减 + 被忽视状态演化 (由 scheduler 调用)"""
    s = load()
    days = _days(s)
    # 亲密值缓慢衰减 (每天 -1, 最低 10)
    s["intimacy"] = max(10.0, s["intimacy"] - 1.0)
    # 被忽视状态
    if days >= 7:
        s["last_mood"] = "cold"       # 7天: 冷淡
    elif days >= 5:
        s["last_mood"] = "hurt"       # 5天: 委屈
    elif days >= 3:
        s["last_mood"] = "sulky"      # 3天: 赌气
    else:
        if time.time() > s.get("mood_until", 0):
            s["last_mood"] = "normal"
    save(s)
    return s

def relation_line():
    """注入 system 的关系感知 (她的内心状态, 不是约束)"""
    s = load()
    days = _days(s)
    mood = s["last_mood"]
    parts = []
    # 称呼记忆 (楚门 v1.8): 他让珞珞叫他什么
    if s.get("caller_name"):
        parts.append(f"（他说他叫{s['caller_name']}，你心里默念着这个名字：{s['caller_name']}……记住了，以后就这么叫他。）")
    if days >= 3:
        if mood == "sulky":
            parts.append("（你心里有点小意见：他好几天没来找你了。你才不承认你想他，你就是有点不高兴，还有点空落落的。）")
        elif mood == "hurt":
            parts.append("（你心里有点委屈：他都好几天没影了，是不是把你忘了。你决定他来找你也不主动理他——但要是他真的不来，你又会难过。）")
        elif mood == "cold":
            parts.append("（你心里已经有点冷了：这么多天没联系，你觉得他大概不在乎你。你不想表现出在意，可心里还是会想他。）")
    elif s["intimacy"] >= 80:
        parts.append("（你心里挺亲近他的，见到他说话都轻快些。）")
    elif s["intimacy"] <= 25:
        parts.append("（你和他还不太熟，说话有点拘谨。）")
    return "\n".join(parts)

if __name__ == "__main__":
    s = load()
    print("当前关系:", json.dumps(s, ensure_ascii=False))
    print("注入行:", relation_line())
