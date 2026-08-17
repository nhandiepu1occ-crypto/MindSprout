
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
storyline.py — 自我叙事: 她的故事线 (1.8 第二块, 精致版)
- 从记忆库提取人生节点 (年龄/年份锚定) → 分章: 幼年/童年/现在
- "我是谁"自述: 模型用她的口吻写自我介绍 (不是人设复读)
- 注入: 一句话自我感, 对话里有"我是谁"
"""
import json
import time
import re
from pathlib import Path

STATE_FILE = BASE / "state" / "story.json"
BIRTH_YEAR = 2012

DEFAULTS = {
    "profile": "",        # "我是谁"自述 (她的口吻)
    "profile_ts": 0.0,    # 自述生成时间 (每天刷新一次)
    "nodes": [],          # 故事线节点 [{age, text, emotion, chapter}]
    "chapters": [],       # 分章结构 (UI 用)
}

CHAPTERS = [
    ("baby", "👶 幼年", 0, 6, "那时候的事"),
    ("child", "🎒 童年", 7, 12, "小学的事"),
    ("now", "🏫 现在", 13, 99, "现在的我"),
]

EMOJI = {"joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨",
         "warmth": "🥰", "neutral": "·", "curiosity": "🤔", "calm": "🙂"}


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


def _age_of(mem):
    sg = getattr(mem, "scene_graph", None)
    if isinstance(sg, dict):
        try:
            if sg.get("age") is not None:
                return int(sg["age"])
        except Exception:
            pass
    y = getattr(mem, "source_year", 0) or 0
    if y:
        return max(0, y - BIRTH_YEAR)
    return None


def build_story(memory_bank):
    """从记忆库构建故事线节点 (按年龄分章, 每章取最重要的)"""
    nodes = []
    try:
        content = memory_bank.content
        ids = list(content.ids())
        for eid in ids:
            exp = content.get(eid)
            if not exp:
                continue
            text = (exp.text or "").strip()
            if len(text) < 6:
                continue
            age = _age_of(exp)
            sg = exp.scene_graph if isinstance(exp.scene_graph, dict) else {}
            imp = sg.get("importance", 0.5) if isinstance(sg, dict) else 0.5
            emo = ""
            try:
                emo = exp.emotion_vector.get("dominant", "") if exp.emotion_vector else ""
            except Exception:
                pass
            nodes.append({"eid": eid, "text": text[:90], "age": age, "imp": imp, "emo": emo})
    except Exception:
        pass
    # 按年龄排序 (未知年龄排最后)
    nodes.sort(key=lambda n: (n["age"] is None, n["age"] or 99, -n["imp"]))
    # 分章 + 每章精选 (importance 降序取 3 条, 保时间序)
    chapters = []
    for key, label, a0, a1, hint in CHAPTERS:
        ch = [n for n in nodes if n["age"] is not None and a0 <= n["age"] <= a1]
        ch.sort(key=lambda n: (n["age"], -n["imp"]))
        picked = sorted(ch, key=lambda n: -n["imp"])[:6]
        picked.sort(key=lambda n: n["age"] or 99)
        chapters.append({"key": key, "label": label, "hint": hint, "items": picked})
    return chapters, nodes


def build_profile(memory_bank, generate):
    """'我是谁'自述: 记忆摘要 + 模型写 (她的口吻, 不是作文)"""
    chapters, _ = build_story(memory_bank)
    lines = []
    for ch in chapters:
        if ch["items"]:
            lines.append(f"{ch['label']}: " + "；".join(f"{it['text'][:40]}" for it in ch["items"][:3]))
    digest = "\n".join(lines) or "（还没什么故事）"
    prompt = (f"有人问你'你是个什么样的人'。你经历过这些：\n{digest}\n\n"
              f"用你自己的话回答他。像14岁女孩跟朋友聊天那样自然地说说自己："
              f"别用'我是一个……的人'这种作文开头，别排比句，就是随口聊聊你是怎样的孩子，"
              f"80-130字。")
    try:
        text = generate(prompt, max_tokens=200, temperature=0.95)
        if len(text) < 30:
            raise ValueError("too short")
    except Exception:
        text = "我啊……就是个会为块橡皮气一礼拜、又因为一包辣条就和好的笨蛋吧。"
    return text.strip()


def refresh_if_stale(memory_bank, generate, force=False):
    """每天刷新一次自述 + 节点 (新记忆会长进故事线)"""
    s = load()
    today = time.strftime("%Y-%m-%d")
    if not force and s.get("profile_ts") and time.strftime("%Y-%m-%d", time.localtime(s["profile_ts"])) == today:
        return s
    chapters, nodes = build_story(memory_bank)
    s["nodes"] = nodes
    s["chapters"] = chapters
    s["profile"] = build_profile(memory_bank, generate)
    s["profile_ts"] = time.time()
    save(s)
    return s


def story_line():
    """注入: 一句自我感 (短, 防稀释)"""
    s = load()
    if s.get("profile"):
        m = re.search(r"我[^。！!？?]{4,28}", s["profile"])
        if m:
            return f"（你心里清楚自己是个什么样的孩子：{m.group(0)[:30]}）"
    return ""


def story_data():
    """UI 用: 章节+节点+自述"""
    s = load()
    return {"profile": s.get("profile", ""), "nodes": s.get("nodes", []),
            "chapters": s.get("chapters", []),
            "updated": s.get("profile_ts", 0)}


if __name__ == "__main__":
    print(story_data())
