
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
semantic_memory.py — 语义记忆层: 她"学会的"知识 (区别于事件记忆)
- 分类: 道理/规则/知识/偏好/关系认知
- 来源: values提炼副产品 / 对话学习(learning强化) / 催化教学 / 好奇解答
- 生命周期: confidence 微降, 反驳-0.2, 强化+0.1
- 注入: 按主题匹配 (engine)
"""
import json
import time
import re
from pathlib import Path

STATE_FILE = BASE / "state" / "semantic.json"

DEFAULTS = {"facts": []}

# 主题词 → 类别 (注入匹配用)
CATEGORY_WORDS = {
    "道理": ["道理", "做人", "不能", "要", "答应", "骗", "借", "分享", "珍惜", "努力"],
    "知识": ["原来", "知道", "发现", "是", "会", "因为", "所以"],
    "关系": ["朋友", "妈妈", "爸爸", "小雨", "吵架", "和好", "信任"],
    "规则": ["规则", "规矩", "不能", "禁止", "应该"],
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


def _categorize(text):
    for cat, words in CATEGORY_WORDS.items():
        if any(w in text for w in words):
            return cat
    return "道理"


def learn_fact(text, source_id="", confidence=0.6, category=""):
    """学习一条语义事实 (去重: 同文本合并强化)"""
    s = load()
    text = text.strip()
    if len(text) < 4:
        return False
    cat = category or _categorize(text)
    for f in s["facts"]:
        if f["text"] == text:
            f["confidence"] = min(1.0, f["confidence"] + 0.1)
            f["learned_ts"] = time.time()
            save(s)
            return True
    s["facts"].append({"text": text, "category": cat, "source_id": source_id,
                       "learned_ts": time.time(), "confidence": confidence, "refutes": 0})
    s["facts"] = s["facts"][-200:]
    save(s)
    return True


def refute(text):
    """被反驳 → confidence 降"""
    s = load()
    for f in s["facts"]:
        if f["text"] == text:
            f["confidence"] = max(0.1, f["confidence"] - 0.2)
            f["refutes"] += 1
            save(s)
            return True
    return False


def decay():
    """时间衰减: 30天未强化 -0.05"""
    s = load()
    now = time.time()
    changed = False
    for f in s["facts"]:
        if now - f.get("learned_ts", now) > 86400 * 30:
            f["confidence"] = max(0.1, f["confidence"] - 0.05)
            changed = True
    if changed:
        save(s)
    return s


def semantic_for_text(text):
    """注入: 对话主题匹配 → 返回相关语义事实 (≤2条, 高置信优先)"""
    s = load()
    if not s["facts"]:
        return ""
    words = [w for w in re.findall(r"[\u4e00-\u9fff]{2,4}", text)]
    hits = []
    for f in sorted(s["facts"], key=lambda x: -x["confidence"]):
        if any(w in f["text"] for w in words[:6]) or \
           any(w in text for w in CATEGORY_WORDS.get(f["category"], [])[:4]):
            hits.append(f)
        if len(hits) >= 2:
            break
    if not hits:
        return ""
    return "\n".join(f"（你懂了一个道理：{f['text']}）" for f in hits)


def semantic_line():
    """全量兜底: 置信度最高1条"""
    s = load()
    if not s["facts"]:
        return ""
    top = sorted(s["facts"], key=lambda x: -x["confidence"])[0]
    return f"（你懂了一个道理：{top['text']}）"


def facts_by_category(cat):
    s = load()
    return [f for f in s["facts"] if f["category"] == cat]


if __name__ == "__main__":
    learn_fact("答应别人的事要做到", source_id="values", confidence=0.7)
    learn_fact("猫怕水", source_id="curiosity", confidence=0.6, category="知识")
    print("facts:", len(load()["facts"]))
    print("注入测试:", semantic_for_text("你答应我的事呢"))
