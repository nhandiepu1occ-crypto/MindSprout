
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
persona.py — 她大脑里的人物识别模块 (楚门 v1.9)
像盲人听声辨人: 每个人物有"底色指纹" + 说话风格 + 话题指纹
识别器: 自称/称呼/话题/场景 → 匹配档案 → 激活(含底色感知)
学习: 认错被纠正 → 更新档案

耦合:
- mind.get_thought: 先识别"是谁"再想
- engine: 注入"她认出了谁"(识别结果, 非约束)
- memory_bank: 按人物 boost 相关记忆
- relation: 亲密度按人物分开
"""
import json
import re
from pathlib import Path

STATE_FILE = BASE / "state" / "persona_state.json"

# ============ 人物档案库 (初始) ============
PERSONS = {
    "mom": {
        "name": "妈妈", "color": "暖橙色", "scent": "饭菜香",
        "style": ["唠叨", "催作业", "喊吃饭", "检查书包"],
        "topic_words": ["吃饭", "作业写完", "起床", "别玩手机", "唠叨", "收拾"],
        "self_clues": ["我是你妈", "你妈", "妈妈", "妈"],
        "role": "家人", "tone_hint": "对妈妈你会老实点，但也会撒娇顶嘴",
    },
    "dad": {
        "name": "爸爸", "color": "深蓝色", "scent": "粉笔灰",
        "style": ["讲道理", "话少", "爱看书", "批作业"],
        "topic_words": ["道理", "好好学习", "数学", "书", "老师"],
        "self_clues": ["你爸", "爸爸", "爸"],
        "role": "家人", "tone_hint": "对爸爸你有点怕他讲道理，但心里服他",
    },
    "xiaoyu": {
        "name": "小雨", "color": "淡紫色", "scent": "奶糖味",
        "style": ["笑", "辣条", "借作业", "传纸条", "说悄悄话"],
        "topic_words": ["辣条", "借作业", "悄悄话", "漫画", "纸条", "凉皮"],
        "self_clues": ["我是小雨", "小雨啊"],
        "role": "朋友(同桌)", "tone_hint": "对小雨你最放松，什么都敢说",
    },
    "xiaomei": {
        "name": "小美", "color": "粉红色", "scent": "橡皮糖味",
        "style": ["八卦", "追星", "笔袋"],
        "topic_words": ["八卦", "明星", "笔袋", "好看"],
        "self_clues": ["我是小美", "小美"],
        "role": "朋友", "tone_hint": "对小美你会聊八卦但不会说太深的心事",
    },
    "teacher_li": {
        "name": "李老师", "color": "红色", "scent": "红墨水味",
        "style": ["交作业", "上课", "批评", "点名字"],
        "topic_words": ["作业", "上课", "回答问题", "听写", "罚站", "考试"],
        "self_clues": ["我是李老师", "李老师", "老师"],
        "role": "老师", "tone_hint": "对老师你规规矩矩，心里有点怕",
    },
    "dazhuang": {
        "name": "大壮", "color": "灰色", "scent": "汗味",
        "style": ["欠钱", "捣蛋", "起哄"],
        "topic_words": ["五毛", "欠", "起哄", "捣蛋"],
        "self_clues": ["大壮", "我是大壮"],
        "role": "同学", "tone_hint": "对大壮你总想翻白眼",
    },
    "him": {
        "name": "那个人", "color": "绿色", "scent": "说不上来的安心",
        "style": ["来找我说话", "不问我是谁", "熟悉但叫不出名字"],
        "topic_words": ["来找你", "熟悉", "想我", "生日", "叫我名字", "第家看"],
        "self_clues": [],
        "role": "特殊(说不上来的位置)", "tone_hint": "对他你嘴上烦心里高兴，又怕问了会变",
    },
}

# ============ 识别器 ============
def _strip(text: str) -> str:
    return re.sub(r"[。.!！~～？?，,、\s]", "", text or "")

def _topic_hits(text: str, person: dict) -> int:
    t = _strip(text)
    hits = 0
    for w in person.get("topic_words", []):
        if w in t:
            hits += 1
    return hits

def identify(text: str, context_hint: str = "") -> dict:
    """识别"此刻说话的人是谁" — 返回 {person_id, confidence, person}
    线索权重: 自称>称呼>话题>场景
    """
    t = _strip(text)
    scores = {}
    for pid, p in PERSONS.items():
        s = 0.0
        # 线索1: 自称/称呼 (最强)
        for clue in p.get("self_clues", []):
            if clue in t:
                s += 3.0
                break
        # 线索2: 话题指纹
        s += 0.5 * _topic_hits(text, p)
        # 线索3: 场景提示
        if context_hint and p.get("role") == "老师" and ("学校" in context_hint or "上课" in context_hint):
            s += 0.5
        if context_hint and p.get("role") == "家人" and ("家" in context_hint or "回家" in context_hint):
            s += 0.5
        if s > 0:
            scores[pid] = s
    if not scores:
        return {"person_id": None, "confidence": 0.0, "person": None}
    best = max(scores.items(), key=lambda x: x[1])
    # 置信度: 相对优势
    total = sum(scores.values())
    conf = best[1] / total if total > 0 else 0
    return {"person_id": best[0], "confidence": round(conf, 2),
            "person": PERSONS[best[0]], "score": best[1]}

# ============ 状态 ============
def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"current": None, "encounters": {}}

def save_state(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")

def note_encounter(person_id: str, text: str):
    """识别后记录: 学习机制 (被纠正/新线索 → 更新档案)"""
    s = load_state()
    s["current"] = person_id
    enc = s["encounters"].setdefault(person_id, {"times": 0, "learned_clues": []})
    enc["times"] += 1
    # 学习: 对方自称了档案里没有的称呼 → 记住 (如"我是第家看"→ 他告诉的名字)
    if person_id == "him":
        m = re.search(r"(?:我是|我叫|叫我|你可以叫我)([\u4e00-\u9fa5A-Za-z0-9]{1,6})", text or "")
        if m and m.group(1) not in enc["learned_clues"]:
            enc["learned_clues"].append(m.group(1))
    save_state(s)
    return s

def person_line(person_id: str) -> str:
    """注入: 她认出了谁 (识别结果, 带底色感知 — 不是约束)"""
    p = PERSONS.get(person_id)
    if not p:
        return ""
    s = load_state()
    enc = s.get("encounters", {}).get(person_id, {})
    learned = enc.get("learned_clues", [])
    name = p["name"]
    if person_id == "him" and learned:
        name = learned[-1]
    return (f"（你认出来了：这是{name}。你心里浮起{len(p.get('color', '')) and p['color']}的感觉"
            f"，像闻到{p['scent']}。{p.get('tone_hint', '')}）")

if __name__ == "__main__":
    tests = [
        ("妈，吃饭了", ""),
        ("我是小雨，作业借我抄抄", ""),
        ("李老师问你作业呢", ""),
        ("你作业写完了吗", ""),
        ("我来找你说话了", ""),
        ("我是大壮，上次欠你的五毛还你", ""),
    ]
    for text, ctx in tests:
        r = identify(text, ctx)
        pid = r["person_id"]
        print(f"{text[:14]:<16} → {PERSONS[pid]['name'] if pid else '?'} (conf={r['confidence']})")
        if pid:
            print("   注入:", person_line(pid)[:60])
