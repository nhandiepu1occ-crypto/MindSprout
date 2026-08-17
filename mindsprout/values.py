
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
values.py — 价值观浮现 (1.8 第三块): 她心里有杆秤
- 从情绪最强烈的记忆里提炼 (不是预设, 是她的生活长出来的)
- 条件注入: 对话主题命中该价值观时才注入 (attention式, 不稀释)
- 在线学习: 她流露"我讨厌/我最不能忍…" → 候选池
- 耦合: memory_bank(强烈记忆) / storyline(来源回溯) / role_registry(关联角色) / engine(条件注入)
"""
import json
import time
import re
from pathlib import Path

STATE_FILE = BASE / "state" / "values.json"

DEFAULTS = {
    "values": [],      # [{text, source, strength, roles, ts}]
    "candidates": [],  # 在线学习候选 (对话中流露的原则)
    "refined_ts": 0.0, # 上次提炼时间 (3天一次)
}

# 主题词 → 价值观匹配 (条件注入用)
TOPIC_WORDS = ["朋友", "撒谎", "骗", "公平", "冤枉", "作业", "辣条", "妈妈", "爸爸",
               "老师", "秘密", "答应", "承诺", "借", "还", "钱", "考试", "分数",
               "转学", "分别", "死", "怕", "道歉", "原谅", "谢谢", "对不起", "帮忙",
               "分享", "抢", "偷", "背叛", "和好"]

REFINE_PROMPT = """你经历过很多事，有些事让你特别有感触。看看这些：
{memories}

用你自己的话，说出 3-5 条你在乎的事、你心里的原则（比如"骗人的时候心里会堵堵的"、"朋友吵架了第二天还是要和好"）。
每行一条，只说原则本身，不要解释原因，不要写"因为"，不要写"我觉得"。"""


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


def extract_strong_memories(memory_bank, top=12):
    """情绪最强烈的记忆 (价值观提炼池)
    双池: 优先 7+岁记忆 (现在的她能反思的), 不足才回退幼年"""
    scored = []
    try:
        content = memory_bank.content
        for eid in list(content.ids()):
            exp = content.get(eid)
            if not exp:
                continue
            text = (exp.text or "").strip()
            if len(text) < 10:
                continue
            ev = exp.emotion_vector if isinstance(exp.emotion_vector, dict) else {}
            val = abs(ev.get("valence", 0.3))
            aro = ev.get("arousal", 0.3)
            sg = exp.scene_graph if isinstance(exp.scene_graph, dict) else {}
            imp = sg.get("importance", 0.5)
            age = None
            try:
                if sg.get("age") is not None:
                    age = int(sg["age"])
            except Exception:
                pass
            scored.append((val * aro * imp, text[:80], eid, age))
    except Exception:
        pass
    scored.sort(key=lambda x: -x[0])
    grown = [s for s in scored if s[3] is not None and s[3] >= 7]
    baby = [s for s in scored if s[3] is None or s[3] < 7]
    if len(grown) >= 6:
        return grown[:top]
    return (grown + baby)[:top]


def refine(generate, memory_bank):
    """提炼: 逐条强记忆 → 一句话道理 (单条生成, 格式可靠; 来源=该记忆本身)"""
    strong = extract_strong_memories(memory_bank)
    if len(strong) < 4:
        return []
    values = []
    for _, t, _, _ in strong[:5]:
        q = (f"你经历过这件事：{t}\n\n"
             f"这件事让你心里明白了一个什么道理？用一句话说（14岁女孩的口吻，"
             f"20-40字，直接说那句话，不要'我明白了''我觉得''要'这种开头）。")
        try:
            ans = generate(q, max_tokens=70, temperature=0.9).strip()
        except Exception:
            continue
        ans = re.sub(r"[a-zA-Z]+", "", ans).strip("，。！!")  # 去英文残留
        # 多句输出 → 按句拆分为多条价值观
        segs = [s.strip().strip("，。！!")
                for s in re.split(r"[。！!？?；;]", ans)
                if 8 <= len(s.strip()) <= 40]
        for seg in segs[:3]:
            # 质量过滤: 排除叙述性/事件重述 (价值观应是原则句)
            if seg.startswith(("我明白", "我觉得", "我认为", "后来", "这次", "哎呀", "那次", "我记得", "然后", "最后", "下次")):
                continue
            if any(w in seg for w in ("我念", "我写", "我把", "我读", "我数", "我气", "我脸")):
                continue
            values.append({"text": seg, "source": t[:50], "strength": 0.6,
                           "roles": [], "ts": time.strftime("%Y-%m-%d")})
        if len(values) >= 6:
            break
    return values


def refresh_if_stale(generate, memory_bank, force=False):
    """每3天提炼一次; 新强烈记忆也会在下次提炼时纳入"""
    s = load()
    now = time.time()
    if not force and s.get("refined_ts") and now - s["refined_ts"] < 86400 * 3:
        return s
    new_values = refine(generate, memory_bank)
    if new_values:
        s["values"] = new_values
        s["refined_ts"] = now
        save(s)
    return s


def learn_candidate(text):
    """在线学习: 对话中流露的原则 → 候选池 (persona式学习)"""
    s = load()
    m = re.search(r"(?:我|我最|我可)(?:讨厌|烦|不能忍|受不了|最怕|最看重|觉得|认为|相信)[：:，,]?\s*([^。！!？?，,]{4,30})", text)
    if not m:
        return False
    cand = m.group(1).strip()
    if len(cand) < 4:
        return False
    s["candidates"].append({"text": cand, "ts": time.strftime("%Y-%m-%d %H:%M")})
    s["candidates"] = s["candidates"][-20:]
    save(s)
    return True


def values_for_text(text):
    """条件注入: 对话主题命中 → 返回相关价值观行 (最多2条)"""
    s = load()
    vals = s.get("values", [])
    if not vals:
        return ""
    hits = []
    for v in vals:
        # 命中: 来源事件关键词 or 价值观文本关键词
        if any(w in text for w in TOPIC_WORDS if len(w) >= 2) or \
           any(w in v["text"] for w in re.findall(r"[\u4e00-\u9fff]{2,}", text)[:8]):
            hits.append(v)
        if len(hits) >= 2:
            break
    if not hits:
        return ""
    lines = [f"（你心里有杆秤：{v['text']}）" for v in hits]
    return "\n".join(lines)


def values_line():
    """全量注入 fallback: strength 前2条 (轻量)"""
    s = load()
    vals = sorted(s.get("values", []), key=lambda v: -v.get("strength", 0.5))[:2]
    if not vals:
        return ""
    return "（你心里有杆秤：" + "；".join(v["text"] for v in vals) + "。）"


if __name__ == "__main__":
    print(json.dumps(load(), ensure_ascii=False, indent=1)[:800])
