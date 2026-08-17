
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
imagination.py — 想象系统 (1.8 第一块: 有想法的孩子)
- 白日梦: 白天大脑随机走神 → 幻想片段 (与夜晚的梦对称)
- 愿望: 白日梦里自然流露的小心愿 → 收集到 wishes.json
- 注入: daydream_line 让她对话里有想象力 ("如果我会飞…")
"""
import json
import time
import random
from pathlib import Path

STATE_FILE = BASE / "state" / "daydream.json"
WISH_FILE = BASE / "state" / "wishes.json"

DEFAULTS = {
    "last_daydream": None,    # 上次白日梦时间
    "daydream": "",           # 当前白日梦
    "count": 0,               # 今天做了几次白日梦
    "day": "",                # 计数日期
}

# 白日梦主题池 (从她的世界/生活抽取)
TOPICS = [
    "如果我会飞，上课铃响的时候我就从窗户飞出去",
    "如果我变成了一只猫，就趴在窗台上看一整天的人",
    "如果我能听懂小动物说话，路边的小狗每天都跟我告状",
    "如果我有一台时光机，就回到六岁那年多要一块糖",
    "如果明天没有作业，我就把书包扔到天花板上",
    "如果我变成大人，第一件事就是买一冰箱的冰淇淋",
    "如果我能隐身，就偷偷去看看小雨上课在干什么",
    "如果我考试考了满分，爸爸会不会高兴得把我举起来",
    "如果我能跟天上的云说话，就让它帮我挡住讨厌的体育课",
    "如果有一天我长大了，还会不会记得现在这些乱七八糟的想法",
]

DAYDREAM_PROMPT = """你上课/发呆走神了，脑子里飘过一个白日梦：{topic}。
用你的话描述这个白日梦（第一人称，40-60字，带画面感，天马行空没关系，像14岁女孩的幻想）。
最后用一句"其实我偷偷希望……"结尾，说一个藏在心里的小心愿。"""


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


def load_wishes():
    if WISH_FILE.exists():
        try:
            return json.loads(WISH_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_wishes(w):
    WISH_FILE.parent.mkdir(parents=True, exist_ok=True)
    WISH_FILE.write_text(json.dumps(w, ensure_ascii=False, indent=1), encoding="utf-8")


def _extract_wish(text):
    """从白日梦文本提取'其实我偷偷希望……'后的愿望"""
    import re
    m = re.search(r"其实我偷偷希望[：:，,]?\s*([^。！!]+)", text)
    if m:
        w = m.group(1).strip().strip("。！!～~")
        if 3 <= len(w) <= 40:
            return w
    return ""


def make_daydream(generate):
    """生成一次白日梦 (大脑驱动: 随机主题 + 模型幻想 + 愿望收集)
    generate: 文本生成函数 (humanizer.generate 包装)
    """
    s = load()
    topic = random.choice(TOPICS)
    try:
        text = generate(DAYDREAM_PROMPT.format(topic=topic), max_tokens=140, temperature=1.05)
        text = text.strip()
        if len(text) < 15:
            raise ValueError("too short")
    except Exception:
        text = f"如果我会飞……{topic[2:12]}，想着想着就笑了"
    s["daydream"] = text
    s["last_daydream"] = time.time()
    today = time.strftime("%Y-%m-%d")
    if s.get("day") != today:
        s["day"] = today
        s["count"] = 0
    s["count"] = s["count"] + 1
    save(s)
    # 愿望收集
    wish = _extract_wish(text)
    if wish:
        w = load_wishes()
        w.append({"text": wish, "ts": time.strftime("%Y-%m-%d %H:%M")})
        save_wishes(w[-20:])  # 保留最近20个
    return s["daydream"]


def daydream_line():
    """注入: 当前幻想状态 (短, 防爆炸)"""
    s = load()
    if s.get("daydream"):
        return f"（你刚才走神了，脑子里飘过：{s['daydream'][:50]}）"
    return ""


def wishes_line():
    """注入: 最近的愿望 (可选, 只取最近1条)"""
    w = load_wishes()
    if w:
        return f"（你心里有个小心愿：{w[-1]['text'][:30]}）"
    return ""


if __name__ == "__main__":
    def fake_gen(prompt, max_tokens=140, temperature=1.05):
        return "如果我变成了一只猫，就趴在窗台上看一整天的人。阳光晒着尾巴尖，楼下卖辣条的爷爷抬头看见我，冲我笑了一下。其实我偷偷希望，能一直这样什么也不用想。"
    print(make_daydream(fake_gen))
    print("愿望:", load_wishes()[-1:])
    print("注入:", daydream_line())
