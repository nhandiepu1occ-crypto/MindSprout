
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
timeline.py — 蠢珞珞的时间轴模块 (v1)
- 现实时钟同步: 每次调用都知道"现在是2026年8月15日 星期六"
- 出生设定: 2012-08-15 (今天正好14岁生日)
- 记忆时间锚定: 125条记忆 → 年份/相对表达
- 时间虚化: 越近越具体, 越远越模糊
集成: engine 记忆注入前缀 + system prompt 时间感知
"""
import re
from datetime import datetime
from pathlib import Path

BIRTH = datetime(2012, 8, 15)
CN_TZ_OFFSET = 8 * 3600  # UTC+8

WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

# ---------- 1. 现实时钟 ----------
def now() -> datetime:
    """当前现实时间 (本地 UTC+8)"""
    return datetime.now()

def now_str() -> str:
    t = now()
    return f"{t.year}年{t.month}月{t.day}日 {WEEKDAYS[t.weekday()]}"

def age_years() -> int:
    t = now()
    return t.year - BIRTH.year - ((t.month, t.day) < (BIRTH.month, BIRTH.day))

def school_year_age(year: int) -> int:
    """某年她几岁"""
    return year - BIRTH.year

# ---------- 2. 年龄/时间推断 ----------
AGE_RULES = [
    # (pattern, age_range)
    (r"幼儿园|小班|中班|大班", (3, 6)),
    (r"一年级", (6, 7)),
    (r"二年级", (7, 8)),
    (r"三年级", (8, 9)),
    (r"四年级", (9, 10)),
    (r"五年级", (10, 11)),
    (r"六年级", (11, 12)),
    (r"初一", (12, 13)),
    (r"初二", (13, 14)),
    (r"小学", (6, 12)),
    (r"初中", (12, 14)),
    (r"上幼儿园|刚会走|学走路|半夜醒|奶瓶|抱在怀里|摇篮|米糊|喂我|胳膊上|胸口|心跳声|哼着调|厚被子|地垫|塑料勺|尿布|澡盆", (0, 3)),
    (r"同桌|听写|作业|期末考试|运动会", (7, 14)),
    (r"滑梯|铲子|沙坑|过家家|跷跷板", (3, 8)),
]

def infer_age(text: str):
    """从文本推断事件发生时的年龄; 返回 (age_min, age_max) 或 None"""
    for pat, rng in AGE_RULES:
        if re.search(pat, text):
            return rng
    return None

def infer_year(text: str, source_year=None, scene_age=None):
    """推断记忆发生年份 → (year, confidence)
    优先级: 文本年龄线索(幼儿园/年级/婴儿特征) > scene_age > source_year(录入年,兜底)
    """
    t = now()
    rng = infer_age(text)
    if rng:
        return BIRTH.year + (rng[0] + rng[1]) // 2, "est"
    if scene_age:
        return BIRTH.year + int(scene_age), "est"
    if source_year and 2000 < int(source_year) <= t.year:
        return int(source_year), "exact"
    return None, "fuzzy"

# ---------- 3. 时间虚化 (相对今年) ----------
def relative_str(year, confidence="est"):
    """把年份转成相对表达 (越近越具体, 越远越模糊)"""
    t = now()
    dy = t.year - year
    if confidence == "fuzzy":
        return "小时候"
    if dy <= 0:
        return "今年"
    if dy == 1:
        return "去年"
    if dy == 2:
        return "前年"
    if dy <= 4:
        return f"{dy}年前"
    if dy <= 6:
        return "小学高年级的时候"
    if dy <= 10:
        return "小学的时候"
    return "很小的时候"

def age_str(year):
    """某年的她几岁 → 表达"""
    a = school_year_age(year)
    if a < 1:
        return "刚出生不久的时候"
    if a < 3:
        return "很小很小的时候"
    if a <= 6:
        return f"{a}岁上幼儿园的时候"
    if a <= 12:
        return f"{a}岁上小学的时候"
    return f"{a}岁上初中的时候"

# 文本内相对时间词 (优先于推断)
RELATIVE_WORDS = [
    (r"今天", "今天"),
    (r"昨天", "昨天"),
    (r"前天", "前天"),
    (r"上周|上个星期", "上周"),
    (r"这周|这个星期|这一周", "这周"),
    (r"上个月|上月", "上个月"),
    (r"前几天|前两天|几天前", "前几天"),
    (r"去年", "去年"),
    (r"前年", "前年"),
]

def memory_time_prefix(text, source_year=None, scene_age=None):
    """给一条记忆生成时间前缀: '(昨天的事)' / '(7岁上小学的时候的事)' / '(记不清什么时候)'"""
    for pat, expr in RELATIVE_WORDS:
        if re.search(pat, text or ""):
            return f"（{expr}的事）"
    year, conf = infer_year(text, source_year, scene_age)
    if conf == "fuzzy":
        return "（记不清什么时候的事了，好像是小时候）"
    return f"（{age_str(year)}的事）"

# ---------- 4. 全库标注 ----------
def tag_all_memories(memory_bank, out_file=None):
    """给记忆库所有记忆标注时间锚 → dict {exp_id: {year, confidence, prefix}}
    结果存 out_file (默认 memory_timeline.json)"""
    tags = {}
    for exp_id in memory_bank.content.ids():
        try:
            exp = memory_bank.content.get(exp_id)
            if exp is None:
                continue
            text = exp.text or ""
            sy = getattr(exp, "source_year", None) or None
            age = None
            try:
                sg = getattr(exp, "scene_graph", None) or {}
                if isinstance(sg, dict):
                    age = sg.get("age")
            except Exception:
                pass
            year, conf = infer_year(text, sy, age)
            tags[exp_id] = {
                "year": year,
                "confidence": conf,
                "prefix": memory_time_prefix(text, sy, age),
            }
        except Exception:
            continue
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(
            __import__("json").dumps(tags, ensure_ascii=False, indent=1),
            encoding="utf-8")
    return tags

def now_context_line() -> str:
    """注入 system 的时间感知行 (fix 2026-08-15: 生日条件判断 + 跨月日期用 timedelta + 前天/后天外部计算)"""
    from datetime import timedelta
    t = now()
    yest = t - timedelta(days=1)
    tom = t + timedelta(days=1)
    bf = t - timedelta(days=2)
    af = t + timedelta(days=2)
    line = (f"现在是{t.year}年{t.month}月{t.day}日{WEEKDAYS[t.weekday()]}，"
            f"你{age_years()}岁，上初二。"
            f"现在是{t.hour}点{t.minute}分（{('上午' if t.hour < 12 else '下午' if t.hour < 18 else '晚上')}）。"
            f"昨天是{yest.month}月{yest.day}日{WEEKDAYS[yest.weekday()]}，"
            f"前天是{bf.month}月{bf.day}日{WEEKDAYS[bf.weekday()]}，"
            f"明天是{tom.month}月{tom.day}日{WEEKDAYS[tom.weekday()]}。")
    # 生日锚 (V3.9.1): 固定告知, 防模型编造"下周生日"等矛盾
    if (t.month, t.day) == (BIRTH.month, BIRTH.day):
        line += "今天是你的生日（8月15日），你刚满14岁。"
    else:
        line += "你的生日是8月15日，今年已经过完了。"
    return line

if __name__ == "__main__":
    # 自测
    print("now:", now_str())
    print("age:", age_years())
    tests = [
        ("幼儿园第一天我抱着我妈大腿哭", None, None),
        ("二年级老师听写生字我写雨字多一横", None, None),
        ("六年级毕业我们把愿望纸条埋进操场", None, None),
        ("我半夜醒了妈妈抱着我摇", None, None),
        ("上个月月考数学考砸了", 2026, None),
        ("今天数学测验完了", 2026, None),
    ]
    for text, sy, age in tests:
        year, conf = infer_year(text, sy, age)
        print(f"  {text[:20]} → {year}({conf}) {relative_str(year, conf)} | {memory_time_prefix(text, sy, age)}")
