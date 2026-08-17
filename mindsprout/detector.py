"""
7项客观AI腔检测指标 v2（按代码审查重构）

v2 修复：
  1. 分句：加 …——" 结尾符；分词含英数
  2. 节奏：std/mean 相对判定 + 短文本跳过
  3. AI词：强/弱分层加权，密度按千字
  4. 模板：固定短语模板（删"是"万能正则）+ 第一人称豁免 + 全文占比
  5. 个人性：高质量细节（数字/时间词/品牌）+ 去省份噪声
  6. 连接词：按句子数 + 成对检测
  7. 情感：基础词扩充 + 否定处理
  8. 语气词：去标点取汉字 + 词集扩充
  9. 统一 severity 字段
  10. 空文本安全默认
"""

from mindsprout.config import BASE

import re
import statistics
from typing import List, Dict, Tuple


# ============================================================
# 工具函数
# ============================================================

def split_sentences(text: str) -> List[str]:
    """中文分句（v2: 含 …——" 结尾符）"""
    parts = re.split(r'[。！？；…——"\n]+', text)
    return [s.strip() for s in parts if s.strip()]


def split_words(text: str) -> List[str]:
    """分词（v2: 中文 + 英数）"""
    return re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text)


def _safe_text(text: str) -> bool:
    """空文本防御"""
    return bool(text and text.strip())


# ============================================================
# 检测1: 句子节奏（v2: 相对标准差 + 短文本豁免）
# ============================================================

def detect_sentence_rhythm(text: str) -> Dict:
    sentences = split_sentences(text)
    total_len = len(text.replace(" ", ""))
    name = "句子节奏"

    # 短文本/句子太少：无法判定，安全通过
    if not _safe_text(text) or len(sentences) < 3 or total_len < 30:
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False,
                "suggestion": "文本过短，跳过节奏检测"}

    lengths = [len(s) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    if mean_len < 3:
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False,
                "suggestion": "句子过短，跳过"}

    std = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    relative = std / mean_len  # 相对节奏

    # 相对标准差 < 0.3 → 节奏过于均匀（AI特征）
    severity = max(0.0, min(1.0, (0.3 - relative) * 3))
    return {
        "name": name, "score": 1.0 - severity, "severity": severity,
        "issue_detected": severity > 0.5,
        "suggestion": f"句长节奏过于均匀（相对标准差{relative:.2f}，AI常写成等长句）" if severity > 0.5 else "节奏正常",
        "detail": {"std": round(std, 1), "mean": round(mean_len, 1), "relative": round(relative, 2)},
    }


# ============================================================
# 检测2: AI 高频词汇（v2: 强/弱分层加权）
# ============================================================

STRONG_AI_WORDS = [
    "值得注意的是", "综上所述", "毋庸置疑", "不言而喻", "众所周知", "不可否认",
    "总体而言", "由此可见", "从XX角度来看", "在当今", "具有重要意义",
    "可以预见", "在某种程度上", "基于上述分析", "综合来看",
    "换言之", "换句话说", "需要强调的是", "在当前形势下",
]

WEAK_AI_WORDS = [
    "推动", "促进", "提升", "优化", "赋能", "落地", "打造", "构建", "聚焦",
    "深耕", "布局", "极大地", "深刻地", "广泛地", "有效地", "积极地",
    "一方面", "另一方面", "首先", "其次", "最后", "与此同时",
]


def detect_ai_vocabulary(text: str) -> Dict:
    name = "AI高频词"
    if not _safe_text(text):
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "空文本"}

    strong_hits = [w for w in STRONG_AI_WORDS if w in text]
    weak_hits = [w for w in WEAK_AI_WORDS if w in text]
    # 加权：强词 2 分/个，弱词 1 分/个；密度 = 加权分 / 千字
    weighted = len(strong_hits) * 2 + len(weak_hits)
    density = weighted / max(1, len(text) / 1000)

    # 强词出现 1 个就算可疑；密度 > 3/千字 触发
    severity = 0.0
    if strong_hits:
        severity = min(1.0, 0.5 + 0.15 * len(strong_hits))
    elif density > 3:
        severity = min(1.0, density / 12)

    return {
        "name": name, "score": 1.0 - severity, "severity": severity,
        "issue_detected": severity > 0.4,
        "suggestion": f"强AI词{strong_hits[:3]} 弱AI词×{len(weak_hits)} 密度{density:.1f}/千字",
        "detail": {"strong": strong_hits, "weak_count": len(weak_hits), "density": round(density, 2)},
    }


# ============================================================
# 检测3: 模板化结构（v2: 固定短语模板 + 第一人称豁免 + 全文占比）
# ============================================================

TEMPLATE_PATTERNS = [
    re.compile(r"在[^。，]{0,8}背景下"),
    re.compile(r"随着[^。，]{0,12}的发展"),
    re.compile(r"具有[^。，]{0,8}意义"),
    re.compile(r"不仅[^。，]{0,20}而且"),
    re.compile(r"一方面[^。，]{0,20}另一方面"),
    re.compile(r"为[^。，]{0,10}提供了[^。，]{0,8}"),
    re.compile(r"有利于[^。，]{0,10}"),
    re.compile(r"综上所述"),
    re.compile(r"由此可见"),
]


def detect_template_structure(text: str) -> Dict:
    name = "模板结构"
    if not _safe_text(text):
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "空文本"}

    sentences = split_sentences(text)
    has_first_person = ("我" in text) or ("我们" in text)

    template_count = 0
    for s in sentences:
        for p in TEMPLATE_PATTERNS:
            if p.search(s):
                template_count += 1
                break

    ratio = template_count / len(sentences)
    # 第一人称出现 → 豁免一半（真人也会偶尔用）
    effective = ratio * (0.5 if has_first_person else 1.0)
    severity = min(1.0, effective * 3)

    return {
        "name": name, "score": 1.0 - severity, "severity": severity,
        "issue_detected": severity > 0.4,
        "suggestion": f"模板句占比{ratio:.0%}" + ("（含第一人称，部分豁免）" if has_first_person else ""),
        "detail": {"template_count": template_count, "ratio": round(ratio, 2), "has_first_person": has_first_person},
    }


# ============================================================
# 检测4: 个人性缺失（v2: 高质量细节 + 时间具体性）
# ============================================================

DETAIL_PATTERNS = [
    re.compile(r"\d{2,4}年"),          # 2021年
    re.compile(r"\d{1,2}月"),          # 3月
    re.compile(r"\d+[岁天周块个次回]"),  # 5岁/3天/2次
    re.compile(r"(昨天|前天|今天|上周|这周|去年|上个月|小时候|当年)"),
    re.compile(r"(淘宝|京东|美团|微信|QQ|抖音|B站|知乎|小红书)"),
    re.compile(r"(麦当劳|肯德基|星巴克|苹果|华为|小米|OPPO|vivo)"),
    re.compile(r"\d{3,}-\d{3,}"),      # 电话号码样式
]


def detect_personal_absence(text: str) -> Dict:
    name = "个人性"
    if not _safe_text(text):
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "空文本"}
    if len(text) < 20:
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "超短文本跳过"}

    # v2: "我们"是集体表达，不算个人色彩（AI 爱用"我们应当"）；(?!们) 排除"我们"里的"我"
    first_person_count = len(re.findall(r"(?:我|咱|俺)(?!们)", text))
    detail_count = sum(len(p.findall(text)) for p in DETAIL_PATTERNS)
    has_time_word = bool(re.search(r"(昨天|前天|今天|上周|去年|上个月|小时候|当年)", text))

    # 综合判定：第一人称 或 高质量细节≥2 或 时间词 → 有个人性
    personal = first_person_count >= 1 or detail_count >= 2 or has_time_word
    severity = 0.0 if personal else 0.8
    if first_person_count == 0 and detail_count == 0 and not has_time_word:
        severity = 0.9

    return {
        "name": name, "score": 1.0 - severity, "severity": severity,
        "issue_detected": severity > 0.5,
        "suggestion": f"第一人称×{first_person_count} 细节×{detail_count} 时间词:{has_time_word}",
        "detail": {"first_person": first_person_count, "details": detail_count, "time_word": has_time_word},
    }


# ============================================================
# 检测5: 连接词密度（v2: 按句子数 + 成对检测）
# ============================================================

CONNECTORS = {
    "因此", "所以", "但是", "然而", "此外", "另外", "同时", "而且", "并且",
    "不过", "虽然", "因为", "由于", "如果", "那么", "总之", "综上所述",
    "首先", "其次", "最后", "一方面", "另一方面", "由此", "进而", "从而",
}

PAIRED_CONNECTORS = [
    (re.compile(r"虽然[^。！？]{0,30}但是"), 1.0),
    (re.compile(r"不仅[^。！？]{0,30}而且"), 1.0),
    (re.compile(r"一方面[^。！？]{0,30}另一方面"), 1.0),
    (re.compile(r"如果[^。！？]{0,30}那么"), 0.8),
    (re.compile(r"因为[^。！？]{0,30}所以"), 0.8),
]


def detect_connector_density(text: str) -> Dict:
    name = "连接词密度"
    if not _safe_text(text):
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "空文本"}

    sentences = split_sentences(text)
    words = re.findall(r"[\u4e00-\u9fff]+", text)
    connector_count = sum(1 for w in words if w in CONNECTORS)

    per_sentence = connector_count / max(1, len(sentences))
    # 成对结构（严谨逻辑链 = AI 特征）
    paired_score = sum(w for p, w in PAIRED_CONNECTORS if p.search(text))

    severity = 0.0
    if per_sentence > 0.6:
        severity = min(0.8, (per_sentence - 0.6) * 1.2)
    severity = max(severity, min(0.7, paired_score * 0.4))

    return {
        "name": name, "score": 1.0 - severity, "severity": severity,
        "issue_detected": severity > 0.4,
        "suggestion": f"连接词/句 {per_sentence:.2f} 成对结构×{paired_score:.1f}",
        "detail": {"per_sentence": round(per_sentence, 2), "paired": paired_score},
    }


# ============================================================
# 检测6: 情感缺失（v2: 基础词扩充 + 否定处理）
# ============================================================

EMOTION_WORDS = [
    "开心", "快乐", "高兴", "难过", "悲伤", "伤心", "委屈", "生气", "愤怒",
    "焦虑", "担心", "害怕", "紧张", "失望", "崩溃", "无奈", "郁闷", "纠结",
    "感动", "惊喜", "幸福", "烦", "累", "哭", "笑", "喜欢", "讨厌", "爱",
    "羡慕", "嫉妒", "后悔", "愧疚", "尴尬", "自豪", "欣慰", "兴奋", "期待",
    "心累", "绝望", "孤独", "温暖", "踏实", "安心",
]

NEG_PREFIX = ("不", "没", "别", "无", "非")


def _is_negated(text: str, pos: int) -> bool:
    """情感词前 2 字符内是否有否定词"""
    pre = text[max(0, pos - 2):pos]
    return any(pre.endswith(n) for n in NEG_PREFIX)


def detect_emotion_absence(text: str) -> Dict:
    name = "情感表达"
    if not _safe_text(text):
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "空文本"}
    if len(text) < 20:
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "超短文本跳过"}

    positive = 0   # 有情感表达（含否定后的情绪："不开心"也是情绪！）
    negated = 0
    for w in EMOTION_WORDS:
        for m in re.finditer(re.escape(w), text):
            if _is_negated(text, m.start()):
                negated += 1
            else:
                positive += 1

    # 有情感词（无论正负）都算有情绪表达
    has_emotion = positive + negated > 0
    severity = 0.0 if has_emotion else 0.55

    return {
        "name": name, "score": 1.0 - severity, "severity": severity,
        "issue_detected": severity > 0.5,
        "suggestion": f"情感词 {positive} 个（否定式 {negated} 个）",
        "detail": {"positive": positive, "negated": negated},
    }


# ============================================================
# 检测7: 句末语气词（v2: 去标点取汉字 + 词集扩充）
# ============================================================

ENDING_PARTICLES = set("啊呀吧呢吗嘛哦哟唉哈嗯诶嘞哒唷啦咧哇呗么呀呐咯呵嘿")

def _last_hanzi(sentence: str) -> str:
    """从后向前找第一个汉字"""
    for ch in reversed(sentence):
        if "\u4e00" <= ch <= "\u9fff":
            return ch
    return ""


def detect_ending_variety(text: str) -> Dict:
    name = "句末语气词"
    if not _safe_text(text):
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "空文本"}

    sentences = split_sentences(text)
    if len(sentences) < 3:
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "句数不足"}

    ends = [_last_hanzi(s) for s in sentences]
    particle_count = sum(1 for e in ends if e in ENDING_PARTICLES)
    ratio = particle_count / len(sentences)

    # 口语化文本语气词多；这里检测"完全没有"（书面 AI 腔）
    severity = 0.0 if ratio > 0.05 else 0.5
    return {
        "name": name, "score": 1.0 - severity, "severity": severity,
        "issue_detected": severity > 0.4,
        "suggestion": f"句末语气词 {particle_count}/{len(sentences)} 句 ({ratio:.0%})",
        "detail": {"count": particle_count, "ratio": round(ratio, 2)},
    }


# ============================================================
# 检测8: 回答回避性（v3 新增：不表态/转折推诿 = AI 客服腔）
# ============================================================

EVASIVE_PHRASES = [
    "好好考虑", "考虑一下", "得看你自己", "看你自己", "怎么说呢", "不好说",
    "看情况", "再说吧", "不一定", "说不准", "看你怎么想", "取决于",
    "得看", "要看你", "看你怎么", "因人而异", "这事可得", "这事儿可得",
    "这个问题挺", "挺棘手", "好好想想", "想想办法", "看情况而定",
]


def detect_evasiveness(text: str) -> Dict:
    name = "回答回避性"
    if not _safe_text(text) or len(text) < 10:
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "文本过短跳过"}
    sentences = split_sentences(text)
    evasive_hits = [p for p in EVASIVE_PHRASES if p in text]
    # 转折开头句（不过/但是/可是 开头 = 回避式推诿）
    pivot_starts = sum(1 for s in sentences if re.match(r"^(不过|但是|可是|不过呢|但是呢)", s))
    pivot_ratio = pivot_starts / max(1, len(sentences))
    severity = 0.0
    if evasive_hits:
        severity = min(1.0, 0.35 + 0.18 * len(evasive_hits))
    if pivot_ratio > 0.3:
        severity = max(severity, min(1.0, pivot_ratio * 1.4))
    return {
        "name": name, "score": 1.0 - severity, "severity": severity,
        "issue_detected": severity > 0.4,
        "suggestion": f"回避短语×{len(evasive_hits)} 转折开头句占比{pivot_ratio:.0%}",
        "detail": {"evasive": evasive_hits[:3], "pivot_ratio": round(pivot_ratio, 2)},
    }


# ============================================================
# 检测9: 叙事细节（v3 新增：真人孩子讲具体经历，AI 泛泛而谈）
# ============================================================

NARRATIVE_ANCHORS = [
    "上次", "那天", "有一次", "有一回", "我妈", "我爸", "我同桌", "我们班",
    "小时候", "记得", "以前", "那次", "上个月", "我朋友", "我同学", "我哥",
    "我姐", "我弟", "我妹", "有一次", "那天晚上", "三年级", "幼儿园",
]


def detect_narrative_detail(text: str) -> Dict:
    name = "叙事细节"
    if not _safe_text(text) or len(text) < 20:
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "超短文本跳过"}
    hits = [a for a in NARRATIVE_ANCHORS if a in text]
    severity = 0.0 if hits else 0.5
    return {
        "name": name, "score": 1.0 - severity, "severity": severity,
        "issue_detected": severity > 0.4,
        "suggestion": f"叙事锚点×{len(hits)}: {hits[:3]}" if hits else "无具体经历叙述",
        "detail": {"anchors": hits[:5]},
    }


# ============================================================
# 检测10: 表态强度（v3 新增：孩子回答有明确立场，AI 圆滑中立）
# ============================================================

DECISIVE_PHRASES = [
    "我就", "我要", "不行", "烦死了", "讨厌", "直接", "必须", "打死",
    "再也不", "懒得", "我才", "绝不", "干脆", "不想", "别想", "别扯",
    "别闹", "别烦", "别再说", "别抄", "别来", "别跟", "别理", "爱咋咋地",
    "管他", "管它", "随便", "懒得理", "不理", "跟他说", "告诉老师",
]


def detect_decisiveness(text: str) -> Dict:
    name = "表态强度"
    if not _safe_text(text) or len(text) < 10:
        return {"name": name, "score": 1.0, "severity": 0.0, "issue_detected": False, "suggestion": "文本过短跳过"}
    hits = [p for p in DECISIVE_PHRASES if p in text]
    severity = 0.0 if hits else 0.45
    return {
        "name": name, "score": 1.0 - severity, "severity": severity,
        "issue_detected": severity > 0.4,
        "suggestion": f"表态短语×{len(hits)}: {hits[:3]}" if hits else "无明确表态",
        "detail": {"decisive": hits[:5]},
    }


# ============================================================
# 汇总
# ============================================================

def run_all_checks(text: str) -> List[Dict]:
    """运行全部 10 项检测"""
    return [
        detect_sentence_rhythm(text),
        detect_ai_vocabulary(text),
        detect_template_structure(text),
        detect_personal_absence(text),
        detect_connector_density(text),
        detect_emotion_absence(text),
        detect_ending_variety(text),
        detect_evasiveness(text),
        detect_narrative_detail(text),
        detect_decisiveness(text),
    ]


def summarize_report(checks: List[Dict]) -> Dict:
    """综合报告（v2: 基于 severity 统一计算）"""
    if not checks:
        return {"composite": 1.0, "issues": [], "score": 1.0}

    severities = [c.get("severity", 0.0) for c in checks]
    # composite: 1 - 加权平均严重度（高严重度项加权）
    avg_sev = sum(s ** 2 for s in severities) / max(1, len(checks))
    composite = max(0.0, 1.0 - avg_sev * 2)

    issues = [c["name"] for c in checks if c.get("issue_detected")]
    return {"composite": round(composite, 3), "issues": issues, "score": round(composite, 3),
            "severities": severities}
