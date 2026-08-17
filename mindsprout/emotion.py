
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
emotion.py — 情绪状态机 + 共情耦合（DNA_DESIGN v1）

1) 情绪状态机: 持久 mood 向量 {joy, sadness, anger, fear, disgust, surprise} ∈ [0,1]
   - 事件影响 apply_impact(): 负面情绪按 DNA.neg_gain 放大
   - 稳态回归 decay(): state → baseline 指数回归（DNA.recovery_rate）
   - 注入: to_prompt() → "（你现在的心情：有点难过）"
2) 共情耦合（出厂能力，架构级先验）:
   - perceive(text): 词典检测对方情绪
   - empathize(perceived): Δmood = α × (对方情绪 − 当前情绪)，负性维度 × negativity_bias
   - 机制生效不需要训练；训练只需要让"情绪被拉动后的语言表达"跟上
"""
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

CN_TZ = timezone(timedelta(hours=8))

EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "surprise"]

# ---- 中文情绪词典（v1 糙版：词 + 强度 1-3；否定词/反讽未处理，v2 换小分类器）----
LEXICON = {
    "joy": {
        "哈哈": 2, "哈哈哈哈": 3, "开心": 2, "高兴": 2, "太好了": 2, "太棒": 2, "嘻嘻": 1,
        "笑死": 2, "乐死": 2, "好玩": 1, "喜欢": 1, "超开心": 3, "激动": 2, "幸福": 2,
        "棒": 1, "耶": 1, "嘻嘻嘻": 2, "666": 1, "nice": 1, "哈哈笑": 2,
    },
    "sadness": {
        "难过": 2, "伤心": 2, "想哭": 2, "哭了": 2, "哭": 2, "难受": 2, "委屈": 2,
        "失落": 2, "郁闷": 1, "沮丧": 2, "心碎": 3, "emo": 2, "emo了": 2, "呜呜": 2,
        "好累": 1, "烦死": 1, "不开心": 2, "眼泪": 2, "崩了": 1, "绝望": 3,
    },
    "anger": {
        "气死": 3, "生气": 2, "气人": 2, "火大": 2, "可恶": 2, "烦死了": 2, "讨厌": 2,
        "过分": 2, "气炸": 3, "气死了": 3, "滚": 2, "欠揍": 2, "恼火": 2, "抓狂": 2,
        "气死我了": 3, "太气人": 3, "好气": 2, "火冒三丈": 3,
    },
    "fear": {
        "害怕": 2, "好怕": 2, "吓死": 3, "吓人": 2, "恐怖": 2, "不敢": 1, "紧张": 2,
        "怕怕": 2, "胆战心惊": 3, "瑟瑟发抖": 2, "慌": 2, "吓我一跳": 2, "慌死了": 2,
    },
    "disgust": {
        "恶心": 2, "想吐": 2, "反胃": 2, "脏": 1, "恶心死": 3, "作呕": 2, "嫌弃": 1, "呕": 1,
    },
    "surprise": {
        "哇": 1, "天哪": 1, "居然": 1, "没想到": 1, "惊讶": 1, "震惊": 2, "什么？": 1,
        "卧槽": 1, "天啊": 1, "不可思议": 2, "我的天": 1,
    },
}


def perceive(text: str) -> dict:
    """词典检测对方情绪 → {emotion: 强度[0,1]}（仅返回命中的维度）"""
    if not text:
        return {}
    out = {}
    for emo, words in LEXICON.items():
        score = 0
        for w, wt in words.items():
            if w in text:
                score += wt
        if score > 0:
            out[emo] = round(min(1.0, score / 3.0), 3)  # 3分=饱和
    return out


class EmotionState:
    """持久情绪状态（一个个体一份，存 state/<id>/emotion.json）"""

    def __init__(self, dna, state: dict = None, path=None):
        self.dna = dna
        self.baseline = dna.baseline()
        self.state = {e: float(state.get(e, self.baseline[e])) for e in EMOTIONS} if state else dict(self.baseline)
        self.path = Path(path) if path else None
        self.last_updated = datetime.now(CN_TZ)

    # ---- 事件影响（后果链/场景 impact 的入口）----
    def apply_impact(self, impact: dict):
        """impact: {sadness: 0.05, anger: 0.02} — 负面情绪的正增量按 DNA 放大（越糟越敏感）"""
        gain = self.dna.neg_gain()
        negative = {"sadness", "anger", "fear", "disgust"}
        for e in EMOTIONS:
            d = impact.get(e, 0.0)
            if e in negative and d > 0:
                d *= gain
            self.state[e] = round(max(0.0, min(1.0, self.state[e] + d)), 4)

    # ---- 稳态回归（时间衰减，指数）----
    def decay(self):
        minutes = max(0.0, (datetime.now(CN_TZ) - self.last_updated).total_seconds() / 60.0)
        if minutes <= 0:
            return
        rate = self.dna.recovery_rate()
        keep = math.exp(-rate * minutes)
        for e in EMOTIONS:
            self.state[e] = round(self.baseline[e] + (self.state[e] - self.baseline[e]) * keep, 4)
        self.last_updated = datetime.now(CN_TZ)

    # ---- 共情耦合（出厂能力）----
    def empathize(self, perceived: dict):
        """Δmood = α × (对方情绪 − 当前情绪)；负性维度 × negativity_bias。返回是否发生了共情。"""
        if not perceived:
            return False
        alpha = self.dna.empathy_alpha()
        neg_bias = self.dna.empathy_neg_bias()
        negative = {"sadness", "anger", "fear", "disgust"}
        changed = False
        for e, pv in perceived.items():
            if e not in self.state:
                continue
            delta = alpha * (pv - self.state[e])
            if e in negative:
                delta *= neg_bias
            if abs(delta) > 1e-4:
                self.state[e] = round(max(0.0, min(1.0, self.state[e] + delta)), 4)
                changed = True
        return changed

    # ---- 每轮对话入口：衰减 + 共情 ----
    def tick(self, text: str) -> dict:
        self.decay()
        perceived = perceive(text)
        if perceived:
            self.empathize(perceived)
        return perceived

    # ---- 注入 prompt ----
    def to_prompt(self) -> str:
        """取偏离基线最大的情绪生成心情描述；偏离<0.12 不注入（正常状态不写心情）"""
        deltas = {e: self.state[e] - self.baseline[e] for e in EMOTIONS}
        emo, d = max(deltas.items(), key=lambda kv: kv[1])
        if d < 0.12:
            return ""
        level = d
        if level < 0.3:
            word = "有点"
        elif level < 0.6:
            word = "挺"
        else:
            word = "非常"
        names = {"joy": "开心", "sadness": "难过", "anger": "生气", "fear": "害怕",
                 "disgust": "恶心", "surprise": "惊讶"}
        return f"（你现在的心情：{word}{names[emo]}）"

    def dominant(self) -> str:
        deltas = {e: self.state[e] - self.baseline[e] for e in EMOTIONS}
        return max(deltas, key=deltas.get)

    def save(self, path=None):
        p = Path(path) if path else self.path
        if not p:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state": self.state, "last_updated": self.last_updated.isoformat()}
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, dna, path):
        p = Path(path)
        if not p.exists():
            return cls(dna, path=p)
        data = json.loads(p.read_text(encoding="utf-8"))
        es = cls(dna, state=data.get("state", {}), path=p)
        try:
            es.last_updated = datetime.fromisoformat(data.get("last_updated", ""))
        except ValueError:
            pass
        return es


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from dna import birth_roll
    d = birth_roll(seed=7)
    es = EmotionState(d)
    for text in ["我今天被老师骂了，好难过", "哈哈哈今天考了第一名！", "你还好吗"]:
        p = es.tick(text)
        print(f"[{text}] → 感知{p} 心情线: '{es.to_prompt()}' 状态: {es.state}")
