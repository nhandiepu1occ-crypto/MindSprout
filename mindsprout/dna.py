
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
dna.py — 出生档案（DNA_DESIGN v1）

出厂设置 = 出生时随机掷骰子，写入后只读（AgeVault：过去不可变）。

结构:
  temperament: Rothbart 三因子 {surgency 外倾, negative_affect 负性情绪, effortful_control 努力控制}
  empathy:     {alpha 共情耦合强度, negativity_bias 负性偏向}
  emotion_baseline: 各基本情绪的稳态基线
  plasticity:  {base, age_decay, big_event_boost}（V13 塑性调度用）

随机 ≠ 等权：每个个体出厂就有不对称偏好（等权会收敛到均值人格）。
"""
import json
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone, timedelta

CN_TZ = timezone(timedelta(hours=8))

# 基本情绪（与 emotion.py 一致）
EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "surprise"]


@dataclass
class DNA:
    id: str
    born_at: str = ""
    temperament: dict = field(default_factory=dict)   # {surgency, negative_affect, effortful_control} ∈ [0,1]
    empathy: dict = field(default_factory=dict)       # {alpha, negativity_bias}
    emotion_baseline: dict = field(default_factory=dict)  # {joy: 0.5, sadness: 0.1, ...}
    plasticity: dict = field(default_factory=dict)    # {base, age_decay, big_event_boost}

    # ---- 性格底色 → 口语化描述（注入 system prompt，一行）----
    def temperament_line(self) -> str:
        t = self.temperament
        s = t.get("surgency", 0.5)             # 高=活泼外向, 低=文静慢热
        n = t.get("negative_affect", 0.5)      # 高=敏感易哭易怒, 低=心大乐观
        e = t.get("effortful_control", 0.5)    # 高=沉得住气, 低=冲动
        parts = []
        parts.append("活泼外向" if s >= 0.6 else ("文静慢热" if s <= 0.4 else "不闹也不闷"))
        parts.append("敏感、情绪来得快" if n >= 0.6 else ("心大乐观" if n <= 0.4 else "情绪普通"))
        parts.append("挺沉得住气" if e >= 0.6 else ("有点冲动" if e <= 0.4 else "还算有分寸"))
        return "你天生的性格底色：" + "、".join(parts) + "。"

    # ---- 情绪更新调制：负性情绪高 → 负面事件放大、恢复更慢 ----
    def neg_gain(self) -> float:
        """负面情绪事件的放大系数（negative_affect=0.5 时为 1.0）"""
        n = self.temperament.get("negative_affect", 0.5)
        return round(0.7 + 0.6 * n, 3)

    def recovery_rate(self) -> float:
        """情绪恢复速率（分钟级指数衰减系数；负性情绪高 → 恢复慢）"""
        n = self.temperament.get("negative_affect", 0.5)
        return round(0.05 * (1.4 - 0.8 * n), 4)   # n=0.5 → 0.05/min

    def empathy_alpha(self) -> float:
        return float(self.empathy.get("alpha", 0.35))

    def empathy_neg_bias(self) -> float:
        return float(self.empathy.get("negativity_bias", 1.6))

    def baseline(self) -> dict:
        return dict(self.emotion_baseline)


def birth_roll(individual_id: str = "luoluo-001", seed: int = None) -> DNA:
    """出生掷骰子：随机采样气质/共情/基线。seed 可复现（测试用），None = 真随机。"""
    rng = random.Random(seed)

    def clip(v, lo=0.05, hi=0.95):
        return round(max(lo, min(hi, v)), 3)

    temperament = {
        "surgency": clip(rng.gauss(0.50, 0.15)),
        "negative_affect": clip(rng.gauss(0.45, 0.15)),
        "effortful_control": clip(rng.gauss(0.50, 0.15)),
    }
    empathy = {
        "alpha": clip(rng.gauss(0.35, 0.12), 0.02, 0.9),   # 共情强度，最低 0.02（不会为 0）
        "negativity_bias": clip(rng.gauss(1.6, 0.25), 1.0, 2.5),
    }
    emotion_baseline = {
        "joy": clip(rng.gauss(0.50, 0.10), 0.1, 0.9),
        "sadness": clip(rng.gauss(0.10, 0.06), 0.02, 0.5),
        "anger": clip(rng.gauss(0.10, 0.06), 0.02, 0.5),
        "fear": clip(rng.gauss(0.10, 0.06), 0.02, 0.5),
        "disgust": clip(rng.gauss(0.05, 0.03), 0.01, 0.3),
        "surprise": clip(rng.gauss(0.05, 0.03), 0.01, 0.3),
    }
    plasticity = {"base": 0.5, "age_decay": 0.03, "big_event_boost": 1.5}
    return DNA(
        id=individual_id,
        born_at=datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        temperament=temperament,
        empathy=empathy,
        emotion_baseline=emotion_baseline,
        plasticity=plasticity,
    )


def load_dna(path) -> DNA:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DNA(**data)


def save_dna(dna: DNA, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(dna), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    d = birth_roll(seed=42)
    print(d.temperament_line())
    print("共情:", d.empathy, " 负性增益:", d.neg_gain(), " 恢复率:", d.recovery_rate())
    print("基线:", d.emotion_baseline)
    print(save_dna(d, str(BASE)))
