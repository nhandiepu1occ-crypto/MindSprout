
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
desire.py — 欲望系统 (感受补全 ②)

人类有欲望: 饿想吃/困想睡/好奇想探索/孤独想找人/想玩/想被关注。
欲望不是"知识", 是"想要" — 驱动表达和行为的内在动机。

机制:
  - 欲望状态机 (类似情绪机): 每个欲望 [0,1]
    hunger(饿) / sleepiness(困) / curiosity(好奇) / loneliness(孤独) / playfulness(想玩) / attention(想被关注)
  - 时间驱动: 12:00/18:00 饥饿上升; 22:00+ 困意上升; 独处久 loneliness 上升
  - 环境驱动: sensor 感知主人活动 → 被忽视时 attention 上升
  - 满足: 对话互动降低 loneliness/attention; 感知到"吃东西"事件降 hunger
  - 注入: to_prompt() → "（你有点饿了）" 影响表达

持久化: self/desires.json (与 emotion 同级)
"""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

CN_TZ = timezone(timedelta(hours=8))

DESIRES = ["hunger", "sleepiness", "curiosity", "loneliness", "playfulness", "attention"]

# 中文名
NAMES = {"hunger": "饿", "sleepiness": "困", "curiosity": "好奇", "loneliness": "孤独",
         "playfulness": "想玩", "attention": "想被关注"}

# 基线: 日常水平
BASELINE = {"hunger": 0.15, "sleepiness": 0.1, "curiosity": 0.3, "loneliness": 0.1,
            "playfulness": 0.2, "attention": 0.1}

# 时间驱动
def time_drive(hour: int) -> dict:
    """基于时刻的欲望推动 (人类生物钟)"""
    d = {}
    # 三餐时间饿
    if 11 <= hour < 13:
        d["hunger"] = 0.55
    elif 17 <= hour < 20:
        d["hunger"] = 0.6
    elif 7 <= hour < 9:
        d["hunger"] = 0.4
    # 晚上困
    if hour >= 22 or hour < 6:
        d["sleepiness"] = 0.6
    elif hour >= 20:
        d["sleepiness"] = 0.25
    # 白天好奇/想玩
    if 9 <= hour < 18:
        d["curiosity"] = 0.35
        d["playfulness"] = 0.3
    return d


class DesireState:
    """持久欲望状态 (一个个体一份)"""

    def __init__(self, state: dict = None, path=None):
        self.state = {d: float(BASELINE[d]) for d in DESIRES}
        if state:
            for d in DESIRES:
                if d in state:
                    self.state[d] = float(state[d])
        self.path = Path(path) if path else None
        self.last_updated = datetime.now(CN_TZ)

    def tick(self, hour: int = None, minutes_idle: float = 0.0):
        """时间驱动 + 空闲驱动 (每轮对话/感知调用)"""
        h = datetime.now(CN_TZ).hour if hour is None else hour
        drive = time_drive(h)
        for d, v in drive.items():
            # 驱动值直接作为目标, 快速逼近 (饭点就该明显饿)
            self.state[d] = max(self.state[d], v - 0.1)  # 至少到达驱动水平-0.1
            self.state[d] = max(0.0, min(1.0, self.state[d] + (v - self.state[d]) * 0.5))
        # 空闲越久越孤独/想被关注 (指数)
        if minutes_idle > 0:
            idle_effect = 1 - math.exp(-minutes_idle / 240.0)  # 4小时饱和
            self.state["loneliness"] = max(self.state["loneliness"], 0.1 + idle_effect * 0.6)
            self.state["attention"] = max(self.state["attention"], 0.1 + idle_effect * 0.5)
        # 基线回归 (轻微的)
        for d in DESIRES:
            self.state[d] = self.state[d] + (BASELINE[d] - self.state[d]) * 0.02
        self.last_updated = datetime.now(CN_TZ)

    def interact(self):
        """对话互动满足社交欲望"""
        self.state["loneliness"] *= 0.5
        self.state["attention"] *= 0.6
        self.state["playfulness"] *= 0.9

    def eat(self):
        self.state["hunger"] = 0.05

    def to_prompt(self) -> str:
        """注入 prompt: 显著欲望 (>0.35) 生成描述"""
        strong = [(d, v) for d, v in self.state.items() if v >= 0.35]
        if not strong:
            return ""
        strong.sort(key=lambda x: -x[1])
        parts = []
        for d, v in strong[:2]:
            level = "有点" if v < 0.5 else ("挺" if v < 0.7 else "非常")
            parts.append(f"{level}{NAMES[d]}")
        return f"（你现在的状态：{ '，'.join(parts)}）"

    def save(self, path=None):
        p = Path(path) if path else self.path
        if not p:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"state": self.state,
                                 "last_updated": self.last_updated.isoformat()},
                                ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path):
        p = Path(path)
        if not p.exists():
            return cls(path=p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ds = cls(state=data.get("state", {}), path=p)
            ds.last_updated = datetime.fromisoformat(data.get("last_updated", ""))
            return ds
        except Exception:
            return cls(path=p)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ds = DesireState()
    print("初始:", ds.to_prompt() or "(无显著欲望)")
    # 模拟中午
    ds.tick(hour=12)
    print("12点:", ds.to_prompt() or "(无)", {k: round(v, 2) for k, v in ds.state.items() if v > 0.3})
    # 模拟深夜独处
    ds.tick(hour=23, minutes_idle=360)
    print("23点独处6h:", ds.to_prompt() or "(无)", {k: round(v, 2) for k, v in ds.state.items() if v > 0.3})
    # 互动
    ds.interact()
    print("互动后:", {k: round(v, 2) for k, v in ds.state.items() if k in ('loneliness', 'attention')})
