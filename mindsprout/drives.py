
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
drives.py — 驱力系统 (出厂能力 P0-1)
饿/渴/困/社交需求/好奇 — 随时间演化, 阈值触发主动性
驱动行为: 高驱力 → 主动表达需求 + 影响情绪 + 影响记忆检索
"""
import json
import time
from pathlib import Path

STATE_FILE = BASE / "state" / "drives.json"

DEFAULTS = {
    "hunger": 30.0,    # 0-100 (每4h涨25, 吃饭归零)
    "thirst": 20.0,
    "sleep": 20.0,     # 晚间涨, 睡觉归零
    "social": 40.0,    # 独处涨, 对话归零
    "curiosity": 50.0, # 遇到未知涨
    "last_tick": 0.0,
    "last_meal": 0.0,
    "last_sleep": 0.0,
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

def tick(now=None):
    """时间演化 (每次生成前调用)"""
    s = load()
    now = now or time.time()
    if s.get("last_tick", 0) <= 0:
        dt_h = 0  # 首次运行不演化 (防初始化爆炸)
    else:
        dt_h = (now - s["last_tick"]) / 3600.0
    if dt_h < 0:
        dt_h = 0
    # 饿/渴/困 随时间涨
    s["hunger"] = min(100, s["hunger"] + 18 * dt_h)
    s["thirst"] = min(100, s["thirst"] + 12 * dt_h)
    import datetime
    h = datetime.datetime.now().hour
    if 21 <= h or h < 6:
        s["sleep"] = min(100, s["sleep"] + 15 * dt_h)
    # 社交需求: 独处涨 (由对话交互清零)
    s["social"] = min(100, s["social"] + 8 * dt_h)
    s["last_tick"] = now
    save(s)
    return s

def satisfy(drive, amount=100):
    """满足驱力 (吃饭/睡觉/聊天后)"""
    s = load()
    s[drive] = max(0, s.get(drive, 0) - amount)
    if drive == "hunger":
        s["last_meal"] = time.time()
    if drive == "sleep":
        s["last_sleep"] = time.time()
    save(s)
    return s

def interact_social(weight=1.0):
    """对话=社交满足"""
    return satisfy("social", 25 * weight)

def top_drive(s=None):
    """当前最强驱力"""
    s = s or load()
    d = {k: s.get(k, 0) for k in ("hunger", "thirst", "sleep", "social", "curiosity")}
    return max(d.items(), key=lambda x: x[1])

def drive_line(s=None):
    """注入: 身体需求感知 (只取最强 2 个, 防爆炸)"""
    s = s or load()
    ranked = sorted([("hunger", s["hunger"]), ("thirst", s["thirst"]),
                     ("sleep", s["sleep"]), ("social", s["social"]),
                     ("curiosity", s["curiosity"])], key=lambda x: -x[1])
    lines = {
        "hunger": [(65, "肚子有点饿了，咕咕叫"), (85, "饿得前胸贴后背，脑子里全是吃的")],
        "thirst": [(70, "嗓子有点干，想喝水")],
        "sleep": [(70, "有点困了，眼皮打架"), (90, "困得不行了，哈欠连天")],
        "social": [(70, "有点想找人说说话")],
        "curiosity": [(80, "心里有件事特别想知道答案")],
    }
    parts = []
    for drive, level in ranked[:2]:
        opts = sorted(lines.get(drive, []), key=lambda x: -x[0])  # 高阈值优先
        for th, txt in opts:
            if level >= th:
                parts.append(txt)
                break
    if parts:
        return "（你此刻：" + "；".join(parts) + "。）"
    return ""

if __name__ == "__main__":
    s = tick()
    print("驱力:", {k: round(v) for k, v in s.items() if k in ("hunger", "thirst", "sleep", "social", "curiosity")})
    print("最强:", top_drive(s))
    print("注入:", drive_line(s))
