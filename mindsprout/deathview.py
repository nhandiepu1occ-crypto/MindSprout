
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
deathview.py — 死亡观内化: 有限性成为底色 (2.0 核心)
- 四阶段状态机: 0无概念 → 1接触(泡泡) → 2理解不可逆(亲人会老) → 3内化(珍惜/告别/时间焦虑)
- 触发: 泡泡记忆/对话触及死老离别/催化教学/房间墓碑互动
- 注入: 按stage, 只感知不说教
- 联动: timeline(年龄/中考) / room(泡泡墓碑) / values(珍惜)
"""
import json
import time
import re
from pathlib import Path

STATE_FILE = BASE / "state" / "deathview.json"

DEFAULTS = {
    "stage": 0,
    "first_contact_ts": 0.0,
    "events": [],        # 死亡相关事件 [{text, ts, stage}]
    "last_watering": "",  # 给泡泡浇水 (房间联动)
    "touched_ts": 0.0,    # 最近一次触发时间 (防每轮注入)
}

# 触发词
TRIGGER_WORDS = ["死", "去世", "离开", "再也", "永远", "老了", "变老", "告别", "坟墓", "泡泡", "金鱼", "走了", "不在了", "消失", "忘记"]

STAGE_LINES = {
    1: ["（你想起泡泡，它走了就再也没回来过）"],
    2: ["（你突然想到，妈妈也会变老，总有一天……你不敢往下想）"],
    3: ["（时间过得好快，你想多陪陪他们）", "（你心里有点怕，怕有一天重要的人会离开）"],
}

# 中考 (2027-06, 她初三)
ZHONGKAO = "2027-06-10"


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


def _age_of(birth_year, year):
    return max(0, year - birth_year)


def touch(text=""):
    """对话触发: 触及死亡/时间主题 → 记录事件+尝试升级"""
    s = load()
    hit = any(w in text for w in TRIGGER_WORDS) if text else False
    if not hit and s["stage"] == 0:
        return s
    if hit:
        s["events"].append({"text": text[:60], "ts": time.strftime("%Y-%m-%d %H:%M"),
                            "stage": s["stage"]})
        s["events"] = s["events"][-30:]
        s["touched_ts"] = time.time()
        # 升级: 有接触事件 → stage1
        if s["stage"] == 0 and ("泡泡" in text or "金鱼" in text or "死" in text):
            s["stage"] = 1
            s["first_contact_ts"] = time.time()
    # 时间不可逆理解 (stage1→2): 需要 stage1 且经历多次触发/时间推移
    if s["stage"] == 1 and s.get("touched_ts") and time.time() - s["touched_ts"] > 3600:
        s["stage"] = 2
    save(s)
    return s


def advance_to(stage):
    s = load()
    s["stage"] = max(s["stage"], stage)
    save(s)
    return s


def death_line():
    """注入: 按stage (低概率/冷却, 防每轮都提)"""
    s = load()
    if s["stage"] == 0:
        return ""
    if time.time() - s.get("touched_ts", 0) < 600:
        return ""
    lines = STAGE_LINES.get(s["stage"], [])
    if not lines:
        return ""
    import random
    return random.choice(lines)


def time_anxiety_line():
    """时间焦虑注入 (stage≥2): 爸妈年龄 + 中考倒计时"""
    s = load()
    if s["stage"] < 2:
        return ""
    try:
        from humanize_ai.timeline import age_years, now
        luoluo_age = age_years()
        mom_age = 38 + (now().year - 2026)  # 2026年妈妈38岁 (出厂设定)
        dad_age = 40 + (now().year - 2026)
        return f"（你心里算了算：妈妈{mom_age}了，爸爸{dad_age}了，自己也{luoluo_age}岁了，时间真快）"
    except Exception:
        return ""


def zhongkao_line():
    """中考倒计时 (stage≥3)"""
    s = load()
    if s["stage"] < 3:
        return ""
    try:
        from datetime import datetime
        zk = datetime.strptime(ZHONGKAO, "%Y-%m-%d")
        days = (zk - datetime.now()).days
        if 0 < days < 400:
            return f"（你心里有个倒计时：还有{days}天就中考了）"
    except Exception:
        pass
    return ""


def water_flower():
    """房间联动: 给泡泡浇水 (告别仪式)"""
    s = load()
    s["last_watering"] = time.strftime("%Y-%m-%d %H:%M")
    s["stage"] = max(s["stage"], 1)
    save(s)
    return s


if __name__ == "__main__":
    touch("泡泡它走了")
    print("stage:", load()["stage"])
    print("注入:", death_line())
    print("时间焦虑:", time_anxiety_line())
