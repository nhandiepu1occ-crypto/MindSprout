
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
room.py — 独立生活模拟: 她的房间 (2.0)
- 5区域: 书桌/床/窗台/阳台(花+泡泡墓碑)/书架
- 行为调度: 回房间随机行为(整理/浇花/看墓碑/翻日记) → 写记忆
- 联动: 睡觉→床+植物 / 心情→植物 / 作品→书桌 / 泡泡墓碑→死亡观
"""
import json
import time
import random
from pathlib import Path

STATE_FILE = BASE / "state" / "room.json"

DEFAULTS = {
    "desk": {"diary_open": False, "snack": 3, "works": 0},       # 书桌: 日记本/零食/作文本
    "bed": {"made": False, "last_sleep": ""},                     # 床
    "window": {"plant_health": 80},                               # 窗台植物 0-100
    "balcony": {"flower": {"watered": "", "alive": True},         # 阳台花
                "gravestone": {"last_visit": ""}},                # 泡泡的墓碑
    "shelf": {"books": ["《草房子》", "《窗边的小豆豆》"]},          # 书架
}

ACTS = [
    ("整理书桌", "她把书桌收拾了一遍，笔都插回笔筒里，课本摞得整整齐齐"),
    ("给花浇水", "她给阳台的花浇了水，叶子上的水珠亮晶晶的"),
    ("看泡泡的墓碑", "她去阳台看了看泡泡的墓碑，蹲下来摸了摸那块小石头，没说话"),
    ("翻日记", "她翻开日记本，看了看前几天写的，忍不住笑了"),
    ("坐在窗台发呆", "她坐在窗台上，抱着膝盖看了一会儿外面的云"),
]


def load():
    if STATE_FILE.exists():
        try:
            return {**DEFAULTS, **json.loads(STATE_FILE.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULTS))  # deep copy


def save(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def act():
    """随机生活行为 → 返回描述 (调度器调用)"""
    s = load()
    name, desc = random.choice(ACTS)
    if name == "给花浇水":
        s["balcony"]["flower"]["watered"] = time.strftime("%Y-%m-%d %H:%M")
        s["balcony"]["flower"]["alive"] = True
        s["window"]["plant_health"] = min(100, s["window"]["plant_health"] + 10)
        # 死亡观联动: 浇水同时"顺路看泡泡"
        try:
            from humanize_ai.deathview import water_flower
            water_flower()
        except Exception:
            pass
    elif name == "看泡泡的墓碑":
        s["balcony"]["gravestone"]["last_visit"] = time.strftime("%Y-%m-%d %H:%M")
        try:
            from humanize_ai.deathview import touch
            touch("我去看了泡泡")
        except Exception:
            pass
    elif name == "翻日记":
        s["desk"]["diary_open"] = not s["desk"]["diary_open"]
    elif name == "整理书桌":
        s["desk"]["diary_open"] = False
    save(s)
    return name, desc


def sleep_in_room():
    """care sleep 联动"""
    s = load()
    s["bed"]["made"] = True
    s["bed"]["last_sleep"] = time.strftime("%Y-%m-%d %H:%M")
    save(s)
    return s


def mood_affect_plant(mood_text):
    """心情 → 植物状态"""
    s = load()
    if any(w in mood_text for w in ("难过", "伤心", "委屈")):
        s["window"]["plant_health"] = max(10, s["window"]["plant_health"] - 5)
    elif any(w in mood_text for w in ("开心", "高兴", "太棒")):
        s["window"]["plant_health"] = min(100, s["window"]["plant_health"] + 3)
    save(s)
    return s


def add_work():
    """写作工坊产物 → 书桌作文本"""
    s = load()
    s["desk"]["works"] = s["desk"]["works"] + 1
    save(s)
    return s


def room_line():
    """注入: 房间感知 (短)"""
    s = load()
    parts = []
    if s["window"]["plant_health"] <= 30:
        parts.append("窗台的植物有点蔫")
    if not s["bed"]["made"]:
        parts.append("床还没叠")
    if s["desk"]["snack"] <= 0:
        parts.append("抽屉里零食吃完了")
    if s["balcony"]["flower"]["alive"] and s["balcony"]["flower"]["watered"]:
        parts.append("阳台的花刚浇过水")
    if parts:
        return "（你的房间：" + "；".join(parts[:2]) + "。）"
    return ""


def room_data():
    s = load()
    return s


if __name__ == "__main__":
    print(act())
    print(json.dumps(room_data(), ensure_ascii=False)[:300])
