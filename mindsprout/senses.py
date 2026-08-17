
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
senses.py — 感官合成器 (楚门 v2.0: 她大脑的感知皮层)
把 场景+人物+情绪 合成"此刻世界"的感官流 (颜色⊕气味⊕声音⊕温度)
像人走进教室: 粉笔灰味+日光灯嗡鸣+纸张味+黑板绿

耦合:
- engine: 每次生成前合成感官流 → system 注入 (持续感知, 非一次性标签)
- persona: 人物底色叠加
- emotion: 情绪色调叠加
- memory_bank: store 时自动打感官标签 (回忆有气味)
"""
import json
import time
from pathlib import Path
from datetime import datetime

STATE_FILE = BASE / "state" / "senses_state.json"

# ============ 场景感官库 ============
SCENES = {
    "classroom": {
        "name": "教室", "color": "白加粉笔灰",
        "smells": ["粉笔灰", "纸张", "橡皮擦"],
        "sounds": ["日光灯嗡鸣", "笔尖沙沙", "同学小声说话"],
        "feel": "课桌凉凉的，窗外有蝉鸣",
        "temperature": "有点闷",
    },
    "canteen": {
        "name": "食堂", "color": "蒸汽白加暖黄",
        "smells": ["番茄炒蛋", "米饭蒸汽", "辣条味"],
        "sounds": ["碗筷叮当", "嘈杂人声", "阿姨喊号"],
        "feel": "热腾腾的，地面有点滑",
        "temperature": "热",
    },
    "home": {
        "name": "家", "color": "暖黄灯光",
        "smells": ["饭菜香", "洗衣粉", "妈妈身上的味道"],
        "sounds": ["拖鞋声", "厨房炒菜声", "电视声"],
        "feel": "沙发软软的，安心",
        "temperature": "暖暖的",
    },
    "playground": {
        "name": "操场", "color": "亮白阳光加绿茵",
        "smells": ["汗味", "塑胶跑道", "青草"],
        "sounds": ["哨声", "球砸地声", "大家喊叫"],
        "feel": "太阳晒着后颈，风一吹很舒服",
        "temperature": "晒得热",
    },
    "homeway": {
        "name": "放学路上", "color": "橘红夕阳",
        "smells": ["路边小吃摊", "树叶", "汽车尾气"],
        "sounds": ["自行车铃", "小贩吆喝", "蝉鸣"],
        "feel": "书包有点沉，风是暖的",
        "temperature": "傍晚凉快些了",
    },
}

# ============ 情绪色调 ============
MOOD_TONES = {
    "joy": ("亮黄色", "心里轻飘飘的，想蹦两下"),
    "sad": ("灰蓝色", "胸口闷闷的，想叹气"),
    "anger": ("暗红色", "心里堵着一团火，手有点抖"),
    "anxiety": ("青灰色", "心里揪着，手心有点出汗"),
    "calm": ("淡蓝色", "心里平平静静的"),
    "sulky": ("暗橘色", "心里堵着气，不想理人"),
    "warmth": ("浅金色", "心里暖烘烘的"),
    "curiosity": ("淡绿色", "心里痒痒的，想弄清楚"),
    "neutral": ("透明色", "没什么特别的感觉"),
}

# ============ 合成器 ============
def _pick_scene(context_hint: str = "", text: str = "") -> str:
    """场景推断: 时间+话题线索 → 场景ID"""
    now = datetime.now()
    t = (text or "") + (context_hint or "")
    if any(k in t for k in ["教室", "上课", "老师", "作业", "听写", "课本"]):
        return "classroom"
    if any(k in t for k in ["食堂", "吃饭", "打饭", "番茄", "辣条"]):
        return "canteen"
    if any(k in t for k in ["操场", "体育", "跑步", "打球"]):
        return "playground"
    if any(k in t for k in ["回家", "放学", "路上", "到家"]):
        return "homeway"
    if any(k in t for k in ["家", "妈妈", "爸爸", "吃饭了", "睡觉"]):
        return "home"
    # 时间兜底
    h = now.hour
    if 7 <= h < 12:
        return "classroom"
    if 12 <= h < 14:
        return "canteen"
    if 14 <= h < 17:
        return "classroom"
    if 17 <= h < 20:
        return "homeway"
    return "home"

def sense_line(text: str = "", context_hint: str = "", person_id: str = None,
               emotion_dom: str = "neutral") -> str:
    """合成此刻感官流 → 注入行 (持续感知, 带人物底色叠加)"""
    scene_id = _pick_scene(context_hint, text)
    sc = SCENES.get(scene_id, SCENES["home"])
    parts = [f"（你此刻在{sc['name']}：闻到{sc['smells'][0]}，听到{sc['sounds'][0]}，{sc['feel']}，{sc['temperature']}，眼前是{sc['color']}的色调。"]
    # 人物底色叠加
    if person_id:
        try:
            from humanize_ai.persona import PERSONS
            p = PERSONS.get(person_id)
            if p:
                parts.append(f"他/她在你旁边，你心里浮起{p['color']}的感觉，像闻到{p['scent']}。")
        except Exception:
            pass
    # 情绪色调
    tone = MOOD_TONES.get(emotion_dom, MOOD_TONES["neutral"])
    parts.append(f"你此刻{tone[1]}。")
    parts.append("）")
    return "".join(parts)

def tag_senses(text: str) -> dict:
    """给记忆打感官标签 (store 时调用 → 回忆有气味)"""
    scene_id = _pick_scene("", text)
    sc = SCENES.get(scene_id, SCENES["home"])
    return {
        "scene": scene_id,
        "smell": sc["smells"][0],
        "sound": sc["sounds"][0],
        "color": sc["color"],
    }

if __name__ == "__main__":
    for t, ctx in [
        ("妈，我回来了", "家"),
        ("李老师：你作业呢", "教室"),
        ("食堂今天有番茄炒蛋", ""),
    ]:
        print("==", t)
        print(sense_line(t, ctx, person_id="mom" if "妈" in t else None))
