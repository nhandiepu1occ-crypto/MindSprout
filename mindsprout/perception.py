
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
perception.py — 感知引擎 (感知系统 ② 核心)

从场景文本中实时解析感知触发点 → 路由器官 → 生成第一人称感知输入。
场景 → 感知 → 子AI反应, 训练"感知→体验→表达"因果链。

工作流:
  1. 解析: 找动作词(吃/摸/闻/听/看) + 对象词(匹配感知知识库)
  2. 路由: 动作决定参与器官 (吃→嘴+手+鼻, 摸→手, 闻→鼻, 听→耳, 看→眼)
  3. 合成: 器官感知文本 + 温度/味道模块 → 第一人称感知段落
  4. 注入: 感知段落拼进场景 → 子AI生成时"带着感觉"

用法:
  from perception import PerceptionEngine
  pe = PerceptionEngine()
  enhanced = pe.enhance(scene_text)   # 返回增强后的场景
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
KB_PATH = BASE / "phase1" / "self" / "sense_kb.json"

# 动作 → 器官 (正则 → [器官...])
ACTION_ORGANS = [
    (re.compile(r"吃|咬|啃|舔|喝|嚼|尝|咽"), ["mouth", "nose", "hand"]),
    (re.compile(r"摸|碰|抓|捏|抱|拿|捧|握|揉|挠|掂"), ["hand"]),
    (re.compile(r"踩|踢|踏|跳进|泡|蹦|站|走"), ["foot", "hand"]),  # 脚部触觉
    (re.compile(r"闻|嗅"), ["nose"]),
    (re.compile(r"听|听见|听到|叮当|咕咚|咔嚓"), ["ear"]),
    (re.compile(r"看|看见|看到|望着|盯着|瞄|瞧"), ["eye"]),
    (re.compile(r"晒|冻|热|冷|暖和|凉快|烫"), ["skin"]),  # 全身温度
    (re.compile(r"写|画|抄|翻|读|翻开|合上"), ["hand", "eye"]),  # 书写/阅读类
]

# 器官标签
ORGAN_NAMES = {"mouth": "嘴", "hand": "手", "foot": "脚", "nose": "鼻子",
               "ear": "耳朵", "eye": "眼睛", "skin": "皮肤"}


class PerceptionEngine:
    def __init__(self, kb_path=None):
        self.kb = {}
        p = Path(kb_path) if kb_path else KB_PATH
        if p.exists():
            self.kb = json.loads(p.read_text(encoding="utf-8"))

    def parse(self, scene: str) -> list:
        """解析场景 → [(action_match, organ_list, object_name, object_sense)]"""
        hits = []
        for pattern, organs in ACTION_ORGANS:
            for m in pattern.finditer(scene):
                window = scene[max(0, m.start() - 8):m.end() + 20]
                matched = False
                for obj, sense in self.kb.items():
                    if obj in window:
                        hits.append((m.group(0), organs, obj, sense))
                        matched = True
                        break
                    # 模糊: 窗口里的 2-4 字词出现在对象名里 ("雪糕" in "草莓雪糕")
                    for kw in re.findall(r"[\u4e00-\u9fa5]{2,4}", window):
                        if kw in obj and obj not in scene:
                            hits.append((m.group(0), organs, obj, sense))
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break  # 每个动作只配一个对象
        return hits

    def synthesize(self, hits: list) -> str:
        """器官感知 → 第一人称感知文本"""
        parts = []
        for action, organs, obj, sense in hits:
            sense_parts = []
            for organ in organs:
                txt = sense.get(organ, "")
                if txt:
                    sense_parts.append(f"{ORGAN_NAMES[organ]}({txt})")
            if sense_parts:
                parts.append(f"【感知·{obj}】{action}它时: " + "；".join(sense_parts))
        return "\n".join(parts)

    def enhance(self, scene: str, source: str = "") -> tuple:
        """增强场景: 返回 (enhanced_scene, perception_text)
        自动记录感知日志 (监测器)"""
        hits = self.parse(scene)
        # 监测: 记录感知事件 (命中/未命中都记)
        try:
            from humanize_ai.perception_monitor import log_perception
            log_perception(scene, hits, bool(hits), source=source)
        except Exception:
            pass
        if not hits:
            return scene, ""
        ptext = self.synthesize(hits)
        enhanced = scene + f"\n（你真实感受到的：{ptext}）"
        return enhanced, ptext


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    pe = PerceptionEngine()
    tests = [
        "放学路上，妈妈给你买了一个草莓雪糕，你撕开包装咬了一口。",
        "冬天，爸爸把刚晒过的被子抱过来，你钻进被窝。",
        "你把橘猫小花抱起来，它在你怀里蹭了蹭。",
        "下雨了，你踩进水坑里，水花溅起来。",
    ]
    for t in tests:
        enhanced, p = pe.enhance(t)
        print(f"\n场景: {t}")
        print(f"感知: {p or '(未触发)'}")
