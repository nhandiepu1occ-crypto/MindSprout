
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
mindstream.py — 内心流引擎 (V13 核心, MINDSTREAM_DESIGN v4)

设计原则 (主人定稿):
  ① 内心完全自由 — 零约束, 不被打分, 输出后我们观察/修补框架
  ② 内心=记忆的一种 — imagination标签, 能影响情绪, 容易忘记(衰减快)
  ③ 子AI"无意识" — 不知道被监督内心 (无监督生成)
  ④ 冥想空间 — 情绪后引导 + 每日固定独处
  ⑤ 内心日记=记录器 — 睡前整理, 供观察/修补/功能补充
  ⑥ 心流大小可学习 — 社会反馈驱动 (被询问/约束→收敛)

v4.1 修复 (主人 code review 12条):
  1. Mind 导入失败不再静默 — 显式检查+警告, 情绪功能独立封装
  2. .env 读取 try/except; api_key 空时直接返回不发起请求
  3. 触发源空值兜底 (默认场景+兜底文案)
  4. imagination 记忆入库带 parent_ids (类型/情绪节点, 避免孤立)
  5. daily_diary 检查 api_key
  6. _load_size 容错损坏 JSON
  7. prompt 长度指令改为语义描述 (不误导 token/字数)
  8. 情绪标记/落地失败记录 warning (不静默)
  9. requests 顶部导入
  10. 文件写入用 with 上下文管理器
  11. latest_thought 支持最近N条/加权
  12. size_for_context 综合情绪状态 (不纯关键词)
"""
import json
import logging
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests  # fix 9: 顶部导入

logger = logging.getLogger("humanize_ai.mindstream")

CN_TZ = timezone(timedelta(hours=8))
BASE = Path(__file__).resolve().parents[1]

# 内心流类型 (自由, 不分类打分 — 只用于观察统计/网络节点)
TYPES = ["emotion_digest", "recall", "plan", "daydream", "rumination", "desire", "self_reflect"]

# 默认冥想场景 (fix 3: 场景文件缺失时的兜底)
DEFAULT_SCENES = {
    "nature": [
        "窗外的树影在动，你看着它发呆。",
        "风轻轻吹过，树叶沙沙响，你坐在门口发呆。",
    ],
    "city": [
        "路上车来车往，你趴在窗边看，没什么特别的事。",
        "楼下的路灯亮着，你站在阳台上看了一会儿。",
    ],
    "emotion_space": [
        "你一个人待在房间里，想着一些事情。",
        "你坐在床边，安静地待了一会儿。",
    ],
}


class Mindstream:
    def __init__(self, base_dir=None, memory_bank=None, emotion_state=None,
                 api_key: str = "", model_path: str = ""):
        self.base = Path(base_dir) if base_dir else BASE
        self.self_dir = self.base / "phase1" / "self"
        self.self_dir.mkdir(parents=True, exist_ok=True)
        self.memory = memory_bank
        self.emotion = emotion_state
        self.api_key = api_key or self._load_api_key()
        self.model_path = model_path
        # 冥想场景 (带默认兜底, fix 3)
        self.scenes = self._load_scenes()
        # 心流大小 (v4: 可控制思考大小的模型, 社会反馈学习)
        self.size = "medium"  # shallow | medium | deep
        self.size_log = self.self_dir / "mindstream_size.json"
        self._load_size()
        # Mind 模块可用性 (fix 1: 提前检查, 不静默; 避免循环引用 — 用外部传入或延迟)
        # 内联情绪词典 (与 emotion.py LEXICON 同源, 解耦 Mind↔Mindstream 循环)
        self._event_lexicon = {
            "joy": ["开心", "高兴", "喜欢", "笑", "好吃", "好玩", "棒", "幸福", "甜", "暖"],
            "sadness": ["难过", "伤心", "哭", "委屈", "难受", "失落", "想哭", "烦", "累"],
            "anger": ["生气", "气死", "讨厌", "烦死", "恼火", "气人"],
            "fear": ["害怕", "怕", "吓", "紧张", "不敢", "慌"],
            "surprise": ["哇", "居然", "没想到", "惊讶", "震惊"],
        }

    # ============ 心流大小控制 (v4) ============
    SIZE_LIMITS = {
        "shallow": {"max_tokens": 40, "temperature": 0.7, "label": "浅"},
        "medium": {"max_tokens": 80, "temperature": 0.9, "label": "中"},
        "deep": {"max_tokens": 150, "temperature": 1.0, "label": "深"},
    }

    # 场景 → 心流大小 (初期规则, 社会反馈学习后自适应)
    SCENE_SIZE = [
        (["妈妈问", "爸爸问", "老师问", "询问", "检查", "约束", "别发呆", "想这么久"], "shallow"),
        (["朋友", "同学", "聊天", "一起", "周末"], "medium"),
        (["独自", "一个人", "窗边", "夜深", "发呆", "躺", "睡前", "哭", "难过", "生气", "开心"], "deep"),
    ]

    def _load_size(self):
        """fix 6: 容错损坏 JSON"""
        try:
            if self.size_log.exists():
                d = json.loads(self.size_log.read_text(encoding="utf-8"))
                self.size = d.get("size", "medium")
        except Exception as e:
            logger.warning("mindstream_size.json 读取失败，使用默认 medium: %s", e)
            self.size = "medium"

    def _save_size(self):
        try:
            self.size_log.write_text(json.dumps({"size": self.size,
                                                  "time": datetime.now(CN_TZ).isoformat(timespec="seconds")},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("保存心流大小失败: %s", e)

    def set_size(self, size: str, reason: str = ""):
        """设置心流大小 (外部: 场景/社会反馈可调)"""
        if size not in self.SIZE_LIMITS:
            return
        if size != self.size:
            self.size = size
            self._save_size()
            log = self.self_dir / "mindstream_size_changes.jsonl"
            try:
                with open(log, "a", encoding="utf-8") as f:  # fix 10: with
                    f.write(json.dumps({"time": datetime.now(CN_TZ).isoformat(timespec="seconds"),
                                        "size": size, "reason": reason}, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning("记录心流大小变化失败: %s", e)

    def size_for_context(self, context: str = "") -> str:
        """fix 12: 综合情绪状态决定心流大小 (不只关键词)"""
        # 情绪权重: 情绪波动大 → 深 (需要消化)
        if self.emotion is not None:
            try:
                strong = max(abs(self.emotion.state[e] - self.emotion.baseline[e])
                             for e in self.emotion.state)
                if strong > 0.35:
                    return "deep"
            except Exception:
                pass
        # 关键词规则
        for keywords, size in self.SCENE_SIZE:
            if any(k in context for k in keywords):
                return size
        return self.size

    def _load_scenes(self) -> dict:
        """fix 3: 默认场景兜底 + 用户场景合并"""
        try:
            p = self.self_dir / "meditation_scenes.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                # 与默认合并, 保证每类非空
                for cat in DEFAULT_SCENES:
                    data.setdefault(cat, DEFAULT_SCENES[cat])
                    if not data[cat]:
                        data[cat] = DEFAULT_SCENES[cat]
                return data
        except Exception as e:
            logger.warning("冥想场景文件读取失败，使用默认场景: %s", e)
        return {k: list(v) for k, v in DEFAULT_SCENES.items()}

    def _load_api_key(self) -> str:
        """fix 2: .env 读取容错; 无 key 返回空 (调用方检查)"""
        import os
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if key:
            return key
        env = self.base / ".env"
        if env.exists():
            try:
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.startswith("DEEPSEEK_API_KEY="):
                        return line.split("=", 1)[1].strip()
            except Exception as e:
                logger.warning(".env 读取失败: %s", e)
        return ""

    # ============ 触发源 ============
    def _trigger(self, scene_text: str = "", context: dict = None) -> str:
        """根据情绪/欲望/未了结事件/随机 生成内心流触发描述
        fix 3: 空值兜底 — 绝不返回空字符串"""
        ctx = context or {}
        if scene_text:
            return scene_text
        # 情绪波动大 → emotion_space 场景
        if self.emotion is not None:
            try:
                strong = max(abs(self.emotion.state[e] - self.emotion.baseline[e])
                             for e in self.emotion.state)
                if strong > 0.3 and self.scenes.get("emotion_space"):
                    return random.choice(self.scenes["emotion_space"])
            except Exception:
                pass
        cats = [c for c in self.scenes if self.scenes.get(c)]
        if cats:
            return random.choice(self.scenes[random.choice(cats)])
        # 最终兜底 (所有场景都空)
        return "现在没什么特别的事，你正发着呆。"

    # ============ 内心生成 (无监督, 零约束) ============
    def generate_thought(self, trigger: str = "", humanizer=None) -> dict:
        """生成一条内心活动。返回 {text, type, trigger, time, size}
        关键: 无监督 — 不评分/不纠错/无"要自然"指令, 高温度自由生成"""
        size = self.size_for_context(trigger)
        limits = self.SIZE_LIMITS[size]
        if humanizer is None:
            return self._generate_via_api(trigger, limits, size)
        return self._generate_via_local(humanizer, trigger, limits, size)

    def _generate_via_api(self, trigger: str, limits: dict, size: str = "medium") -> dict:
        """DeepSeek 生成 (无监督) — 仅开发测试用, 正式用本地
        fix 2: 空 key 直接返回, 不发起无效请求"""
        if not self.api_key:
            return {"text": "", "trigger": trigger[:80], "size": size,
                    "time": datetime.now(CN_TZ).isoformat(timespec="seconds"),
                    "error": "no_api_key"}
        # fix 7: 长度用语义描述, 不误导 token/字数
        # fix 13: 轻量中文约束 (不强指令, 保留无监督自由)
        prompt = f"{trigger}\n\n你正在发呆/独处。此刻你心里在想什么？把你的内心活动写下来，想到什么写什么，不用太长，不要解释。\n你平时只说中文，心里想事情也用中文。"
        try:
            r = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": limits["temperature"], "max_tokens": limits["max_tokens"]},
                timeout=60)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            return {"text": text, "trigger": trigger[:80], "size": size,
                    "time": datetime.now(CN_TZ).isoformat(timespec="seconds")}
        except Exception as e:
            logger.warning("内心流 API 生成失败: %s", str(e)[:100])
            return {"text": "", "trigger": trigger[:80], "size": size,
                    "time": datetime.now(CN_TZ).isoformat(timespec="seconds"), "error": str(e)[:80]}

    def _generate_via_local(self, humanizer, trigger: str, limits: dict, size: str = "medium") -> dict:
        """本地子AI生成 (更"无意识" — 用她自己的模型, v4决策④)
        fix 13: 轻量中文约束 (不强指令, 保留无监督自由)
        fix 15: 独白化 — 明确这是一个人发呆时的内心念头, 不是对任何人说话
        fix 16: 对话特征检测 — 生成内容像在跟人说话时重试一次"""
        prompt = (f"{trigger}\n\n"
                  f"你现在一个人待着，正在发呆。你心里在想什么？把心里那个念头写出来，"
                  f"想到什么写什么，不用太长，不要解释。\n"
                  f"注意：这是你心里的念头，不是跟任何人说话——不要称呼对方、不要问对方问题、"
                  f"不要用'咱们''你''好不好'这类对别人说话的口气。\n"
                  f"你平时只说中文，心里想事情也用中文。")
        import re as _re
        _talk_pat = _re.compile(r"你说|咱们|你呢|咋样|好不好|对吧|跟你|告诉你|听我说|这样咋样")
        last = ""
        for attempt in range(2):
            try:
                text = humanizer.generate(prompt, temperature=limits["temperature"],
                                          max_tokens=limits["max_tokens"], memory_query=None)
                text = text.strip()
                last = text
                if text and not _talk_pat.search(text):
                    break
                logger.warning("内心流含对话特征, 重试 %d/2", attempt + 1)
            except Exception as e:
                logger.warning("内心流本地生成失败: %s", str(e)[:100])
                return {"text": "", "trigger": trigger[:80], "size": size,
                        "time": datetime.now(CN_TZ).isoformat(timespec="seconds"), "error": str(e)[:80]}
        return {"text": last.strip(), "trigger": trigger[:80], "size": size,
                "time": datetime.now(CN_TZ).isoformat(timespec="seconds")}

    # ============ 落地: 记忆+情绪+日志 ============
    def _tag_emotion(self, text: str) -> dict:
        """fix 1/8: 情绪标签 — 内联词典实现 (不依赖 Mind, 无循环风险)"""
        scores = {}
        for emo, words in self._event_lexicon.items():
            s = sum(1 for w in words if w in text)
            if s:
                scores[emo] = s
        if not scores:
            return {"dominant": "neutral", "arousal": 0.1, "valence": 0.0}
        dom = max(scores, key=scores.get)
        # 简化 valence/arousal 映射
        vmap = {"joy": (0.7, 0.5), "sadness": (-0.6, 0.4), "anger": (-0.4, 0.7),
                "fear": (-0.5, 0.6), "surprise": (0.1, 0.7)}
        val, aro = vmap.get(dom, (0.0, 0.2))
        return {"dominant": dom, "arousal": aro, "valence": val}

    def _apply_somatic(self, emotion: str, arousal: float, text: str):
        """fix 1/8: 躯体标记 — 直接操作 emotion_state (内联, 无循环)"""
        if self.emotion is None or emotion == "neutral":
            return
        try:
            impact = {emotion: min(0.06 * arousal, 0.20)}
            self.emotion.apply_impact(impact)
        except Exception as e:
            logger.warning("躯体标记失败: %s", e)

    def process_thought(self, thought: dict) -> dict:
        """① imagination记忆入库(带情绪标签+父节点, 弱衰减标记)
           ② 情绪影响 (躯体标记: 想起难过的事→心情变差)
           ③ 内心流日志"""
        text = thought.get("text", "")
        if not text:
            return thought
        # fix 14: 语言质量门 — 英文占比>5% 丢弃不入库 (防污染记忆库/训练集)
        import re as _re
        en_words = _re.findall(r"[a-zA-Z]+\b", text)
        cn_chars = _re.findall(r"[\u4e00-\u9fff]", text)
        if en_words and len(en_words) / max(len(cn_chars), 1) > 0.05:
            thought["low_quality"] = "eng_ratio"
            logger.warning("内心流英文占比过高(%.0f%%), 丢弃不入库: %s",
                           len(en_words) / max(len(cn_chars), 1) * 100, text[:40])
            return thought
        # 情绪标签 (fix 1: 独立方法)
        t = self._tag_emotion(text)
        emotion = t["dominant"]
        thought["emotion"] = emotion
        # ② 情绪影响 (内心记忆同样触发躯体标记, 浅层)
        self._apply_somatic(emotion, t.get("arousal", 0.1), text)
        # ① 入库 (imagination标签 + 父节点, fix 4: 避免网络孤立)
        if self.memory is not None and len(text) >= 8:
            try:
                exp_id = "exp_" + uuid.uuid4().hex[:12]
                # fix 4: 父节点 = 类型节点 + 情绪节点 (坐标网关联)
                parents = []
                if emotion != "neutral":
                    parents.append(f"emotion_{emotion}")
                parents.append("type_imagination")
                self.memory.store(exp_id=exp_id, text=text,
                                  source_url="imagination",
                                  parent_ids=parents,
                                  source_year=2026,
                                  emotion_vector={"valence": t.get("valence", 0.0),
                                                  "arousal": t.get("arousal", 0.1),
                                                  "dominant": emotion})
                thought["memory_id"] = exp_id
            except Exception as e:
                logger.warning("内心记忆入库失败: %s", e)
        # ③ 日志 (fix 10: with)
        today = datetime.now(CN_TZ).strftime("%Y-%m-%d")
        log_file = self.self_dir / f"mindstream_{today}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(thought, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("内心流日志写入失败: %s", e)
        return thought

    # ============ 冥想入口 (情绪后/定时) ============
    def meditate(self, humanizer=None, trigger: str = "") -> dict:
        """进入冥想: 触发场景 → 内心生成 → 落地"""
        t = self._trigger(trigger)
        thought = self.generate_thought(t, humanizer)
        return self.process_thought(thought)

    # ============ 每日内心日记 (记录器) ============
    def daily_diary(self, api_key: str = "") -> dict:
        """睡前: 整理当天内心流 → 内心日记 (观察数据, 不给她)
        fix 5: 无 key 直接返回"""
        key = api_key or self.api_key
        if not key:
            return {"text": "", "reason": "no_api_key"}
        today = datetime.now(CN_TZ).strftime("%Y-%m-%d")
        log_file = self.self_dir / f"mindstream_{today}.jsonl"
        if not log_file.exists():
            return {"text": "", "reason": "no_mindstream"}
        try:
            lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception as e:
            logger.warning("内心流日志读取失败: %s", e)
            return {"text": "", "reason": "log_read_error"}
        if not lines:
            return {"text": "", "reason": "empty"}
        thoughts = "\n".join(f"- {l.get('text', '')}" for l in lines[-15:])
        prompt = f"""这是蠢珞珞今天的内心活动记录(无监督, 她不知道被记录):
{thoughts}

请以研究者的视角, 写一份"内心观察日记"(150字内):
1. 她今天的主要心理活动/情绪轨迹
2. 值得注意的模式(反刍/回避/渴望)
3. 对系统改进的提示(情绪机/记忆/欲望 哪里需要调)
只输出观察日记本体。"""
        try:
            r = requests.post("https://api.deepseek.com/chat/completions",
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": "deepseek-chat",
                                    "messages": [{"role": "user", "content": prompt}],
                                    "temperature": 0.4, "max_tokens": 300},
                              timeout=60)
            r.raise_for_status()
            diary = r.json()["choices"][0]["message"]["content"].strip()
            out = self.self_dir / f"mind_diary_{today}.md"
            out.write_text(f"# {today} 内心观察日记\n\n{diary}\n", encoding="utf-8")
            return {"text": diary, "path": str(out)}
        except Exception as e:
            logger.warning("内心日记生成失败: %s", str(e)[:100])
            return {"text": "", "reason": str(e)[:80]}

    # ============ 内心流注入对话 ============
    def latest_thought(self, hours: int = 6, top_k: int = 1) -> str:
        """fix 11: 最近内心流 (支持 top_k 条/时间加权随机, 默认最新1条)"""
        today = datetime.now(CN_TZ).strftime("%Y-%m-%d")
        log_file = self.self_dir / f"mindstream_{today}.jsonl"
        if not log_file.exists():
            return ""
        try:
            lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception:
            return ""
        if not lines:
            return ""
        # 取最近 top_k 条, 按时间加权随机选一条 (近期权重高)
        recent = lines[-max(1, top_k * 3):]
        if not recent:
            return ""
        weights = list(range(1, len(recent) + 1))  # 越新权重越大
        pick = random.choices(recent, weights=weights, k=1)[0]
        return pick.get("text", "")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.stdout.reconfigure(encoding="utf-8")
    ms = Mindstream()
    thought = ms.meditate()
    print(f"内心活动: {thought.get('text', '(空)')[:80]}")
    print(f"情绪: {thought.get('emotion', '?')} | size: {thought.get('size', '?')} | 入库: {thought.get('memory_id', '无')}")
