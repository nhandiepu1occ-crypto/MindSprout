"""
mind.py — 出厂心智模块 (Consciousness as a Birth Feature)

把 9 个意识工程整合为 Humanizer 的出厂机制, 与 memory_bank / dna / emotion 同级。
出生(构造 Humanizer)即有, 无需外部脚本。

机制清单 (9项, 日记除外 — 日记是每日后台任务, 归 cron 管):
  1. sensor           常驻感知: 时间/主人活动/自我心情 → self/day_*.jsonl
  2. sleep_consolidate 睡眠巩固: 当天感知 → DeepSeek 提炼 → 记忆库
  3. emotion_tag      记忆情绪标签: 记忆入库时自动打 emotion_vector
  4. emotion_bias     情绪偏置检索: 当前情绪匹配的记忆信号增强 (Bower 1981)
  5. somatic          躯体标记: 回忆强情绪记忆 → 情绪状态被唤起 (Damasio)
  6. confidence       置信度: 知识给出度+回答一致性 → 真"不知道"
  7. inner_monologue  内在叙事: 生成前小预算"念头" → 注入 system 影响表达
  8. initiative       主动意愿: open loops 检索 → 主动发起话题
  9. preference       偏好层: 记忆统计喜欢/讨厌 → system 注入

对话时自动挂钩 (generate 内部):
  before: 情绪 tick → 偏置检索 → 躯体标记 → 念头 → (可选)置信度
  after:  情绪保存

后台任务接口 (cron 调):
  mind.sense() / mind.consolidate() / mind.initiative() / mind.build_preferences()
"""

from mindsprout.config import BASE

import json
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("humanize_ai.mind")

CN_TZ = timezone(timedelta(hours=8))
EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "surprise"]

# ---- 记忆情绪标签 (emotion_tag, 出厂词表) ----
EVENT_LEXICON = {
    "joy": ["生日", "礼物", "奖状", "第一名", "表扬", "夸奖", "奖", "糖", "新书包", "高兴", "喜欢", "开心",
            "好玩", "笑", "蛋糕", "逛街", "好吃的", "红包", "新衣服", "小木马", "滑梯", "跳皮筋", "赢了",
            "胜利", "春游", "秋游", "动物园", "游乐园", "玩具"],
    "sadness": ["哭", "骂", "摔", "丢", "不见", "走丢", "打架", "欺负", "嘲笑", "罚站", "不及格", "批评",
                "难过", "伤心", "委屈", "想哭", "受伤", "生病", "打针", "摔跤", "后悔", "遗憾", "分离",
                "转学", "搬走", "去世", "走了", "离开"],
    "fear": ["怕", "黑", "打雷", "闪电", "鬼", "噩梦", "吓", "害怕", "紧张", "发抖", "不敢", "高", "跳",
             "摔下去", "迷路", "陌生人", "狗叫"],
    "anger": ["气", "讨厌", "烦", "吵", "抢", "赖皮", "不讲理", "冤枉", "误会", "吼", "瞪"],
    "surprise": ["突然", "居然", "没想到", "惊讶", "震惊", "吓一跳", "惊喜", "哇"],
}
EMO_VALENCE = {"joy": 0.8, "sadness": -0.7, "anger": -0.5, "fear": -0.6, "disgust": -0.4, "surprise": 0.1}
EMO_AROUSAL = {"joy": 0.6, "sadness": 0.4, "anger": 0.8, "fear": 0.7, "disgust": 0.4, "surprise": 0.8}

# ---- 偏好主题词 (preference) ----
PREF_THEMES = {
    "滑梯": "滑梯", "小木马": "小木马", "皮筋": "跳皮筋", "跳皮筋": "跳皮筋",
    "积木": "积木", "玩具": "玩具", "布偶": "布偶", "洋娃娃": "洋娃娃",
    "小猫": "小猫", "小花": "小花", "橘猫": "小花", "宠物": "小动物",
    "蛋糕": "蛋糕", "糖": "糖果", "零食": "零食", "冰激凌": "冰淇淋",
    "画画": "画画", "橡皮泥": "橡皮泥", "手工": "手工",
    "唱歌": "唱歌", "跳舞": "跳舞", "故事": "听故事", "绘本": "绘本",
    "游泳": "游泳", "爬山": "爬山", "秋游": "秋游", "春游": "春游",
    "打针": "打针", "医院": "医院", "药": "吃药", "发烧": "生病",
    "天黑": "天黑", "雷": "打雷", "打雷": "打雷",
    "吵架": "吵架", "抢": "被抢", "欺负": "被欺负", "嘲笑": "被嘲笑",
    "作业": "写作业", "考试": "考试", "罚站": "被罚站", "批评": "被批评",
    "手机": "手机", "手表": "电话手表", "电话手表": "电话手表",
    "学校": "上学", "上课": "上课", "老师": "老师", "同学": "同学",
    "操场": "操场", "秋千": "秋千", "跷跷板": "跷跷板",
}

# ---- 置信度: 逃避/间接知识词 (confidence) ----
EVASION_WORDS = ["超纲", "没学过", "没教过", "不知道", "不懂", "不记得", "没听过", "没听说过",
                 "没学到", "不会", "我哪会", "这题", "换一个", "别提", "不会吧", "什么玩意",
                 "听不懂", "咱不说", "你别问", "好难", "学不会", "忘了",
                 "不想学", "不想听", "不想懂", "好烦", "搞不懂", "不太理解", "不理解", "复杂",
                 "爸爸提过", "爸爸说过", "爸爸经常聊", "爸爸聊过", "我爸聊", "我爸说", "妈妈提过",
                 "我妈说", "爸妈聊", "听我爸", "听我妈", "爸爸加班", "我爸加班"]

# 古诗字集 (knowledge_score 用)
CN_CHARS = set("鹅曲项向天歌白毛浮绿水红掌拨清波床前明月光疑是地上霜举头望明月低头思故乡锄禾日当午汗滴禾下土谁知盘中餐粒粒皆辛苦")


class Mind:
    """出厂心智: 9 项机制打包, 出生即有"""

    def __init__(self, base_dir: Path, dna=None, memory_bank=None, emotion_state=None,
                 model_dir: str = ""):
        self.base = Path(base_dir)
        self.self_dir = self.base / "phase1" / "self"
        self.self_dir.mkdir(parents=True, exist_ok=True)
        self.dna = dna
        self.memory = memory_bank
        self.emotion = emotion_state
        self.model_dir = model_dir
        # 偏好缓存 (惰性加载)
        self._prefs = None
        # 内心流引擎 (V13 出厂机制: 内心=记忆的一种+心流大小可调)
        self.mindstream = None
        try:
            from humanize_ai.mindstream import Mindstream
            self.mindstream = Mindstream(base_dir=self.base, memory_bank=self.memory,
                                         emotion_state=self.emotion,
                                         model_path=self.model_dir)
        except Exception as e:
            print(f"⚠️ Mindstream 加载失败: {e}")

    # ============ 1. 常驻感知 (sensor) ============
    def sense(self) -> list:
        """感知 时间/活动/自我 → 追加 self/day_*.jsonl。返回事件列表。"""
        events = [self._sense_time(), self._sense_activity(), self._sense_self()]
        today = datetime.now(CN_TZ).strftime("%Y-%m-%d")
        with open(self.self_dir / f"day_{today}.jsonl", "a", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return events

    def _sense_time(self) -> dict:
        now = datetime.now(CN_TZ)
        wd = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
        hour = now.hour
        period = ("清晨" if 5 <= hour < 8 else "上午" if hour < 12 else "中午" if hour < 14
                  else "下午" if hour < 18 else "晚上" if hour < 22 else "深夜")
        return {"type": "time", "time": now.isoformat(timespec="seconds"), "hour": hour,
                "weekday": wd, "period": period,
                "text": f"现在是{wd}{hour}点多（{period}），{now.month}月{now.day}日。"}

    def _sense_activity(self) -> dict:
        try:
            recent = []
            cutoff = datetime.now().timestamp() - 900
            ws = self.base.parent  # workspace
            for p in ws.rglob("*"):
                if p.is_file() and p.stat().st_mtime > cutoff:
                    try:
                        rel = p.relative_to(ws)
                        if any(x in str(rel).lower() for x in [".git", "__pycache__", ".openclaw"]):
                            continue
                        recent.append(str(rel))
                    except Exception:
                        pass
                    if len(recent) >= 5:
                        break
            return {"type": "activity", "time": datetime.now(CN_TZ).isoformat(timespec="seconds"),
                    "recent_files": recent,
                    "text": f"主人最近在动 {'、'.join(recent[:3]) if recent else '（没什么动静）'}。"}
        except Exception as e:
            return {"type": "activity", "time": datetime.now(CN_TZ).isoformat(timespec="seconds"),
                    "recent_files": [], "text": f"（感知主人活动失败: {e}）"}

    def _sense_self(self) -> dict:
        try:
            mood = self.emotion.to_prompt() if self.emotion else ""
            return {"type": "self", "time": datetime.now(CN_TZ).isoformat(timespec="seconds"),
                    "mood": mood, "text": f"我的心情：{mood or '（心情平稳）'}。"}
        except Exception as e:
            return {"type": "self", "time": datetime.now(CN_TZ).isoformat(timespec="seconds"),
                    "mood": "", "text": f"（感知自我失败: {e}）"}

    # ============ 2. 睡眠巩固 (sleep_consolidate) ============
    def consolidate(self, api_key: str = "", dry_run: bool = False) -> dict:
        """当天感知 → DeepSeek 提炼 ≤3 条 → 记忆库。返回 {"added": n, "items": [...]}"""
        today = datetime.now(CN_TZ).strftime("%Y-%m-%d")
        df = self.self_dir / f"day_{today}.jsonl"
        if not df.exists():
            return {"added": 0, "items": [], "reason": "no_day_log"}
        events = [json.loads(l) for l in df.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(events) < 3:
            return {"added": 0, "items": [], "reason": "too_few_events"}
        if not api_key:
            api_key = self._load_api_key()
        if not api_key:
            return {"added": 0, "items": [], "reason": "no_api_key"}

        prompt = self._consolidate_prompt(events, today)
        resp = self._deepseek(prompt, api_key, max_tokens=800)
        items = self._extract_json_list(resp)
        if not items:
            return {"added": 0, "items": [], "reason": "parse_fail"}
        if dry_run:
            return {"added": 0, "items": items, "reason": "dry_run"}
        added = 0
        for m in items:
            txt = (m.get("text") or "").strip()
            if len(txt) < 10:
                continue
            self._store_memory(txt, source="sleep_consolidate",
                               emotion_vector={"valence": 0.0, "arousal": 0.1})
            added += 1
        return {"added": added, "items": items, "reason": "ok"}

    def _consolidate_prompt(self, events, today_str) -> str:
        ev_lines = "\n".join(f"- {e.get('text', '')}" for e in events)
        existing = self._existing_memories()
        ex_lines = "\n".join(f"- {m[:80]}" for m in existing[-20:]) if existing else "(暂无)"
        return f"""你是蠢珞珞的记忆整理系统（睡前记忆巩固）。

今天（{today_str}）她感知到/经历的事件：
{ev_lines}

她已有的长期记忆（最近20条，供去重参考）：
{ex_lines}

请把今天的经历提炼成"值得长期记住"的记忆条目，规则：
1. 最多输出 3 条，每条 30-80 字，第一人称（"我..."）
2. 只保留有意义的：时间感（今天星期几/日期）、主人的重要活动、她自己心情变化
3. 普通琐事（如"现在8点"）不保留；与已有记忆重复的不保留
4. 输出 JSON 数组：[{{"text": "...", "emotion": "joy|sadness|anger|fear|neutral"}}]
只输出 JSON，不要其他文字。"""

    # ============ 3. 情绪标签 (emotion_tag) ============
    def tag_memory(self, text: str) -> dict:
        """给一条记忆打情绪标签 → {dominant, valence, arousal, hits}"""
        scores = {}
        for emo, words in EVENT_LEXICON.items():
            s = sum(1 for w in words if w in text)
            if s:
                scores[emo] = s
        try:
            from humanize_ai.emotion import LEXICON
            for emo, words in LEXICON.items():
                s = sum(wt for w, wt in words.items() if w in text)
                if s:
                    scores[emo] = scores.get(emo, 0) + s / 3.0
        except Exception:
            pass
        if not scores:
            return {"dominant": "neutral", "valence": 0.0, "arousal": 0.1, "hits": {}}
        dom = max(scores, key=scores.get)
        return {"dominant": dom, "valence": EMO_VALENCE[dom], "arousal": EMO_AROUSAL[dom], "hits": scores}

    def tag_all_memories(self) -> int:
        """给记忆库全部记忆打标 (启动时自动; 幂等)"""
        if self.memory is None:
            return 0
        n = 0
        for exp_id in self.memory.content.ids():
            exp = self.memory.content.get(exp_id)
            if exp and exp.text and not exp.emotion_vector:
                t = self.tag_memory(exp.text)
                exp.emotion_vector = {"valence": t["valence"], "arousal": t["arousal"],
                                      "dominant": t["dominant"]}
                n += 1
        if n:
            try:
                self.memory.save(str(self.base / "phase1" / "memory_luoluo"))
            except Exception:
                pass
        return n

    # ============ 4. 情绪偏置检索 (emotion_bias) ============
    def bias_for_current_emotion(self) -> str:
        """当前主导情绪 → 偏置标签 (neutral 不偏置)"""
        if self.emotion is None:
            return None
        try:
            dom = self.emotion.dominant()
            return dom if dom != "neutral" else None
        except Exception:
            return None

    # ============ 5. 躯体标记 (somatic) ============
    def apply_somatic(self, contents, gain: float = 0.08, max_delta: float = 0.20) -> dict:
        """回忆强情绪记忆 → 情绪被唤起。返回 {emotion: delta}"""
        if self.emotion is None or not contents:
            return {}
        dom_to_state = {"joy": {"joy": 1.0}, "sadness": {"sadness": 1.0}, "anger": {"anger": 1.0},
                        "fear": {"fear": 1.0}, "disgust": {"disgust": 1.0}, "surprise": {"surprise": 1.0}}
        deltas = {}
        for exp, signal in contents[:3]:
            ev = exp.emotion_vector or {}
            dom = ev.get("dominant", "neutral")
            if dom == "neutral" or dom not in dom_to_state:
                continue
            strength = min(gain * ev.get("arousal", 0.3) * signal, max_delta)
            if strength <= 0.01:
                continue
            impact = {emo: strength * w for emo, w in dom_to_state[dom].items()}
            self.emotion.apply_impact(impact)
            for emo, d in impact.items():
                deltas[emo] = deltas.get(emo, 0) + d
        return deltas

    # ============ 6. 置信度 (confidence) ============
    def knowledge_score(self, text: str) -> float:
        if not text:
            return 0.0
        import re
        quotes = re.findall(r'[""「」“”《》]([^""「」“”《》]{2,20})', text)
        q_len = sum(len(q) for q in quotes)
        nums = len(re.findall(r"\d+", text))
        poem_hits = sum(1 for ch in text if ch in CN_CHARS)
        return (q_len + nums * 3 + poem_hits * 0.5) / max(len(text), 1) * 100

    def estimate_confidence(self, answers: list) -> dict:
        """输入 1-2 条回答 → {level: high|medium|low, knowledge, evasion, agreement}"""
        if not answers:
            return {"level": "medium", "knowledge": 0.0, "evasion": 0.0, "agreement": 0.0}
        k = max(self.knowledge_score(a) for a in answers)
        ev = sum(1 for a in answers if any(w in a for w in EVASION_WORDS))
        ev_ratio = ev / len(answers)
        agreement = 1.0
        if len(answers) >= 2:
            def bigrams(s):
                s = s.strip()
                return set(s[i:i + 2] for i in range(len(s) - 1))
            b1, b2 = bigrams(answers[0]), bigrams(answers[1])
            agreement = (len(b1 & b2) / len(b1 | b2)) if (b1 and b2) else 0.0
        if ev_ratio >= 0.5:
            level = "low"
        elif k >= 2.0 or (agreement >= 0.5 and ev_ratio < 0.5):
            level = "high"
        elif agreement < 0.25 and k < 1.0:
            level = "low"
        else:
            level = "medium"
        return {"level": level, "knowledge": round(k, 2), "evasion": round(ev_ratio, 2),
                "agreement": round(agreement, 2)}

    # ============ 7. 内在叙事 + 对话理解 (inner_monologue + comprehension) ============
    def get_thought(self, humanizer, user_text: str, seed: int = 7) -> str:
        """生成内在念头 (小预算, 纯中文) — 需要 humanizer 实例
        2026-08-15 v3: 理解层 = ①听懂对方 ②想起相关经历(记忆直嵌任务, 防长prompt注意力丢失) ③内心反应
        """
        import re
        # 指代历史的问题 (上次/刚才/之前/我说过): 答案在对话历史不在记忆, 念头跳过记忆检索
        is_ref = bool(re.search(r"上次|刚才|之前|我说过|我说了|你记得我说|上句话|上一句|还记得我吗", user_text))
        # 楚门 v1.6: "你是谁/认识我吗" → 念头定向检索"他"的认知记忆 (模糊熟悉感)
        who_ask = bool(re.search(r"你是谁|我是谁|认识我吗|你猜我是谁|你知道我是谁|认识很久|认得我", user_text))
        mems = []
        if humanizer._memory is not None:
            try:
                if who_ask:
                    r = humanizer._memory.query(
                        query_text="一个经常来找我说话的人，我记不清他是谁但觉得熟悉",
                        top_k=3, k_hops=1)
                elif not is_ref:
                    r = humanizer._memory.query(query_text=user_text, top_k=3, k_hops=1)
                else:
                    r = None
                if r:
                    mems = [e.text for e, s in r["contents"][:3] if getattr(e, "text", "")]
            except Exception:
                pass
        mem_block = ""
        if mems:
            mem_block = "你想起自己经历过：" + "；".join(
                f"({i}){m[:60]}" for i, m in enumerate(mems[:2], 1)) + "\n"
        # 楚门 v1.9: 人物识别 — 念头先认人, 再决定内心基调 (对妈妈/对他/对小雨不一样)
        person_hint = ""
        try:
            from humanize_ai.persona import identify, person_line
            _pr = identify(user_text, context_hint="")
            if _pr["person_id"] and _pr["confidence"] >= 0.55:
                person_hint = person_line(_pr["person_id"]) + "\n"
        except Exception:
            pass
        ref_note = "对方说'上次/刚才'，但你可能不记得说过 — 心里老实说'记不得'，别瞎编。\n" if is_ref else ""
        # 楚门 v1.6: "你是谁"类 → 熟悉感引导 (记忆里有"他"的影子)
        if who_ask:
            ref_note = ("你心里对这个人有种说不上来的熟悉感，像认识很久很久了，可你想不起来他是谁、叫什么。"
                        "别急着否认，也别乱编名字和身份。\n")
        # 楚门 v1.9: 人物识别 — 念头先认人, 再决定内心基调 (对妈妈/对他/对小雨不一样)
        person_hint = ""
        try:
            from humanize_ai.persona import identify, person_line
            _pr = identify(user_text, context_hint="")
            if _pr["person_id"] and _pr["confidence"] >= 0.55:
                person_hint = person_line(_pr["person_id"]) + "\n"
        except Exception:
            pass
        prompt = (f"{mem_block}"
                  f"{person_hint}"
                  f"对方刚对你说：{user_text[:150]}\n"
                  f"（你叫珞珞，有时被叫小珞珞，那都是在叫你。\n"
                  f"对方可能是任何人：家人、朋友、老师、邻居、陌生人。\n"
                  f"如果认不出对方是谁、不记得他说的事，心里就老实说'认不出/不记得'，别瞎猜，也别猜对方是什么关系。）\n"
                  f"{ref_note}"
                  f"你心里先听懂对方在做什么（他/她是在…），想起相关的事（如果有），"
                  f"然后冒出一个念头。\n"
                  f"写这个念头，像心里话，15-40字，口语、带情绪。只写念头。")
        try:
            thought = humanizer.generate(prompt, temperature=0.6, max_tokens=60, seed=seed,
                                         memory_query=None,  # 记忆已直嵌, 防递归
                                         _internal_skip_emotion=True,  # M3
                                         _internal_skip_mind=True)     # 防递归
            thought = thought.strip().strip('"').strip("「」")
            thought = re.sub(r"[a-zA-Z][a-zA-Z ]{1,40}", "", thought).strip()
            thought = re.sub(r"\s+", " ", thought).strip()
            print(f"[THOUGHT] {user_text[:15]!r} -> {thought[:60]!r}", flush=True)  # 楚门debug
            # 关系断言过滤: 念头瞎猜关系(好朋友/妈咪/你是我) → 不注入, 让训练样本主导
            if re.search(r"好朋友|妈咪|你是我|你是我的|我认识你|你是我妈|爸爸|妈妈在", thought):
                return ""
            return thought[:80]
        except Exception:
            return ""

    def build_system_with_thought(self, base_system: str, thought: str) -> str:
        if not thought:
            return base_system
        return base_system + f"\n（你心里刚刚闪过一个念头：{thought}。顺着这个念头自然回应，但别把这句话直接说出来。）"

    # ============ 8. 主动意愿 (initiative) ============
    def find_open_loops(self, limit: int = 5) -> list:
        """强情绪记忆 (sadness/fear/anger + arousal≥0.4) → open loops"""
        if self.memory is None:
            return []
        cands = []
        for exp_id in self.memory.content.ids():
            exp = self.memory.content.get(exp_id)
            if not exp or not exp.text:
                continue
            ev = exp.emotion_vector or {}
            if ev.get("dominant", "neutral") in ("sadness", "fear", "anger") and ev.get("arousal", 0.1) >= 0.4:
                cands.append((exp_id, exp.text, ev.get("dominant"), ev.get("arousal", 0.1)))
        cands.sort(key=lambda x: -x[3])
        return cands[:limit]

    def initiative(self, api_key: str = "", dry_run: bool = False) -> dict:
        """open loops + 心情 → DeepSeek 生成主动话题。返回 {"text": ..., "loops": [...]}"""
        loops = self.find_open_loops()
        if not api_key:
            api_key = self._load_api_key()
        if not api_key:
            return {"text": "", "loops": [t for _, t, _, _ in loops], "reason": "no_api_key"}
        mood = "平稳"
        if self.emotion:
            try:
                mood = self.emotion.to_prompt().replace("（你现在的心情：", "").replace("）", "") or "平稳"
            except Exception:
                pass
        now = datetime.now(CN_TZ)
        anchor = ""
        for m, d, name in [(1, 1, "元旦"), (2, 14, "情人节"), (3, 8, "妇女节"), (6, 1, "儿童节"),
                           (9, 10, "教师节"), (10, 1, "国庆节"), (12, 25, "圣诞节")]:
            if now.month == m and now.day == d:
                anchor = f"今天是{name}！"
                break
        loop_lines = "\n".join(f"- [{dom}] {txt[:80]}" for _, txt, dom, _ in loops)
        prompt = f"""你是蠢珞珞，14岁初二女生。现在是{now.strftime('%A')}。
{anchor}

你最近想起的一些事（记忆库自动浮现）：
{loop_lines}

你的心情：{mood}

现在没有人跟你说话，但你心里有点想跟人说说话。请用一句 15-40 字的、自然的、14岁女孩的口吻的话，主动开启话题（像突然想起什么跟人分享一样）。可以关于上面浮现的记忆、或今天的时间感、或你此刻的心情。不要解释，只说那句话。"""
        txt = self._deepseek(prompt, api_key, max_tokens=200)
        if not txt:
            return {"text": "", "loops": [t for _, t, _, _ in loops], "reason": "gen_fail"}
        if not dry_run:
            today = now.strftime("%Y-%m-%d")
            rec = {"time": now.isoformat(timespec="seconds"), "text": txt, "mood": mood, "anchor": anchor,
                   "loops": [t[:60] for _, t, _, _ in loops]}
            with open(self.self_dir / f"initiative_{today}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"text": txt, "loops": [t for _, t, _, _ in loops], "reason": "ok"}

    # ============ 9. 偏好层 (preference) ============
    def build_preferences(self, force: bool = False) -> dict:
        """扫描记忆情绪标签 → 统计喜欢/讨厌 → self/preferences.json"""
        pf = self.self_dir / "preferences.json"
        if pf.exists() and not force:
            try:
                return json.loads(pf.read_text(encoding="utf-8"))
            except Exception:
                pass
        if self.memory is None:
            return {"likes": [], "dislikes": []}
        from collections import Counter
        like_cnt, dislike_cnt = Counter(), Counter()
        for exp_id in self.memory.content.ids():
            exp = self.memory.content.get(exp_id)
            if not exp or not exp.text:
                continue
            dom = (exp.emotion_vector or {}).get("dominant", "neutral")
            themes = [t for kw, t in PREF_THEMES.items() if kw in exp.text]
            if not themes:
                continue
            if dom == "joy":
                for t in themes:
                    like_cnt[t] += 1
            elif dom in ("sadness", "anger", "fear"):
                for t in themes:
                    dislike_cnt[t] += 1
        like_map, dislike_map = dict(like_cnt), dict(dislike_cnt)
        for theme in set(like_map) & set(dislike_map):
            net = like_map[theme] - dislike_map[theme]
            if net > 0:
                dislike_map.pop(theme, None)
                like_map[theme] = net
            elif net < 0:
                like_map.pop(theme, None)
                dislike_map[theme] = -net
            else:
                like_map.pop(theme, None)
                dislike_map.pop(theme, None)
        prefs = {"likes": [{"item": t, "strength": n} for t, n in
                           sorted(like_map.items(), key=lambda x: -x[1])[:8]],
                 "dislikes": [{"item": t, "strength": n} for t, n in
                              sorted(dislike_map.items(), key=lambda x: -x[1])[:8]]}
        pf.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
        self._prefs = prefs
        return prefs

    def preference_line(self) -> str:
        prefs = self._prefs or self.build_preferences()
        likes = "、".join(p["item"] for p in prefs.get("likes", [])[:4])
        dislikes = "、".join(p["item"] for p in prefs.get("dislikes", [])[:4])
        if not likes and not dislikes:
            return ""
        return f"你的偏好：喜欢{likes or '（暂无）'}；不喜欢/怕{dislikes or '（暂无）'}。"

    # ============ 内部工具 ============
    def _load_api_key(self) -> str:
        import os
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            env = self.base / ".env"
            if env.exists():
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.startswith("DEEPSEEK_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
        return key

    def _deepseek(self, prompt: str, key: str, max_tokens=500) -> str:
        import time
        import requests
        for attempt in range(3):
            try:
                r = requests.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "deepseek-chat",
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0.7, "max_tokens": max_tokens},
                    timeout=60)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        return ""

    def _extract_json_list(self, text: str):
        import re
        if not text:
            return None
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    def _existing_memories(self) -> list:
        if self.memory is None:
            return []
        return [exp.text for exp_id in self.memory.content.ids()
                if (exp := self.memory.content.get(exp_id)) and exp.text]

    def _store_memory(self, text: str, source: str = "", emotion_vector: dict = None):
        """入库: 编码+入图+情绪标签 (出生机制完整链路)"""
        if self.memory is None:
            return
        exp_id = "exp_" + uuid.uuid4().hex[:12]
        if emotion_vector is None:
            t = self.tag_memory(text)
            emotion_vector = {"valence": t["valence"], "arousal": t["arousal"], "dominant": t["dominant"]}
        self.memory.store(exp_id=exp_id, text=text, source_url=source, source_year=2026,
                          emotion_vector=emotion_vector)
        try:
            self.memory.save(str(self.base / "phase1" / "memory_luoluo"))
        except Exception:
            pass
        return exp_id

    # ============ 对话挂钩 (engine 调用) ============
    def before_generate(self, humanizer, text: str):
        """generate 内部调用: 偏置+躯体标记+念头+内心流。返回 (bias, thought)"""
        bias = self.bias_for_current_emotion()
        thought = ""
        # 记忆检索已在 _build_messages 做, 躯体标记在那里挂; 念头在此
        try:
            if bias:
                pass  # bias 由 _build_messages 用 emotion_bias 传入
            thought = self.get_thought(humanizer, text) if self._thought_enabled else ""
        except Exception:
            pass
        # V13 内心流: 最近的内心活动作为"她正在想的事" (若没有即时念头)
        if not thought and self.mindstream is not None:
            try:
                lt = self.mindstream.latest_thought(hours=12)
                if lt:
                    thought = lt
            except Exception:
                pass
        return bias, thought

    def after_generate(self):
        """generate 内部调用: 情绪持久化"""
        if self.emotion is not None:
            try:
                self.emotion.save()
            except Exception:
                pass

    @property
    def _thought_enabled(self) -> bool:
        """念头开关: 默认开, 可用环境变量 MIND_THOUGHT=0 关"""
        import os
        return os.environ.get("MIND_THOUGHT", "1") != "0"
