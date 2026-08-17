"""
场景图构建器 v2（按代码审查意见重构）

从真实人类文本自动构建结构化场景图

核心原则：不用AI生成内容。只做确定性的结构化提取。
- 依存句法分析（HanLP）→ 语义角色（v2: 完整 dep_role_map）
- 场景加权计分 + 否定检测（v2: 长词权重大，"没吃/不疼"不触发）
- 医疗咨询过滤器（v2: 改为"模糊时修正"，不覆盖明确日常场景）
- 情感/地点词库提取（v2 新增）
- 置信度：核心槽位缺失/施事=受事惩罚（v2）
"""

from mindsprout.config import BASE

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SceneGraph:
    """一个场景图"""
    exp_id: str
    source_text: str              # 原始人类文本（不改动）
    source_url: str = ""
    source_year: int = 2020
    
    # 结构化提取
    agent: str = ""               # 谁
    action: str = ""              # 做了什么
    recipient: str = ""           # 对谁
    tool: str = ""                # 用什么工具
    location: str = ""            # 在哪
    purpose: str = ""             # 为什么
    outcome: str = ""             # 结果
    emotion: str = ""             # 情感基调
    
    # 场景分类
    scene_type: str = ""          # 预定义场景类型
    parent_branch: str = ""       # 父分支（如"饮食分支"）
    shared_branches: List[str] = field(default_factory=list)
    
    # 元信息
    confidence: float = 0.5       # 自动提取的置信度
    human_verified: bool = False  # 是否经过人工验证


class SceneGraphBuilder:
    """
    从真实人类文本中自动构建场景图（v2）
    """

    # ============================================================
    # 场景触发器（v2: 关键词按专指度加权，单字1分/双字2分/三字+3分）
    # ============================================================
    SCENE_TRIGGERS = {
        "feeding": {
            "keywords": ["喂奶", "喂饭", "喝奶", "吃奶", "喂水", "奶瓶", "奶粉", "辅食", "喂", "吃奶瓶"],
            "parent_branch": "饮食分支",
            "slots": ["agent", "action", "recipient", "tool", "purpose", "emotion"],
        },
        "communicating": {
            "keywords": ["打电话", "发消息", "说话", "喊", "告诉", "问", "回", "聊", "微信", "说"],
            "parent_branch": "通讯分支",
            "slots": ["agent", "action", "recipient", "tool", "purpose"],
        },
        "learning": {
            "keywords": ["学习", "练习", "训练", "上课", "写作业", "学会", "考试", "背书", "第一次", "学"],
            "parent_branch": "成长分支",
            "slots": ["agent", "action", "recipient", "skill", "method", "difficulty"],
        },
        "caring": {
            "keywords": ["照顾", "抱", "哄", "安慰", "担心", "心疼", "陪", "守护", "照顾孩子"],
            "parent_branch": "情感分支",
            "slots": ["agent", "action", "recipient", "emotion", "outcome"],
        },
        "playing": {
            "keywords": ["玩游戏", "玩具", "跑步", "跑来跑去", "捉迷藏", "跳绳", "踢球", "玩", "游戏"],
            "parent_branch": "社交分支",
            "slots": ["agent", "action", "recipient", "tool", "location", "emotion"],
        },
        "conflict": {
            "keywords": ["吵架", "打架", "打起来", "打人", "发脾气", "骂", "哭闹", "委屈", "生气", "不理"],
            "parent_branch": "社交分支",
            "slots": ["agent", "action", "recipient", "cause", "outcome", "emotion"],
        },
        "comforting": {
            "keywords": ["别哭", "不怕", "别难过", "抱着", "摸头", "乖", "哄哄"],
            "parent_branch": "情感分支",
            "slots": ["agent", "action", "recipient", "emotion", "outcome"],
        },
        "discovering": {
            "keywords": ["发现", "第一次见", "好奇", "摸", "看看", "打开", "拆开", "观察"],
            "parent_branch": "认知分支",
            "slots": ["agent", "action", "object", "location", "emotion"],
        },
        "sleeping": {
            "keywords": ["睡觉", "失眠", "睡不着", "熬夜", "做梦", "困", "躺", "眯", "休息", "睡眠"],
            "parent_branch": "健康分支",
            "slots": ["agent", "action", "location", "cause", "outcome", "emotion"],
        },
        "working": {
            "keywords": ["上班", "工作", "辞职", "加班", "工资", "同事", "领导", "老板", "面试", "出差", "打工"],
            "parent_branch": "事业分支",
            "slots": ["agent", "action", "recipient", "location", "cause", "outcome", "emotion"],
        },
        "dating": {
            "keywords": ["分手", "挽回", "相亲", "男朋友", "女朋友", "恋爱", "前男友", "前女友", "结婚", "对象", "暧昧", "表白"],
            "parent_branch": "情感分支",
            "slots": ["agent", "action", "recipient", "cause", "outcome", "emotion"],
        },
    }

    # ============================================================
    # 医疗咨询/症状检测（v2: 加权 + 否定检测 + 不覆盖明确场景）
    # ============================================================
    CONSULT_WORDS = {"请问": 2, "咨询": 2, "症状": 2, "治疗": 2, "医院": 1, "医生": 2, "中医": 2,
                     "检查": 1, "诊断": 2, "门诊": 1, "调理": 2, "用药": 2, "治愈": 2, "康复": 1,
                     "吃药": 2, "就诊": 2, "求医": 2}
    CONSULT_STRONG = {"气血": 3, "肝经": 3, "湿气": 3, "肾虚": 3, "体虚": 3, "上火": 3, "宫寒": 3, "虚火": 3}
    CONSULT_PATTERNS = [
        re.compile(r"(如何|怎么|该).{0,6}(治|调理|吃|用|办)"),
        re.compile(r"想咨询|想请问|想问一下"),
        re.compile(r"需要.{0,4}(治疗|就医|吃药|住院)"),
    ]
    BODY_PARTS = ["头", "脸", "眼", "鼻", "嘴", "牙", "脖子", "肩膀", "背", "腰", "肚子", "胃", "腿", "脚",
                  "手", "皮肤", "舌", "肝", "肾", "脾", "肺", "尿", "大便", "肛门", "心", "全身", "身体"]
    SYMPTOM_WORDS = {"疼": 1, "痛": 1, "肿": 1, "痒": 1, "烧": 1, "酸": 1, "胀": 1, "麻": 1, "血": 1,
                     "痰": 1, "咳": 1, "感冒": 2, "发炎": 2, "过敏": 2, "难受": 2, "不舒服": 2, "怕冷": 2,
                     "乏力": 2, "无力": 2, "腹泻": 2, "便秘": 2, "呕吐": 2, "头晕": 2, "恶心": 2,
                     "出冷汗": 2, "心跳": 2, "冰凉": 2, "压力": 1}

    # 否定前缀（"没吃""不疼"不触发场景/症状）
    NEG_PREFIX = re.compile(r"(没|不|没有|无|别|不用|不要|并非|毫不)$")

    # 情感词库（v2 新增）
    EMOTION_LEXICON = {"开心": 0.8, "高兴": 0.7, "快乐": 0.7, "幸福": 0.7, "兴奋": 0.6, "期待": 0.5,
                       "难过": 0.6, "伤心": 0.6, "委屈": 0.5, "生气": 0.5, "愤怒": 0.6, "烦": 0.4,
                       "焦虑": 0.5, "担心": 0.4, "害怕": 0.5, "恐惧": 0.6, "紧张": 0.4, "失望": 0.5,
                       "崩溃": 0.6, "哭": 0.4, "累": 0.3, "疲惫": 0.4, "无奈": 0.4, "郁闷": 0.5,
                       "纠结": 0.4, "愧疚": 0.5, "后悔": 0.5, "尴尬": 0.4, "自豪": 0.6, "欣慰": 0.5}

    # 地点词表（v2 新增）
    LOCATION_LEXICON = ["家", "学校", "医院", "公园", "床上", "公司", "教室", "办公室", "路上", "厨房",
                        "客厅", "厕所", "外面", "操场", "图书馆", "楼上", "楼下", "小区", "超市", "菜市场",
                        "幼儿园", "商场", "店里", "老家", "单位", "床上", "卧室"]

    # HanLP 依存角色映射（v2: 完整映射 + 前缀匹配）
    DEP_ROLE_MAP = [
        ("nsubj", "agent"), ("top", "agent"),
        ("dobj", "recipient"), ("iobj", "recipient"), ("nsubjpass", "recipient"),
        ("nmod:tool", "tool"), ("nmod:inst", "tool"),
        ("nmod:loc", "location"), ("nmod:place", "location"), ("advmod:loc", "location"),
        ("nmod:tmod", "location"), ("advcl:loc", "location"),
        ("advcl:cause", "purpose"), ("advmod:cause", "purpose"), ("nmod:reason", "purpose"),
        ("ccomp", "action"), ("xcomp", "action"), ("advmod:degree", "action"),
    ]

    # 正则角色模式（fallback，v2: 补充 location/emotion/cause）
    ROLE_PATTERNS = {
        "agent": [
            (r'^(.{1,5})(给|帮|为|对|教|让|叫)', 1),
            (r'(.{1,5})(用|拿|端|抱)', 1),
        ],
        "recipient": [
            (r'(给|对|向|帮)(.{1,8})(喂|说|教|打|发)', 2),
        ],
        "tool": [
            (r'(用|拿)(.{1,8})(给|喂|打|发|做|吃)', 2),
        ],
        "purpose": [
            (r'(为了|想|要|让)(.{2,15})(，|。|；)', 2),
            (r'因为(.{2,12})[，,。]', 1),
        ],
        "location": [
            (r'在(.{1,6})(?:做|玩|睡|写|吃|学|上班|休息)', 1),
        ],
    }

    # ============================================================
    # 初始化（v2: HanLP 单例懒加载）
    # ============================================================
    _nlp_cache = None  # 类级缓存，避免重复加载

    @classmethod
    def _get_nlp(cls):
        if cls._nlp_cache is None:
            try:
                import hanlp
                cls._nlp_cache = hanlp.load(hanlp.pretrained.dep.CTD9_DEP_ELECTRA_SMALL)
            except ImportError:
                print("⚠️ HanLP未安装，使用正则匹配模式。pip install hanlp")
                cls._nlp_cache = False
        return cls._nlp_cache or None

    def __init__(self, use_hanlp: bool = True):
        self.use_hanlp = use_hanlp and (self._get_nlp() is not None)
        self._nlp = self._get_nlp() if self.use_hanlp else None
        self.last_stats = {}

    # ============================================================
    # 核心构建
    # ============================================================
    def build(self, text: str, **kwargs) -> Optional[SceneGraph]:
        """从一段真实人类文本构建场景图（返回 None = 无场景/低置信度）"""
        scene_type, config = self._identify_scene(text)
        if scene_type is None:
            return None

        slots = self._extract_roles(text, config["slots"])
        conf = self._estimate_confidence(slots, config["slots"])

        sg = SceneGraph(
            exp_id=kwargs.get("exp_id", f"sg_{abs(hash(text)) % 100000:05d}"),
            source_text=text,
            source_url=kwargs.get("source_url", ""),
            source_year=kwargs.get("source_year", 2020),
            agent=slots.get("agent", ""),
            action=slots.get("action", ""),
            recipient=slots.get("recipient", ""),
            tool=slots.get("tool", ""),
            location=slots.get("location", ""),
            purpose=slots.get("purpose", ""),
            outcome=slots.get("outcome", ""),
            emotion=slots.get("emotion", ""),
            scene_type=scene_type,
            parent_branch=config["parent_branch"],
            shared_branches=kwargs.get("shared_branches", []),
            confidence=conf,
        )
        return sg

    # ============================================================
    # 场景识别（v2: 加权计分 + 否定检测 + 医疗修正）
    # ============================================================
    def _kw_score(self, text: str, kw: str) -> int:
        """关键词加权分（双字2/三字+3/单字1），带否定检测"""
        if len(kw) == 1 and not re.search(r"[\u4e00-\u9fff]", kw):
            return 0
        base = 2 if len(kw) == 2 else (3 if len(kw) >= 3 else 1)
        count = 0
        for m in re.finditer(re.escape(kw), text):
            pre = text[max(0, m.start() - 2):m.start()]
            if self.NEG_PREFIX.search(pre):
                continue  # "没吃""不疼" 不触发
            count += 1
        return base * min(count, 3)

    def _scene_scores(self, text: str) -> Dict[str, int]:
        scores = {}
        for scene_type, config in self.SCENE_TRIGGERS.items():
            s = sum(self._kw_score(text, kw) for kw in config["keywords"])
            if s > 0:
                scores[scene_type] = s
        return scores

    def _consult_score(self, text: str) -> int:
        s = sum(v for w, v in self.CONSULT_WORDS.items() if w in text)
        s += sum(v for w, v in self.CONSULT_STRONG.items() if w in text)
        for p in self.CONSULT_PATTERNS:
            if p.search(text):
                s += 2
        return s

    def _symptom_score(self, text: str) -> Tuple[int, int]:
        """返回 (body_count, symptom_score)，带否定检测"""
        body = 0
        for b in self.BODY_PARTS:
            for m in re.finditer(re.escape(b), text):
                pre = text[max(0, m.start() - 2):m.start()]
                if not self.NEG_PREFIX.search(pre):
                    body += 1
                    break
        sym = 0
        for w, v in self.SYMPTOM_WORDS.items():
            for m in re.finditer(re.escape(w), text):
                pre = text[max(0, m.start() - 2):m.start()]
                if not self.NEG_PREFIX.search(pre):
                    sym += v
                    break
        return body, sym

    def _identify_scene(self, text: str) -> Tuple[Optional[str], Optional[Dict]]:
        """v2: 加权计分 + 医疗修正（不覆盖明确场景）"""
        scene_scores = self._scene_scores(text)
        consult = self._consult_score(text)
        body, sym = self._symptom_score(text)

        best = max(scene_scores, key=scene_scores.get) if scene_scores else None
        best_score = scene_scores.get(best, 0) if best else 0

        # 医疗证据极强（问诊>=6）：即使场景分高也归咨询（如"请问医生宝宝便秘怎么治"）
        if consult >= 6:
            return "communicating", self.SCENE_TRIGGERS["communicating"]

        # 医疗证据中等（问诊>=4 或 症状明显）且场景证据弱 → 咨询
        if (consult >= 4 or (body >= 1 and sym >= 3)) and best_score < 4:
            return "communicating", self.SCENE_TRIGGERS["communicating"]

        if not scene_scores:
            return None, None
        return best, self.SCENE_TRIGGERS[best]

    # ============================================================
    # 角色提取（v2: emotion/location 词库 + 完整 dep 映射）
    # ============================================================
    def _extract_roles(self, text: str, slot_names: List[str]) -> Dict[str, str]:
        slots = {}
        if self.use_hanlp and self._nlp:
            slots = self._extract_with_hanlp(text, slot_names)
        else:
            slots = self._extract_with_regex(text, slot_names)

        # v2: 情感/地点词库提取
        if "emotion" in slot_names and not slots.get("emotion"):
            e = self._extract_emotion(text)
            if e:
                slots["emotion"] = e
        if "location" in slot_names and not slots.get("location"):
            loc = self._extract_location(text)
            if loc:
                slots["location"] = loc

        # 补充：用文本本身的前几个字填充空slot
        for name in slot_names:
            if name not in slots or not slots[name]:
                slots[name] = f"[从文本推断]"
        return slots

    def _extract_with_hanlp(self, text: str, slot_names: List[str]) -> Dict[str, str]:
        try:
            parsed = self._nlp(text)
            slots = {}
            tokens = parsed.get("token", [])
            deps = parsed.get("dep", [])
            for i, (token, dep) in enumerate(zip(tokens, deps)):
                for prefix, role in self.DEP_ROLE_MAP:
                    if dep.startswith(prefix) and role in slot_names and role not in slots:
                        slots[role] = token
                        break
            return slots
        except Exception as e:
            print(f"HanLP解析失败: {e}，回退到正则模式")
            return self._extract_with_regex(text, slot_names)

    def _extract_with_regex(self, text: str, slot_names: List[str]) -> Dict[str, str]:
        slots = {}
        for role in slot_names:
            if role in self.ROLE_PATTERNS:
                for pattern, group in self.ROLE_PATTERNS[role]:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            slots[role] = match.group(group)
                        except IndexError:
                            pass
                        break
        return slots

    def _extract_emotion(self, text: str) -> str:
        """情感词库匹配 → 返回情感词（最高强度）"""
        best_word, best_val = "", 0.0
        for w, v in self.EMOTION_LEXICON.items():
            if w in text and v > best_val:
                best_word, best_val = w, v
        return best_word

    def _extract_location(self, text: str) -> str:
        """地点词表匹配 → 第一个命中"""
        for loc in self.LOCATION_LEXICON:
            if loc in text:
                return loc
        return ""

    # ============================================================
    # 置信度（v2: 核心槽位 + 施事=受事惩罚）
    # ============================================================
    CORE_SLOTS = ["agent", "action"]

    def _estimate_confidence(self, slots: Dict, expected: List[str]) -> float:
        filled = sum(1 for s in expected if slots.get(s) and "[从文本推断]" not in slots.get(s, ""))
        conf = filled / max(1, len(expected))

        # 核心槽位缺失 → 惩罚（v2: 0.15/个，避免过度悲观）
        core_missing = sum(1 for s in self.CORE_SLOTS if s in expected
                           and (not slots.get(s) or "[从文本推断]" in slots.get(s, "")))
        if core_missing:
            conf *= (1 - 0.15 * core_missing)

        # 施事=受事 → 可疑，惩罚
        a, r = slots.get("agent"), slots.get("recipient")
        if a and r and a == r and "[从文本推断]" not in a:
            conf *= 0.5

        return min(0.9, max(0.2, conf))

    # ============================================================
    # 批量构建（v2: 分开统计 no_scene / low_conf）
    # ============================================================
    def build_batch(
        self,
        texts: List[Dict],
        min_confidence: float = 0.4,
    ) -> List[SceneGraph]:
        """批量构建场景图（v2 算法已含核心槽位惩罚，0.4 阈值合理）"""
        graphs, no_scene, low_conf = [], 0, 0
        for item in texts:
            sg = self.build(
                item["text"],
                source_url=item.get("url", ""),
                source_year=item.get("year", 2020),
            )
            if sg is None:
                no_scene += 1
            elif sg.confidence >= min_confidence:
                graphs.append(sg)
            else:
                low_conf += 1
        self.last_stats = {
            "total": len(texts), "built": len(graphs),
            "no_scene": no_scene, "low_conf": low_conf,
        }
        print(f"✅ 构建 {len(graphs)} 个场景图 | 无法识别 {no_scene} | 低置信度 {low_conf} | 共 {len(texts)}")
        return graphs

    def export_for_review(self, graphs: List[SceneGraph], output_path: str):
        """导出为人工审查格式"""
        review_data = []
        for sg in graphs:
            review_data.append({
                "exp_id": sg.exp_id,
                "text": sg.source_text,
                "scene_type": sg.scene_type,
                "agent": sg.agent,
                "action": sg.action,
                "recipient": sg.recipient,
                "tool": sg.tool,
                "emotion": sg.emotion,
                "confidence": sg.confidence,
                "verified": sg.human_verified,
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)
        print(f"📋 {len(review_data)} 条待审查 → {output_path}")
