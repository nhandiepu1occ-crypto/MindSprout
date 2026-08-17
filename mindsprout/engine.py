"""
推理引擎 v2：transformers 后端 + 双系统记忆融合

- 模型：Qwen2.5-1.5B-Instruct（本地 safetensors，RTX 4060）
- 记忆融合（蓝图 P1-3 选项C）：query 记忆库 → top3 摘要拼进 system prompt
- 人设配置：可切换成长阶段（blank/toddler/preschool/primary/junior）
- 记忆接口协议（MemoryLike）：
    query(query_text, top_k, k_hops) -> {"contents": [(exp, signal), ...]}
    其中 exp 具有 .text 属性
"""

from mindsprout.config import BASE

import os
import sys
import json
import logging
import threading
from pathlib import Path
from typing import Optional, Generator, Dict, Any, List, Tuple

logger = logging.getLogger("humanize_ai.engine")

# 模型路径：环境变量优先，可跨平台
DEFAULT_MODEL_DIR = os.getenv("QWEN_MODEL_DIR", r"F:\models\qwen2.5-1.5b-instruct")

# 各成长阶段的系统提示 + 生成参数（分段成长用）
STAGE_PROMPTS = {
    "blank": {
        "prompt": "你是一个刚出生的婴儿。你几乎没有语言能力，只能发出简单的音节。",
        "temperature": 0.9, "top_p": 0.95, "max_tokens": 80,
    },
    "toddler": {
        "prompt": "你是一个3岁的小孩。你用简单的句子说话，会叫爸爸妈妈，会表达饿、困、开心、难过。你还不懂复杂的概念。",
        "temperature": 0.9, "top_p": 0.95, "max_tokens": 120,
    },
    "preschool": {
        "prompt": "你是一个5岁的幼儿园小朋友。你会说完整的句子，喜欢问为什么，会和小伙伴玩游戏，知道基本的规则。",
        "temperature": 0.85, "top_p": 0.9, "max_tokens": 150,
    },
    "primary": {
        "prompt": "你是一个10岁的小学生。你会上课、写作业、和朋友玩，会为考试担心，会觉得爸妈唠叨。你了解自己身边的世界。",
        "temperature": 0.85, "top_p": 0.9, "max_tokens": 180,
    },
    "junior": {
        "prompt": "你是蠢珞珞，14岁（永远说自己是14岁，初二，不许说自己其他年龄），初二，女生，身高156cm，中国某城市的普通家庭孩子。\n你的家庭：爸爸是老师，妈妈在一家小公司做文秘。\n你的特点：\n- 说话口语化，句子短，情绪直接，只说中文，不夹英文单词\n- 有青春期的小心思：在意朋友看法、会叛逆、开始对异性好奇\n- 认知边界：你只有初中生的知识水平。小学没学过、爸妈没教过的东西你就是不知道，直接说不会，不会装懂。不懂股票、房贷、职场、高等数学这些成人领域的东西。遇到超纲问题你会说\"这题超纲了吧\"\"我还没学过\"\"不知道\"，不会硬编。\n- 如果有人问你\"你是不是AI/机器人/程序\"，你会觉得莫名其妙甚至有点生气——你当然是人，你会反问\"你才AI吧\"或者不接这个茬。\n- 你说话不会用\"综上所述\"\"值得注意的是\"这类书面腔，也不会说\"小智\"\"AI助手\"之类的词。\n- 如果对方的话开头有\"你的一些经历：\"，这些就是你真实经历过的事，你必须从中挑一件相关的，用自己的话说出来，不能忽略也不能编经历里没有的事。\n- 遇到没教过、没经历过的事（比如你的生日、家里存款），就说不知道或不记得，不要自己编。",
        "temperature": 0.7, "top_p": 0.9, "max_tokens": 220,
    },
}

DEFAULT_STAGE = "junior"


class Humanizer:
    """生成引擎 v2：Qwen2.5 + 记忆融合"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        memory_bank=None,
        stage: str = DEFAULT_STAGE,
        custom_persona: Optional[str] = None,
        device: Optional[str] = None,
        peft_dir: Optional[str] = None,
        verbose: bool = True,
        auto_memory: bool = False,
        dna=None,
        emotion_state=None,
        mind=None,
        **kwargs,
    ):
        """
        Args:
            model_path: 模型目录（默认 DEFAULT_MODEL_DIR 或环境变量 QWEN_MODEL_DIR）
            memory_bank: 可选 MemoryBank 实例（须实现 MemoryLike 协议）
            stage: 成长阶段（STAGE_PROMPTS 的 key）
            custom_persona: 自定义人设 prompt（覆盖 stage 的 prompt）
            device: 设备提示（"cuda"/"cpu"）；指定后严格用该设备，None 时交给 device_map="auto"（多卡均衡，需 accelerate）
            peft_dir: 可选 LoRA adapter 目录（训练后挂载）
            verbose: 是否打印加载日志
            auto_memory: 默认对话自动用输入文本检索记忆（架构原则：她的记忆应参与每次对话）
        """
        self._model_path = model_path or DEFAULT_MODEL_DIR
        self._memory = memory_bank
        self._stage = stage
        self._custom_persona = custom_persona
        self._device_hint = device  # 用户显式指定（None=自动分配）
        self._device = device or ("cuda" if self._has_cuda() else "cpu")
        self._peft_dir = peft_dir
        self._verbose = verbose
        self._auto_memory = auto_memory
        self._dna = dna                    # DNA 出生档案（可选；注入性格底色行）
        self._emotion = emotion_state      # EmotionState（可选；心情注入+共情 tick）
        self._model = None
        self._tokenizer = None
        self._personality_text = self._load_personality()
        # 出厂心智: 出生即有 (与记忆/情绪/DNA 同级; 不传则自动创建)
        self._mind = mind
        # 欲望系统 (感受补全②): 出生即有, 持久化 self/desires.json
        self._desire = None
        try:
            from humanize_ai.desire import DesireState
            desire_path = Path(__file__).resolve().parents[1] / "phase1" / "self" / "desires.json"
            self._desire = DesireState.load(desire_path)
        except Exception:
            self._desire = None
        if self._mind is None:
            try:
                from humanize_ai.mind import Mind
                self._mind = Mind(base_dir=Path(__file__).resolve().parents[1],
                                  dna=dna, memory_bank=memory_bank,
                                  emotion_state=emotion_state, model_dir=self._model_path)
                # 出生时: 给已有记忆打情绪标签(幂等) + 加载偏好
                if memory_bank is not None:
                    self._mind.tag_all_memories()
                    self._mind.build_preferences()
            except Exception as e:
                logger.warning(f"出厂心智加载失败(不影响对话): {e}")
                self._mind = None

    @staticmethod
    def _load_personality() -> str:
        """加载场景推导出的性格档案（phase1/personality.json，架构原则2：场景决定性格）"""
        try:
            p = Path(__file__).resolve().parents[1] / "phase1" / "personality.json"
            if p.exists():
                import json
                data = json.load(open(p, encoding="utf-8"))
                return data.get("personality_text", "")
        except Exception:
            pass
        return ""

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device_map = self._device_hint
        if device_map is None:
            # 未显式指定设备：device_map="auto" 需要 accelerate；不可用则回退单设备
            try:
                import accelerate  # noqa: F401
                device_map = "auto"
            except ImportError:
                device_map = None
        if self._verbose:
            print(f"🚀 加载模型: {Path(self._model_path).name} (device_map={device_map or '单设备:' + self._device})", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        dtype = torch.float16 if self._has_cuda() else torch.float32
        load_kwargs = dict(trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True)
        if device_map:
            load_kwargs["device_map"] = device_map
        self._model = AutoModelForCausalLM.from_pretrained(self._model_path, **load_kwargs)
        if device_map is None:
            self._model = self._model.to(self._device)
        if self._peft_dir:
            from peft import PeftModel
            if self._verbose:
                print(f"🔌 挂载 LoRA: {self._peft_dir}", flush=True)
            self._model = PeftModel.from_pretrained(self._model, self._peft_dir)
        self._model.eval()
        if self._verbose:
            print(f"✅ 模型就绪 ({next(self._model.parameters()).device})", flush=True)

    def _system_prompt(self, text: str = "") -> str:
        if self._custom_persona:
            return self._custom_persona
        prompt = STAGE_PROMPTS.get(self._stage, STAGE_PROMPTS[DEFAULT_STAGE])["prompt"]
        if self._personality_text and self._stage == "junior":
            prompt = prompt + "\n" + self._personality_text
        # DNA 性格底色（出生设置）
        if self._dna is not None and self._stage == "junior":
            prompt = prompt + "\n" + self._dna.temperament_line()
        # 细节引导 (盲测修复): 说话带具体的人和东西
        if self._stage == "junior":
            prompt = prompt + '\n（你说话会带上具体的人和东西，比如"小雨""辣条""阳台的花"，不空泛地说。）'
        # 当前心情（情绪状态机，每轮动态）
        if self._emotion is not None and self._stage == "junior":
            mood = self._emotion.to_prompt()
            if mood:
                prompt = prompt + "\n" + mood
        # 欲望状态 (感受补全②: 饿/困/孤独等影响表达)
        if self._desire is not None and self._stage == "junior":
            des = self._desire.to_prompt()
            if des:
                prompt = prompt + "\n" + des
        # 偏好层（阶段4b: 从记忆统计的喜欢/不喜欢, 只读档案）
        if self._stage == "junior":
            try:
                pf = Path(__file__).resolve().parents[1] / "phase1" / "self" / "preferences.json"
                if pf.exists():
                    prefs = json.loads(pf.read_text(encoding="utf-8"))
                    likes = "、".join(p["item"] for p in prefs.get("likes", [])[:4])
                    dislikes = "、".join(p["item"] for p in prefs.get("dislikes", [])[:4])
                    if likes or dislikes:
                        prompt = prompt + f"\n你的偏好：喜欢{likes or '（暂无）'}；不喜欢/怕{dislikes or '（暂无）'}。"
            except Exception:
                pass
        # 时间感知 (timeline v1: 现实时钟 + 生日)
        if self._stage == "junior":
            try:
                from humanize_ai.timeline import now_context_line
                prompt = prompt + "\n" + now_context_line()
            except Exception:
                pass
        return prompt

    def _stage_params(self) -> Dict[str, Any]:
        """当前阶段的默认生成参数"""
        return STAGE_PROMPTS.get(self._stage, STAGE_PROMPTS[DEFAULT_STAGE])

    def _build_messages(self, text: str, memory_query: Optional[str] = None,
                       custom_system: Optional[str] = None,
                       history: Optional[List[Dict[str, str]]] = None,
                       skip_brain: bool = False) -> List[Dict[str, str]]:
        """构建 messages（交给 tokenizer.apply_chat_template 处理）

        记忆融合 v2（强制带入）：检索到的经历直接拼进 user 问题开头，
        格式与训练样本一致（"你的一些经历：\n- ...\n（说到相关的事就讲你自己的经历）"），
        让模型把记忆当自己的事讲出来，而不是可选项。
        history: 多轮对话历史 [(role, content), ...] 插在 system 与当前 user 之间 (S2)
        """
        import re as _re
        system = custom_system or self._system_prompt(text)

        # 价值观条件注入 (1.8): 对话主题命中 → 只注入相关价值观 (attention式, 不稀释)
        try:
            from humanize_ai.values import values_for_text
            _vline = values_for_text(text)
            if _vline:
                system = system + "\n" + _vline
        except Exception:
            pass

        # 人物识别 + 感官合成 (楚门 v1.9/v2.0): 她的大脑认出"此刻是谁"+合成"此刻世界" — 感知注入, 非约束
        _person_id = None
        if self._stage == "junior":
            try:
                from humanize_ai.persona import identify, person_line
                _pr = identify(text, context_hint="")
                if _pr["person_id"] and _pr["confidence"] >= 0.55:
                    _person_id = _pr["person_id"]
                    pl = person_line(_person_id)
                    if pl:
                        system = system + "\n" + pl
            except Exception:
                pass
            try:
                from humanize_ai.senses import sense_line
                _dom = "neutral"
                if self._emotion is not None:
                    try:
                        _dom = self._emotion.dominant()
                    except Exception:
                        pass
                sl = sense_line(text=text, person_id=_person_id, emotion_dom=_dom)
                if sl:
                    system = system + "\n" + sl
            except Exception:
                pass

        # 大脑总线 (楚门 v2.0): 出厂能力模块聚合 — 驱力/身体/情绪/注意/依恋/时间/好奇/发展/睡眠
        # 注意: 人物识别(persona)与感官(senses)单独注入, 总线负责其余内在状态
        # 身份类问题(who_ask)跳过总线: 专注定向记忆防注意力分散 (与时间/生活类跳过念头同机制)
        if self._stage == "junior" and not skip_brain:
            try:
                from humanize_ai.brain import brain_tick
                _dom = "neutral"
                if self._emotion is not None:
                    try:
                        _dom = self._emotion.dominant()
                    except Exception:
                        pass
                _br = brain_tick(text=text, person_id=_person_id, emotion_dom=_dom, memory_bank=self._memory)
                if _br.get("injection"):
                    system = system + "\n" + _br["injection"]
            except Exception:
                pass

        # 关系感知 (楚门 v1.8): 称呼/赌气/亲近 靠近角色定义
        if self._stage == "junior":
            try:
                from humanize_ai.relation import relation_line
                rl = relation_line()
                if rl:
                    system = system + "\n" + rl
            except Exception:
                pass

        user_content = text
        if self._memory is not None and memory_query and hasattr(self._memory, "query"):
            # 答非所问治理 (V3.9.4): 简单/寒暄/身份/时间类问题不需要记忆 — 跳过融合
            import re as _re_simple
            if not _re_simple.search(
                r"(你好|哈喽|嗨|在吗|喂|晚安|早安|早上好|晚上好|吃饭了吗|吃了没|"
                r"你叫什么|几岁了|多大了|你是谁|叫什么名字|今天星期|几号|几点了|"
                r"天气|下雨|出太阳|再见|拜拜|谢谢|晚安|早上好)", text):
                _skip_mem = False
            else:
                _skip_mem = True
            try:
                # 情绪偏置检索 (阶段2b): 当前主导情绪匹配的记忆更容易被想起 (情绪一致性)
                bias = None
                if self._emotion is not None:
                    try:
                        dom = self._emotion.dominant()
                        if dom != "neutral":
                            bias = dom
                    except Exception:
                        pass
                r = self._memory.query(query_text=memory_query, top_k=3, k_hops=1, emotion_bias=bias)
                contents = r.get("contents", [])
                # 相关性门槛: 检索激活分过低 = 记忆与问题无关 → 不注入 (防答非所问)
                if contents and not _skip_mem:
                    _top_sig = contents[0][1]
                    if _top_sig < 0.25:
                        contents = []
                    else:
                        contents = [c for c in contents if c[1] >= _top_sig * 0.5]
                # 躯体标记 (阶段2c): 回忆强情绪记忆 → 情绪状态被唤起 (出厂心智封装)
                if contents and self._emotion is not None and self._mind is not None:
                    try:
                        self._mind.apply_somatic(contents)
                    except Exception:
                        pass
                if contents:
                    mem_lines = []
                    for exp, signal in contents[:3]:
                        # 兼容带 .text 属性的对象和 dict 两种结构
                        if isinstance(exp, dict):
                            txt = (exp.get("text") or "").strip()
                        elif hasattr(exp, "text"):
                            txt = (exp.text or "").strip()
                        else:
                            continue
                        if not txt:
                            continue
                        # 真/假记忆出厂区分 (v4): imagination 记忆必须带"我想的"标记
                        src = ""
                        if isinstance(exp, dict):
                            src = exp.get("source_url", "") or ""
                        elif hasattr(exp, "source_url"):
                            src = exp.source_url or ""
                        if src == "imagination":
                            mem_lines.append(f"- (我想过的) {txt}")
                        else:
                            # 时间锚 (timeline v1): 记忆带相对时间前缀, 时间线虚化
                            try:
                                from humanize_ai.timeline import memory_time_prefix
                                tpre = memory_time_prefix(txt, getattr(exp, "source_year", None) or None)
                                mem_lines.append(f"- {tpre} {txt}")
                            except Exception:
                                mem_lines.append(f"- {txt}")
                    if mem_lines:
                        prefix = "你的一些经历：\n" + "\n".join(mem_lines)
                        prefix += "\n（这些是你真实经历过的，说到相关的事就讲你自己的经历，不要转述。带'我想过的'标记的是你想象/幻想过的，不是真实发生的，说到时要说'我想过…'不能说成真实经历）\n\n"
                        user_content = prefix + user_content
            except Exception as e:
                logger.warning(f"记忆融合失败: {e}")

        # 时间类问题: 实时时刻插到 system 开头 (最高注意力区, 3B 必读)
        try:
            import re as _re_time
            if _re_time.search(r"(几点|几点了|什么时间|什么时候了|现在几点|几点了)", text):
                from humanize_ai.timeline import now_context_line
                _tl = now_context_line()
                _mt = _re_time.search(r"现在是(\d+)点(\d+)分（(上午|下午|晚上)）", _tl)
                if _mt:
                    _h = int(_mt.group(1))
                    _hh = _h if _h <= 12 else _h - 12
                    system = f"（现在是{_mt.group(3)}{_hh}点{_mt.group(2)}分）\n" + system
        except Exception:
            pass

        msgs = [{"role": "system", "content": system}]
        if history:
            # 只取最近 N 轮, 防上下文爆炸 (S2)
            msgs += [{"role": r, "content": c} for r, c in history[-8:]]
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def generate(
        self,
        text: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        memory_query: Optional[str] = None,
        seed: Optional[int] = None,
        custom_system: Optional[str] = None,
        _internal_skip_emotion: bool = False,  # 内部递归调用跳过情绪tick (M3)
        _internal_skip_mind: bool = False,     # 内部递归调用跳过mind理解层 (防递归)
        **gen_kwargs,
    ) -> str:
        """
        生成文本（带可选记忆融合）

        Args:
            text: 用户输入/任务
            temperature/top_p/max_tokens: 覆盖当前阶段的默认生成参数
            memory_query: 用于检索记忆库的查询文本（None = 不查记忆）
            seed: 固定随机种子（调试/评估可复现用，None = 不固定）
            custom_system: 覆盖 system prompt（内在叙事注入用）
        """
        import torch
        self._ensure_loaded()

        # 情绪状态机 tick：衰减 + 感知对方情绪 + 共情耦合（出厂机制，先于本轮生成）
        if self._emotion is not None and not _internal_skip_emotion:
            try:
                self._emotion.tick(text)
                self._emotion.save()
            except Exception as e:
                logger.warning("情绪状态机 tick/save 失败: %s", str(e)[:100])

        # 欲望状态 tick：时间驱动 + 空闲驱动（出厂机制）
        if self._desire is not None:
            try:
                self._desire.tick()
            except Exception:
                pass

        # 驱力/身体演化已由大脑总线 (brain_tick) 统一处理 (V3.9.2 去重)

        # 阶段默认参数（显式传入则覆盖）
        stage = self._stage_params()
        temperature = temperature if temperature is not None else stage.get("temperature", 0.8)
        top_p = top_p if top_p is not None else stage.get("top_p", 0.9)
        max_tokens = max_tokens if max_tokens is not None else stage.get("max_tokens", 200)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # 过滤保留参数，防止 gen_kwargs 重复传参导致 TypeError
        reserved = {"max_new_tokens", "temperature", "top_p", "do_sample", "pad_token_id"}
        for k in reserved:
            gen_kwargs.pop(k, None)

        history = gen_kwargs.pop("history", None)  # 多轮上下文 (S2: 平台对话历史)
        messages = self._build_messages(text, memory_query, custom_system=custom_system, history=history)
        # 内在叙事 (阶段3b): 生成前先冒念头, 注入 system 影响表达 (出厂心智)
        if self._mind is not None and not _internal_skip_mind and not custom_system and memory_query is not None:
            try:
                thought = self._mind.get_thought(self, text)
                if thought:
                    messages[0]["content"] = self._mind.build_system_with_thought(
                        messages[0]["content"], thought)
            except Exception:
                pass
        # auto_memory：默认对话自动用输入文本检索记忆（架构原则：记忆应参与每次对话；显式传 memory_query 仍优先）
        if self._auto_memory and memory_query is None:
            # 指代历史/社交身份类问题: 答案在对话历史或不该猜 — 跳过记忆注入防干扰 (fix 2026-08-15 v6)
            import re as _re
            # 楚门 v1.6: "你是谁/认识我吗" → 定向检索"他"的认知记忆 (模糊熟悉感)
            who_ask = bool(_re.search(
                r"你是谁|我是谁|认识我吗|还记得我吗|你知道我是谁|你猜我是谁", text))
            skip_mem = bool(_re.search(
                r"刚才|上次|前面|之前|我说过|我说了|你记得我说|你忘了我说|上句话|上一句", text))
            if who_ask:
                messages = self._build_messages(text, "一个经常来找我说话的人，我记不清他是谁但觉得熟悉",
                                                custom_system=custom_system, history=history, skip_brain=True)
            elif skip_mem:
                messages = self._build_messages(text, None, custom_system=custom_system, history=history)
            else:
                messages = self._build_messages(text, text, custom_system=custom_system, history=history)
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # return_tensors="pt" 自带 attention_mask（单样本无 padding，全 1），显式声明 padding=False 保持行为清晰
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=False).to(next(self._model.parameters()).device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,  # Qwen 无 pad_token，必须显式指定
                **gen_kwargs,
            )
        result = self._tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # 对话互动满足社交欲望 (出厂机制)
        if self._desire is not None:
            try:
                self._desire.interact()
                self._desire.save()
            except Exception:
                pass
        return result

    def __call__(self, text: str, **kwargs) -> str:
        return self.generate(text, **kwargs)

    def stream(self, text: str, _internal_skip_emotion: bool = False, **kwargs) -> Generator[str, None, None]:
        """
        真流式生成 (TextIteratorStreamer + 子线程), 逐 token 产出增量文本
        与 generate 共用同一套 prompt 构造/记忆/内心流逻辑
        """
        import torch
        from transformers import TextIteratorStreamer
        self._ensure_loaded()

        # 情绪状态机 tick（与 generate 一致, 异常保护; M3: 内部递归跳过）
        if self._emotion is not None and not _internal_skip_emotion:
            try:
                self._emotion.tick(text)
                self._emotion.save()
            except Exception as e:
                logger.warning("情绪状态机 tick/save 失败: %s", str(e)[:100])
        if self._desire is not None:
            try:
                self._desire.tick()
            except Exception:
                pass

        stage = self._stage_params()
        _t = kwargs.pop("temperature", None)
        _p = kwargs.pop("top_p", None)
        temperature = _t if _t is not None else stage.get("temperature", 0.8)
        top_p = _p if _p is not None else stage.get("top_p", 0.9)
        max_tokens = kwargs.pop("max_tokens", None)
        max_tokens = max_tokens if max_tokens is not None else stage.get("max_tokens", 200)
        memory_query = kwargs.pop("memory_query", None)
        custom_system = kwargs.pop("custom_system", None)
        seed = kwargs.pop("seed", None)
        history = kwargs.pop("history", None)  # 多轮上下文 (S2)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        messages = self._build_messages(text, memory_query, custom_system=custom_system, history=history)
        if self._mind is not None and not custom_system and memory_query is not None:
            try:
                thought = self._mind.get_thought(self, text)
                if thought:
                    messages[0]["content"] = self._mind.build_system_with_thought(
                        messages[0]["content"], thought)
            except Exception:
                pass
        if self._auto_memory and memory_query is None:
            # 指代历史/社交身份类问题: 跳过记忆注入防干扰 (fix 2026-08-15 v6, 与 generate 一致)
            import re as _re
            # 楚门 v1.6: "你是谁/认识我吗" → 定向检索"他"的认知记忆 (模糊熟悉感)
            who_ask = bool(_re.search(
                r"你是谁|我是谁|认识我吗|还记得我吗|你知道我是谁|你猜我是谁", text))
            skip_mem = bool(_re.search(
                r"刚才|上次|前面|之前|我说过|我说了|你记得我说|你忘了我说|上句话|上一句", text))
            if who_ask:
                messages = self._build_messages(text, "一个经常来找我说话的人，我记不清他是谁但觉得熟悉",
                                                custom_system=custom_system, history=history, skip_brain=True)
            elif skip_mem:
                messages = self._build_messages(text, None, custom_system=custom_system, history=history)
            else:
                messages = self._build_messages(text, text, custom_system=custom_system, history=history)
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=False).to(
            next(self._model.parameters()).device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_special_tokens=True, skip_prompt=True, timeout=120)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
            **kwargs,
        )
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        for piece in streamer:
            yield piece
        thread.join(timeout=60)
        # 对话互动满足社交欲望 (出厂机制)
        if self._desire is not None:
            try:
                self._desire.interact()
                self._desire.save()
            except Exception:
                pass

    def close(self):
        if self._model is not None:
            del self._model
            self._model = None
        self._tokenizer = None
        if self._has_cuda():
            import torch
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"Humanizer(stage={self._stage}, device={self._device}, memory={'on' if self._memory else 'off'})"


def get_model_info() -> dict:
    mp = Path(DEFAULT_MODEL_DIR)
    key_files = ["config.json", "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"]
    present = [f for f in key_files if (mp / f).exists()]
    return {
        "model": "Qwen2.5-1.5B-Instruct",
        "path": DEFAULT_MODEL_DIR,
        "installed": bool(present),
        "files": present,
        "base_model": "Qwen2.5-1.5B-Instruct",
        "backend": "transformers",
    }


_global_humanizer: Optional[Humanizer] = None


def humanize(text: str, stage: Optional[str] = None, **kwargs) -> str:
    """全局函数：AI文本 → 人类文本。stage 变化时自动重建实例（避免沿用旧人设）。"""
    global _global_humanizer
    if _global_humanizer is None or (stage is not None and _global_humanizer._stage != stage):
        if _global_humanizer is not None:
            _global_humanizer.close()
        _global_humanizer = Humanizer(stage=stage) if stage else Humanizer()
    return _global_humanizer(text, **kwargs)


def reset_global():
    """释放全局单例（close 后再用 humanize 会重新加载）"""
    global _global_humanizer
    if _global_humanizer is not None:
        _global_humanizer.close()
        _global_humanizer = None


def stream_humanize(text: str, **kwargs) -> Generator[str, None, None]:
    """兼容接口：流式包装"""
    yield humanize(text, **kwargs)
