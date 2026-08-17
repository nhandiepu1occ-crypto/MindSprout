"""
三组信号评估系统

组1: 自动化指标（每轮训练后自动跑，0成本）
组2: 对抗指标（AI检测器作为反向探针，API成本）
组3: 人类盲测（最终答案，人力成本）

用法:
    from humanize_ai.evaluator import Evaluator
    eval = Evaluator()
    report = eval.full_evaluation(original, rewritten)
"""

from mindsprout.config import BASE

import json
import time
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# 组1: 自动化风格指标（基于 detector.py 的7项检测）
# ============================================================

class AutoMetrics:
    """自动化的AI腔特征检测（跑得快，0成本）"""
    
    # 这7项来自 detector.py —— 直接在这里复用以减少依赖
    HIGH_FREQ_AI_WORDS = {
        "值得注意的是", "综上所述", "在当今", "随着XX的发展",
        "具有重要意义", "不仅...而且", "一方面...另一方面",
        "极大地", "深刻地", "广泛地", "推动", "促进", "提升",
        "优化", "赋能", "打造", "构建", "聚焦", "深耕", "布局",
    }
    
    HIGH_FREQ_CONNECTORS = {
        "因此", "所以", "但是", "然而", "此外", "另外",
        "同时", "而且", "并且", "不过", "虽然", "因为", "总之",
    }
    
    EMOTION_WORDS = {
        "开心", "高兴", "激动", "惊喜", "感动", "难过", "失望",
        "崩溃", "无语", "离谱", "气死", "烦", "心累", "破防",
        "说实话", "讲道理", "说真的", "我跟你讲", "嗯", "诶", "啊", "吧", "呢",
    }
    
    ENDING_PARTICLES = set('吧呢啊嘛啦哦哈诶哇呀呗喽嘞')
    
    def __init__(self):
        self.history: List[Dict] = []
    
    def evaluate(self, text: str) -> Dict:
        """跑全部7项检测，返回综合分"""
        scores = {}
        suggestions = []
        
        # 1. 句子节奏
        rhythm = self._rhythm_score(text)
        scores["rhythm"] = rhythm
        
        # 2. AI高频词
        vocab = self._vocab_score(text)
        scores["vocab"] = vocab
        
        # 3. 模板化结构
        template = self._template_score(text)
        scores["template"] = template
        
        # 4. 个人性
        personal = self._personal_score(text)
        scores["personal"] = personal
        
        # 5. 连接词密度
        connector = self._connector_score(text)
        scores["connector"] = connector
        
        # 6. 情感词
        emotion = self._emotion_score(text)
        scores["emotion"] = emotion
        
        # 7. 句末语气词
        ending = self._ending_score(text)
        scores["ending"] = ending
        
        # 加权综合分
        weights = {
            "rhythm": 0.15, "vocab": 0.25, "template": 0.15,
            "personal": 0.15, "connector": 0.10, "emotion": 0.10, "ending": 0.10,
        }
        composite = sum(scores[k]["score"] * weights[k] for k in weights)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text),
            "scores": scores,
            "composite": round(composite, 3),
            "suggestions": [s for s in [
                rhythm.get("fix"), vocab.get("fix"), template.get("fix"),
                personal.get("fix"), connector.get("fix"), emotion.get("fix"),
                ending.get("fix"),
            ] if s],
        }
        
        self.history.append(result)
        return result
    
    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'[。！？；\n]', text) if s.strip()]
    
    def _rhythm_score(self, text: str) -> Dict:
        sents = self._split_sentences(text)
        if len(sents) < 3:
            return {"score": 0.5, "fix": None}
        lengths = [len(s) for s in sents]
        std = statistics.stdev(lengths) if len(lengths) > 1 else 0
        score = min(1.0, std / 10)
        return {
            "score": round(score, 3),
            "std_dev": round(std, 1),
            "fix": "句子节奏太均匀，打散长短句" if std < 5 else None,
        }
    
    def _vocab_score(self, text: str) -> Dict:
        hits = [w for w in self.HIGH_FREQ_AI_WORDS if w in text]
        density = len(hits) / max(len(text) / 100, 1)
        score = max(0, 1.0 - density)
        return {
            "score": round(score, 3),
            "hits": hits,
            "fix": f"删除AI高频词: {', '.join(hits[:5])}" if hits else None,
        }
    
    def _template_score(self, text: str) -> Dict:
        sents = self._split_sentences(text)
        if len(sents) < 2:
            return {"score": 0.7, "fix": None}
        patterns = [r'^.+是.+的$', r'^随着.+的发展', r'^在.+的(背景|时代)下', r'^根据.+的(研究|数据)']
        template_count = sum(
            1 for s in sents[:3]
            for p in patterns if re.match(p, s)
        )
        score = 1.0 - template_count / min(len(sents[:3]), 3)
        return {
            "score": round(score, 3),
            "fix": "前几句全是模板化开头" if template_count >= 2 else None,
        }
    
    def _personal_score(self, text: str) -> Dict:
        has_me = bool(re.search(r'我[^们]|咱|俺', text))
        has_detail = bool(re.search(r'\d+|[京津沪渝]|(北京|上海|广州)', text))
        score = (0.3 if has_me else 0) + (0.3 if has_detail else 0) + 0.4
        return {
            "score": round(score, 3),
            "fix": "缺少个人色彩：加'我'和具体细节" if not has_me and not has_detail else None,
        }
    
    def _connector_score(self, text: str) -> Dict:
        count = sum(text.count(c) for c in self.HIGH_FREQ_CONNECTORS)
        density = count / max(len(text) / 100, 1)
        score = max(0, 1.0 - density / 3)
        return {
            "score": round(score, 3),
            "density": round(density, 2),
            "fix": f"连接词过多({count}个)" if density > 2.0 else None,
        }
    
    def _emotion_score(self, text: str) -> Dict:
        found = [w for w in self.EMOTION_WORDS if w in text]
        density = len(found) / max(len(text) / 100, 1)
        score = min(1.0, density * 5)
        return {
            "score": round(score, 3),
            "found": found,
            "fix": "没有任何情感词" if not found else None,
        }
    
    def _ending_score(self, text: str) -> Dict:
        sents = self._split_sentences(text)
        if not sents:
            return {"score": 0.5, "fix": None}
        ending_count = sum(1 for s in sents if s and s[-1] in self.ENDING_PARTICLES)
        ratio = ending_count / len(sents)
        score = min(1.0, ratio * 3)
        return {
            "score": round(score, 3),
            "ratio": round(ratio, 2),
            "fix": f"句末语气词太少({ending_count}/{len(sents)})" if ratio < 0.1 and len(sents) >= 3 else None,
        }
    
    def trend(self) -> Dict:
        """综合分趋势（用于仪表盘）"""
        if not self.history:
            return {"direction": "flat", "values": []}
        values = [h["composite"] for h in self.history]
        if len(values) >= 3:
            if values[-1] > values[-3] + 0.05:
                direction = "up"
            elif values[-1] < values[-3] - 0.05:
                direction = "down"
            else:
                direction = "flat"
        else:
            direction = "flat"
        return {"direction": direction, "values": values[-20:], "current": values[-1]}


# ============================================================
# 组2: 对抗指标（AI检测器反向探针）
# ============================================================

class AdversarialMetrics:
    """
    用AI检测器作为反向探针——不是目标，是诊断工具。
    
    三个检测器的得分趋势揭示：
    - 都降 → 方向对
    - 两个降一个不降 → 只规避了部分检测特征
    - 全不降 → 改写和原文在统计上没区别（严重问题）
    """
    
    def __init__(self):
        self.history: List[Dict] = []
    
    def evaluate(self, original: str, rewritten: str) -> Dict:
        """
        跑三个检测器，对比改写前后
        
        TODO: 接入实际检测器API
        - GPTZero API
        - Originality.ai API  
        - Sapling AI Detector (免费)
        """
        # 临时用统计模拟
        result = {
            "timestamp": datetime.now().isoformat(),
            "original_scores": {
                "gptzero": self._mock_detect(original),
                "originality": self._mock_detect(original),
                "sapling": self._mock_detect(original),
            },
            "rewritten_scores": {
                "gptzero": self._mock_detect(rewritten),
                "originality": self._mock_detect(rewritten),
                "sapling": self._mock_detect(rewritten),
            },
        }
        
        # 计算降幅
        for detector in ["gptzero", "originality", "sapling"]:
            orig = result["original_scores"][detector]
            new = result["rewritten_scores"][detector]
            result[f"{detector}_delta"] = round(orig - new, 2)
        
        self.history.append(result)
        return result
    
    def _mock_detect(self, text: str) -> float:
        """临时模拟检测器得分（实际需替换为API调用）"""
        # AI腔越重 → 得分越高
        ai_signals = sum(text.count(w) for w in AutoMetrics.HIGH_FREQ_AI_WORDS)
        base = 0.7 + ai_signals * 0.02
        return round(min(0.99, base + (hash(text) % 10) / 100), 2)
    
    def trend(self) -> Dict:
        """三个检测器的得分趋势"""
        if not self.history:
            return {"direction": "flat"}
        
        latest = self.history[-1]
        deltas = {
            k: latest[k] for k in latest if k.endswith("_delta")
        }
        avg_delta = sum(deltas.values()) / len(deltas) if deltas else 0
        
        return {
            "deltas": deltas,
            "avg_delta": round(avg_delta, 2),
            "direction": "improving" if avg_delta > 0.1 else "stalled" if avg_delta > 0 else "worsening",
        }


# ============================================================
# 组3: 人类盲测
# ============================================================

class HumanBlindTest:
    """
    A/B对比盲测
    
    设计原则：
    - 不评绝对分数，只做相对选择（二选一）
    - 小样本、轮换评估者
    - 统计显著性检验
    """
    
    def __init__(self):
        self.results: List[Dict] = []
    
    def create_test_set(
        self,
        ai_texts: List[str],
        method_a_outputs: List[str],  # 例如: Paperxie改写
        method_b_outputs: List[str],  # 例如: 子AI改写
    ) -> List[Dict]:
        """创建盲测对（随机化顺序）"""
        import random
        pairs = []
        for i, (ai, a, b) in enumerate(zip(ai_texts, method_a_outputs, method_b_outputs)):
            # 随机决定A/B顺序
            if random.random() > 0.5:
                pairs.append({
                    "id": i,
                    "ai_original": ai,
                    "option_1": a, "method_1": "A",
                    "option_2": b, "method_2": "B",
                })
            else:
                pairs.append({
                    "id": i,
                    "ai_original": ai,
                    "option_1": b, "method_1": "B",
                    "option_2": a, "method_2": "A",
                })
        return pairs
    
    def record_result(self, pair_id: int, chosen_option: int, evaluator: str):
        """记录一次选择"""
        self.results.append({
            "pair_id": pair_id,
            "chosen": chosen_option,
            "evaluator": evaluator,
            "timestamp": datetime.now().isoformat(),
        })
    
    def analyze(self, test_pairs: List[Dict]) -> Dict:
        """
        统计分析
        
        如果方法B的胜率>70%且Wilcoxon p<0.05 → 显著优于A
        """
        # 统计每个方法的胜出次数
        wins_a = 0
        wins_b = 0
        
        # 按pair聚合
        for pair in test_pairs:
            pair_results = [r for r in self.results if r["pair_id"] == pair["id"]]
            if not pair_results:
                continue
            # 多数投票
            vote_1 = sum(1 for r in pair_results if r["chosen"] == 1)
            vote_2 = sum(1 for r in pair_results if r["chosen"] == 2)
            
            if vote_1 > vote_2:
                winner = pair["method_1"]
            else:
                winner = pair["method_2"]
            
            if winner == "A":
                wins_a += 1
            else:
                wins_b += 1
        
        total = wins_a + wins_b
        if total == 0:
            return {"error": "no results"}
        
        win_rate_b = wins_b / total
        significance = "significant" if win_rate_b >= 0.7 else "not_significant"
        
        return {
            "total_pairs": total,
            "method_a_wins": wins_a,
            "method_b_wins": wins_b,
            "method_b_win_rate": round(win_rate_b, 2),
            "significance": significance,
            "verdict": (
                "✅ 方法B显著优于方法A" if significance == "significant"
                else "❌ 无法证明方法B优于方法A"
            ),
        }


# ============================================================
# 综合评估器
# ============================================================

@dataclass
class EvalReport:
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    auto: Dict = field(default_factory=dict)
    adversarial: Dict = field(default_factory=dict)
    summary: str = ""

class Evaluator:
    """三组信号综合评估"""
    
    def __init__(self):
        self.auto = AutoMetrics()
        self.adversarial = AdversarialMetrics()
    
    def evaluate_rewrite(self, original: str, rewritten: str) -> EvalReport:
        """评估一次改写"""
        # 组1: 自动化指标
        auto_result = self.auto.evaluate(rewritten)
        
        # 组2: 对抗指标
        adv_result = self.adversarial.evaluate(original, rewritten)
        
        # 生成摘要
        auto_trend = self.auto.trend()
        adv_trend = self.adversarial.trend()
        
        summary = (
            f"综合分: {auto_result['composite']:.2f} ({auto_trend['direction']}) | "
            f"检测器降幅: {adv_trend['avg_delta']:.2f} ({adv_trend['direction']}) | "
            f"问题数: {len(auto_result['suggestions'])}"
        )
        
        return EvalReport(
            auto=auto_result,
            adversarial=adv_result,
            summary=summary,
        )
    
    def evaluate_batch(
        self, 
        pairs: List[Tuple[str, str]],
        progress_callback=None,
    ) -> List[EvalReport]:
        """批量评估"""
        reports = []
        for i, (orig, rewritten) in enumerate(pairs):
            report = self.evaluate_rewrite(orig, rewritten)
            reports.append(report)
            if progress_callback:
                progress_callback(i + 1, len(pairs))
        return reports
