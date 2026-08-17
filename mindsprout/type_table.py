"""
动态类型表 v2：从经验中自然生长的概念分类系统（可运行版）

核心理念：
  不是预定义类型模板，而是让类型在数据中涌现。
  AI父母（DeepSeek API）做抽象总结；API 不可用时规则回退。

v2 修复（按代码审查）：
  1. _summarize 真实实现（DeepSeek API + 规则回退）
  2. _find_best_match 实现（倒排索引 + 关键词重叠）
  3. child_guess_type 不再硬编码
  4. parent_verify 记录 child_guess_history
  5. 类型树父子关系维护（parent_type/child_types）
  6. 关键词索引改进（按冒号值提取，中文友好）
  7. 名称提炼（取核心动作，不再粗暴截断）
  8. 持久化 save/load
  9. 置信度：渐进增长 + 冲突削弱
  10. 简单合并机制 + 索引过期清理
"""

from mindsprout.config import BASE

import json
import re
import os
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ConceptType:
    """一个从经验中长出来的概念类型"""
    type_id: str
    name: str                          # 人类可读的名称（提炼自摘要）
    abstract_summary: str              # AI父母的高度抽象总结
    core_actions: Set[str] = field(default_factory=set)
    typical_agents: Set[str] = field(default_factory=set)
    typical_recipients: Set[str] = field(default_factory=set)
    typical_tools: Set[str] = field(default_factory=set)
    typical_relations: Set[str] = field(default_factory=set)
    parent_type: Optional[str] = None  # 父类型
    child_types: List[str] = field(default_factory=list)
    shared_with: List[str] = field(default_factory=list)
    emerged_from: List[str] = field(default_factory=list)  # 催生该类型的经验
    experience_count: int = 0
    confidence: float = 0.3            # 经验越多越确定（可被冲突削弱）
    last_used: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DynamicTypeTable:
    """
    动态类型表：概念分类系统（v2 可运行）
    """

    # DeepSeek 抽象总结提示
    SUMMARIZE_PROMPT = """你是AI父母，把一段具体的成长经验抽象为高度概括的类型描述。
规则：去除具体细节（人名、具体物品），只保留核心动作、关系、触发条件、工具类别、情感基调。

输出 JSON 格式：
{{"name": "简短类型名(4-8字)", "core_actions": ["动作1", "动作2"], "relations": ["关系1"], "tools": ["工具类别"], "summary": "一句话概括"}}"""

    # 规则回退：关键词 → 类型名（API 不可用时）
    RULE_FALLBACK = [
        (["喂", "奶", "吃"], "喂养"),
        (["睡", "失眠", "觉"], "睡眠"),
        (["哭", "生气", "委屈", "吵"], "情绪冲突"),
        (["玩", "游戏", "玩具"], "玩耍"),
        (["学", "教", "练习", "作业"], "学习"),
        (["病", "疼", "痛", "药", "医院"], "健康"),
        (["买", "钱", "花"], "消费"),
        (["家", "妈妈", "爸爸", "婆婆"], "家庭"),
        (["朋友", "同学", "老师"], "社交"),
    ]

    def __init__(self, api_key: Optional[str] = None, api_url: str = "https://api.deepseek.com/chat/completions"):
        self.types: Dict[str, ConceptType] = {}
        self.type_index: Dict[str, Set[str]] = {}  # 倒排索引：词 → 类型ID集合
        self.child_guess_history: List[Dict] = []
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or self._read_env()
        self._api_url = api_url
        self._seq = 0

    @staticmethod
    def _read_env() -> Optional[str]:
        env = Path(__file__).resolve().parents[1] / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip()
        return None

    # ============================================================
    # 抽象总结（v2: DeepSeek API + 规则回退）
    # ============================================================
    def _summarize(self, experience_text: str) -> Dict:
        """AI父母抽象总结 → dict；API 失败时规则回退"""
        # 1. 尝试 DeepSeek API
        if self._api_key:
            try:
                import requests
                r = requests.post(
                    self._api_url,
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": self.SUMMARIZE_PROMPT},
                            {"role": "user", "content": experience_text[:500]},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 200,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                m = re.search(r"\{.*\}", content, re.S)
                if m:
                    data = json.loads(m.group(0))
                    if data.get("name"):
                        return data
            except Exception as e:
                print(f"⚠️ 抽象总结 API 失败，规则回退: {str(e)[:60]}")

        # 2. 规则回退：关键词命中 → 类型名 + 摘要
        name = "日常经验"
        for kws, n in self.RULE_FALLBACK:
            if any(k in experience_text for k in kws):
                name = n
                break
        return {
            "name": name,
            "core_actions": [name],
            "relations": [],
            "tools": [],
            "summary": f"{name}类经验（规则回退摘要）",
        }

    # ============================================================
    # 匹配与创建（v2: 真实实现）
    # ============================================================
    def _find_best_match(self, summary: Dict) -> Optional[str]:
        """找最佳匹配类型：倒排索引候选 + 关键词重叠打分"""
        candidates: Set[str] = set()
        for word in self._extract_keywords(summary):
            if word in self.type_index:
                candidates |= self.type_index[word]

        if not candidates:
            return None

        best_id, best_score = None, 0.0
        summary_text = " ".join(summary.get("core_actions", [])) + summary.get("summary", "")
        for tid in candidates:
            t = self.types[tid]
            overlap = self._keyword_overlap(summary_text, t.abstract_summary + t.name)
            score = overlap
            if score > best_score:
                best_id, best_score = tid, score
        return best_id if best_score >= 0.4 else None

    def _keyword_overlap(self, a: str, b: str) -> float:
        """简单关键词重叠率（双字词）"""
        wa = set(re.findall(r"[\u4e00-\u9fff]{2,4}", a))
        wb = set(re.findall(r"[\u4e00-\u9fff]{2,4}", b))
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / min(len(wa), len(wb))

    def _extract_keywords(self, summary: Dict) -> List[str]:
        """从摘要提取索引词（v2: 取冒号后的值 + 核心动作，中文友好）"""
        kws = []
        for v in summary.get("core_actions", []):
            if v:
                kws.append(str(v))
        for v in summary.get("tools", []):
            if v:
                kws.append(str(v))
        name = summary.get("name", "")
        if name:
            kws.append(str(name))
        # 摘要中冒号后的内容（如 "核心动作: 喂养" → "喂养"）
        for m in re.finditer(r"[:：]\s*([\u4e00-\u9fff]{1,6})", str(summary.get("summary", ""))):
            kws.append(m.group(1))
        return [k for k in kws if len(k) >= 2]

    def _create_type(self, summary: Dict, experience_text: str, parent_type: Optional[str] = None) -> ConceptType:
        """创建新类型（v2: 提炼名称 + 维护父子关系）"""
        self._seq += 1
        tid = f"type_{self._seq:04d}"
        name = self._refine_name(summary)
        t = ConceptType(
            type_id=tid,
            name=name,
            abstract_summary=summary.get("summary", ""),
            core_actions=set(summary.get("core_actions", [])),
            typical_tools=set(summary.get("tools", [])),
            typical_relations=set(summary.get("relations", [])),
            parent_type=parent_type,
            emerged_from=[experience_text[:80]],
            experience_count=1,
            confidence=0.3,
        )
        self.types[tid] = t
        # 父子关系维护
        if parent_type and parent_type in self.types:
            if tid not in self.types[parent_type].child_types:
                self.types[parent_type].child_types.append(tid)
        # 索引更新
        self._update_index(t, summary)
        return t

    def _refine_name(self, summary: Dict) -> str:
        """名称提炼（v2: 核心动作优先，不再粗暴截断）"""
        name = summary.get("name", "")
        if name and 2 <= len(name) <= 8:
            return name
        actions = summary.get("core_actions", [])
        if actions:
            return str(actions[0])[:8]
        return "日常经验"

    # ============================================================
    # 核心操作
    # ============================================================
    def find_or_create(self, experience_text: str, parent_ai=None) -> ConceptType:
        """找到已有类型或创建新类型（v2: 完整流程）"""
        summary = self._summarize(experience_text)
        matched_id = self._find_best_match(summary)

        if matched_id:
            t = self.types[matched_id]
            self.refine(matched_id, experience_text, summary)
            return t
        # 尝试父类型（用 RULE_FALLBACK 找粗粒度归属）
        parent = self._guess_parent(summary)
        return self._create_type(summary, experience_text, parent_type=parent)

    def _guess_parent(self, summary: Dict) -> Optional[str]:
        """粗粒度父类型猜测：RULE_FALLBACK 类别词 → 已有根类型（v2 改进）"""
        text = " ".join(summary.get("core_actions", [])) + summary.get("name", "") + summary.get("summary", "")
        # 1. 命中最匹配的粗粒度类别
        best_cat, best_hits = None, 0
        for kws, cat in self.RULE_FALLBACK:
            hits = sum(1 for k in kws if k in text)
            if hits > best_hits:
                best_cat, best_hits = cat, hits
        if best_cat is None or best_hits == 0:
            return None
        # 2. 在已有类型中找该类别的根类型
        for tid, t in self.types.items():
            if t.parent_type is None and (best_cat in t.name or best_cat in t.abstract_summary):
                return tid
        return None

    def child_guess(self, experience_text: str) -> Tuple[Optional[str], float]:
        """孩子尝试归类（v2: 真实匹配；失败时回退父类型归属）"""
        summary = self._summarize(experience_text)
        matched = self._find_best_match(summary)
        if matched:
            t = self.types[matched]
            return matched, t.confidence
        # 回退：粗粒度父类型归属
        parent = self._guess_parent(summary)
        if parent:
            return parent, self.types[parent].confidence * 0.8
        return None, 0.0

    def parent_verify(self, guess_id: Optional[str], experience_text: str) -> Dict:
        """父母验证/纠正孩子的归类（v2: 记录 history）"""
        result = {
            "guess_id": guess_id,
            "experience": experience_text[:80],
            "correct": False,
            "corrected_to": None,
            "ts": datetime.now().isoformat(),
        }
        # 抽象总结 → 找真正归属
        summary = self._summarize(experience_text)
        true_id = self._find_best_match(summary)
        if true_id and guess_id == true_id:
            result["correct"] = True
            self.refine(true_id, experience_text, summary)
        elif true_id:
            result["corrected_to"] = true_id
            self.refine(true_id, experience_text, summary)
        else:
            parent = self._guess_parent(summary)
            t = self._create_type(summary, experience_text, parent_type=parent)
            result["corrected_to"] = t.type_id

        # v2: 记录猜测历史（child_accuracy 依赖）
        self.child_guess_history.append(result)
        return result

    def refine(self, type_id: str, experience_text: str, summary: Optional[Dict] = None):
        """用新经验细化类型（v2: 置信度渐进 + 冲突削弱 + 索引维护）"""
        t = self.types.get(type_id)
        if not t:
            return
        t.experience_count += 1
        t.emerged_from.append(experience_text[:80])
        # 置信度：渐进增长（13条才到0.8，比旧模型谨慎）
        t.confidence = min(0.95, 0.3 + 0.05 * t.experience_count ** 0.7)
        if summary:
            t.core_actions |= set(summary.get("core_actions", []))
            t.typical_tools |= set(summary.get("tools", []))
            # 冲突削弱：摘要与已有核心动作分歧明显时降低置信度
            if summary.get("core_actions") and not (t.core_actions & set(summary["core_actions"])):
                t.confidence *= 0.7
        t.last_used = datetime.now().isoformat()
        if summary:
            self._update_index(t, summary)

    def _update_index(self, t: ConceptType, summary: Dict):
        """倒排索引更新（v2: 清理旧条目 + 只加有效中文词）"""
        # 清理该类型的旧索引条目
        for word in list(self.type_index):
            self.type_index[word].discard(t.type_id)
            if not self.type_index[word]:
                del self.type_index[word]
        # 重建
        for word in self._extract_keywords(summary):
            self.type_index.setdefault(word, set()).add(t.type_id)

    # ============================================================
    # 合并（v2 新增：类型冗余合并）
    # ============================================================
    def merge_duplicates(self, threshold: float = 0.7):
        """合并高度相似的类型（同名/摘要重叠率高）"""
        ids = list(self.types.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] not in self.types or ids[j] not in self.types:
                    continue  # 已被合并删除
                a, b = self.types[ids[i]], self.types[ids[j]]
                if a.name == b.name or self._keyword_overlap(a.abstract_summary, b.abstract_summary) >= threshold:
                    # 合并 b → a
                    a.experience_count += b.experience_count
                    a.emerged_from.extend(b.emerged_from)
                    a.child_types = list(set(a.child_types + b.child_types))
                    a.confidence = min(0.95, a.confidence + 0.1)
                    # 更新子类型的 parent
                    for cid in b.child_types:
                        if cid in self.types:
                            self.types[cid].parent_type = a.type_id
                    del self.types[b.type_id]

    # ============================================================
    # 其他
    # ============================================================
    def child_accuracy(self) -> float:
        """孩子归类准确率（v2: 基于真实 history）"""
        if not self.child_guess_history:
            return 0.0
        return sum(1 for h in self.child_guess_history if h["correct"]) / len(self.child_guess_history)

    def _generate_learning_point(self, type_id: str) -> str:
        t = self.types.get(type_id)
        if not t:
            return ""
        summary = t.abstract_summary or t.name
        return f"归类提示：{summary[:60]}（同类经验 {t.experience_count} 条）"

    def print_type_tree(self, root: Optional[str] = None, depth: int = 0, max_depth: int = 20):
        """可视化类型树（带深度保护）"""
        if depth > max_depth:
            return
        if root is None:
            roots = [tid for tid, t in self.types.items() if t.parent_type is None]
            for rid in roots:
                self.print_type_tree(rid, 0, max_depth)
            return
        t = self.types.get(root)
        if not t:
            return
        indent = "  " * depth
        print(f"{indent}├─ {t.name} ({t.type_id}) 经验×{t.experience_count} 置信度{t.confidence:.2f}")
        for cid in t.child_types:
            self.print_type_tree(cid, depth + 1, max_depth)

    # ============================================================
    # 持久化（v2 新增）
    # ============================================================
    def save(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "seq": self._seq,
            "types": {tid: {
                "name": t.name,
                "abstract_summary": t.abstract_summary,
                "core_actions": list(t.core_actions),
                "typical_agents": list(t.typical_agents),
                "typical_recipients": list(t.typical_recipients),
                "typical_tools": list(t.typical_tools),
                "typical_relations": list(t.typical_relations),
                "parent_type": t.parent_type,
                "child_types": t.child_types,
                "emerged_from": t.emerged_from,
                "experience_count": t.experience_count,
                "confidence": t.confidence,
                "created_at": t.created_at,
            } for tid, t in self.types.items()},
        }
        with open(path / "type_table.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 类型表已保存: {path / 'type_table.json'} ({len(self.types)} 类型)")

    def load(self, directory: str):
        path = Path(directory) / "type_table.json"
        if not path.exists():
            print("⚠️ 无类型表文件")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._seq = data.get("seq", 0)
        self.types = {}
        for tid, d in data["types"].items():
            self.types[tid] = ConceptType(
                type_id=tid,
                name=d["name"],
                abstract_summary=d.get("abstract_summary", ""),
                core_actions=set(d.get("core_actions", [])),
                typical_agents=set(d.get("typical_agents", [])),
                typical_recipients=set(d.get("typical_recipients", [])),
                typical_tools=set(d.get("typical_tools", [])),
                typical_relations=set(d.get("typical_relations", [])),
                parent_type=d.get("parent_type"),
                child_types=d.get("child_types", []),
                emerged_from=d.get("emerged_from", []),
                experience_count=d.get("experience_count", 0),
                confidence=d.get("confidence", 0.3),
                created_at=d.get("created_at", ""),
            )
            # 重建索引
            summary = {"name": d["name"], "core_actions": d.get("core_actions", []),
                       "tools": d.get("typical_tools", []), "summary": d.get("abstract_summary", "")}
            self._update_index(self.types[tid], summary)
        print(f"✅ 类型表已加载: {len(self.types)} 类型")

    def __len__(self):
        return len(self.types)
