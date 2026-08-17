from mindsprout.config import BASE
"""
双系统记忆库：坐标网 + 内容库

系统A（坐标网）：NetworkX有向图
  - 节点 = 经验坐标ID（无内容）
  - 边 = 加权连接
  - 支持：增量添加、增强/衰减、扩散激活、多父节点

系统B（内容库）：字典
  - key = 经验坐标ID
  - value = {text, scene_graph, anchor_embedding, emotion}
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx


@dataclass
class Experience:
    """一条经验"""
    exp_id: str
    text: str                              # 原始人类文本
    source_url: str = ""                   # 来源URL
    source_type: str = ""                  # 来源类型: experience/imagination/...
    source_year: int = 2020                # 来源年份（验证是2022前）
    scene_graph: Optional[Dict] = None     # 场景图
    anchor_embedding: Optional[np.ndarray] = None  # 锚点embedding（768维）
    emotion_vector: Optional[Dict] = None  # 情感标注 {valence: +0.8, arousal: 0.3}
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activated: str = ""


def _make_jsonable(obj):
    """递归转 JSON 可序列化 (scene_graph 可能含 numpy/Path 等)"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)

class ContentStore:
    """系统B：内容存储"""
    
    def __init__(self):
        self._store: Dict[str, Experience] = {}
    
    def put(self, exp: Experience):
        self._store[exp.exp_id] = exp
    
    def get(self, exp_id: str) -> Optional[Experience]:
        return self._store.get(exp_id)
    
    def get_batch(self, exp_ids: List[str]) -> List[Experience]:
        return [self._store[eid] for eid in exp_ids if eid in self._store]
    
    def update(self, exp_id: str, **kwargs):
        if exp_id in self._store:
            exp = self._store[exp_id]
            for k, v in kwargs.items():
                setattr(exp, k, v)
            exp.last_activated = datetime.now().isoformat()
    
    def remove(self, exp_id: str):
        self._store.pop(exp_id, None)

    def ids(self) -> List[str]:
        """返回所有经验 ID（供外部安全遍历，避免直接访问内部字典）"""
        return list(self._store.keys())
    
    def __len__(self):
        return len(self._store)
    
    def __contains__(self, exp_id: str):
        return exp_id in self._store


class ConnectionNetwork:
    """
    系统A：坐标网络
    
    纯图结构——节点不存内容，只存连接关系。
    支持增量添加、权重调整、扩散激活。
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_experience(
        self, 
        exp_id: str, 
        parent_ids: List[str] = None,
        initial_weight: float = 0.5,
    ):
        """新增一个经验坐标节点（v2: 父节点不存在时自动创建为类型节点）"""
        import time as _time
        ts = _time.time()
        self.graph.add_node(exp_id, created_at=datetime.now().isoformat(),
                            last_activated_ts=ts)
        
        if parent_ids:
            for pid in parent_ids:
                if pid not in self.graph:
                    # 父节点不存在 → 自动创建（类型节点）
                    self.graph.add_node(pid, is_type_node=True, created_at=datetime.now().isoformat(),
                                        last_activated_ts=ts)
                self.graph.add_edge(pid, exp_id, weight=initial_weight)
    
    def add_shared_connection(
        self, 
        exp_id: str, 
        shared_parent_ids: List[str],
        weight: float = 0.3,
    ):
        """添加共子节点连接（一个节点有多个父分支）"""
        for pid in shared_parent_ids:
            if pid not in self.graph:
                # 与 add_experience 一致: 父节点缺失自动创建 (类型节点)
                self.graph.add_node(pid, is_type_node=True, created_at=datetime.now().isoformat(),
                                    last_activated_ts=time.time())
            self.graph.add_edge(pid, exp_id, weight=weight, shared=True)
    
    def spread_activate(
        self,
        seed_ids: List[str],
        k_hops: int = 2,
        decay: float = 0.8,
        threshold: float = 0.1,
        direction: str = "both",
    ) -> Dict[str, float]:
        """
        扩散激活：从种子节点扩散 k_hops 跳（v2 重写）

        - 种子信号 = 1.0，每经过一条边衰减 ×weight×decay
        - 双向扩散（both）：经验节点通常只有入边，单向会扩散不出去
        - 最强信号保留：new > 已激活才更新（弱信号不覆盖强信号）
        """
        activated: Dict[str, float] = {}
        current = [(sid, 1.0) for sid in seed_ids if sid in self.graph]
        for sid, _sig in current:
            activated[sid] = 1.0

        for _ in range(k_hops):
            next_layer: Dict[str, float] = {}  # 每节点只保留该层最强信号（防重复处理）
            for node, signal in current:
                if direction == "out":
                    neighbors = set(self.graph.successors(node))
                elif direction == "in":
                    neighbors = set(self.graph.predecessors(node))
                else:
                    neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                for nb in neighbors:
                    edge_weight = self.graph[node][nb].get("weight", 0.5) if self.graph.has_edge(node, nb) else 0.5
                    new_signal = signal * edge_weight * decay
                    if new_signal > threshold and new_signal > activated.get(nb, 0):
                        activated[nb] = new_signal
                        if new_signal > next_layer.get(nb, 0):
                            next_layer[nb] = new_signal
            current = list(next_layer.items())

        return activated
    
    def strengthen(self, co_activated_pairs: List[Tuple[str, str]], delta: float = 0.05):
        """同时激活的节点对 → 连接增强（模拟记忆巩固）"""
        for u, v in co_activated_pairs:
            if self.graph.has_edge(u, v):
                current = self.graph[u][v].get('weight', 0.5)
                self.graph[u][v]['weight'] = min(1.0, current + delta)
    
    def weaken_all(self, decay_rate: float = 0.001):
        """全局连接衰减（v2: 时间感知——最近激活的边衰减更慢，模拟'最近想起的不容易忘'）"""
        import time as _time
        now = _time.time()
        for u, v in self.graph.edges():
            current = self.graph[u][v].get('weight', 0.5)
            # 用边两端节点的最近激活时间估算新鲜度
            last_act = 0.0
            for n in (u, v):
                la = self.graph.nodes[n].get('last_activated_ts', 0)
                last_act = max(last_act, la)
            staleness = 1.0 if last_act == 0 else min(1.0, (now - last_act) / 86400.0)
            effective = decay_rate * (0.3 + 0.7 * staleness)  # 新鲜边衰减慢
            self.graph[u][v]['weight'] = max(0.01, current - effective)
    
    def get_neighbors(self, exp_id: str, direction: str = "both") -> List[str]:
        """获取邻居节点"""
        if exp_id not in self.graph:
            return []
        if direction == "out":
            return list(self.graph.successors(exp_id))
        elif direction == "in":
            return list(self.graph.predecessors(exp_id))
        else:
            return list(set(self.graph.predecessors(exp_id)) | set(self.graph.successors(exp_id)))
    
    def __len__(self):
        return self.graph.number_of_nodes()
    
    def __contains__(self, exp_id: str):
        return exp_id in self.graph


class MemoryBank:
    """
    记忆库：将系统A（坐标网）和系统B（内容库）组合
    
    对外暴露统一接口：
    - store(): 存储经验 → 同时更新坐标网+内容库+锚点
    - query(): 给定查询向量 → 返回相关经验内容
    - reinforce(): 强化连接
    - decay(): 全局衰减
    """
    
    def __init__(self, encoder=None):
        """
        Args:
            encoder: Qwen Encoder（用于生成锚点embedding）。
                    如果不传，使用内置的字符bigram哈希编码器（v1轻量替代，后续换Qwen）。
        """
        self.network = ConnectionNetwork()
        self.content = ContentStore()
        self.encoder = encoder  # 可选的Qwen Encoder
        self._anchor_cache: Dict[str, np.ndarray] = {}  # 内存中缓存锚点
        # 内置轻量编码器：字符bigram → 768维确定性哈希向量（sha256，跨进程可复现）
        self._EMBED_DIM = 768
        # 主题词典 (timeline v2: 轻量语义增强, 弥补哈希编码的语义鸿沟)
        self._TOPIC_LEXICON = {
            "宠物": ["宠物", "金鱼", "猫", "狗", "鱼", "兔子", "鸟", "泡泡", "养", "死"],
            "考试": ["考试", "考砸", "测验", "月考", "分数", "卷子", "成绩", "排名", "复习", "数学"],
            "朋友": ["朋友", "同桌", "小雨", "闺蜜", "同学", "玩伴", "邻居", "小美"],
            "家庭": ["妈妈", "爸爸", "爸妈", "奶奶", "姥姥", "姑姑", "家人", "弟弟", "妹妹", "爷爷"],
            "学校": ["学校", "老师", "上课", "作业", "教室", "操场", "体育", "听写", "课本", "班长"],
            "天气": ["下雨", "雨", "晴天", "雪", "伞", "淋", "天气"],
            "食物": ["零食", "辣条", "糖", "饭", "蛋糕", "冰淇淋", "西红柿", "排骨", "鸡蛋"],
            "玩具": ["玩具", "铲子", "积木", "气球", "滑梯", "秋千", "风筝", "过家家", "沙坑"],
            "情绪": ["哭", "笑", "生气", "委屈", "高兴", "难过", "害怕", "紧张", "开心", "烦"],
            "幼儿园": ["幼儿园", "小班", "大班", "老师", "小朋友"],
            "生病": ["发烧", "感冒", "病", "医院", "药", "打针", "难受"],
        }
        self._TOPIC_WORDS = [w for ws in self._TOPIC_LEXICON.values() for w in ws]

    def _topics(self, text: str) -> set:
        """命中主题集合"""
        hits = set()
        for topic, words in self._TOPIC_LEXICON.items():
            for w in words:
                if w and w in (text or ""):
                    hits.add(topic)
                    break
        return hits

    def _encode(self, text: str) -> np.ndarray:
        """生成文本的锚点embedding

        v2: 使用 hashlib.sha256 确定性哈希（Python 内置 hash() 跨进程随机化，不可用！）
        内部 float32 运算，持久化时再转 float16。
        """
        if self.encoder is not None:
            try:
                emb = self.encoder.encode(text)
                if hasattr(emb, "detach"):  # torch tensor
                    emb = emb.detach().cpu()
                return np.asarray(emb, dtype=np.float32)
            except Exception as e:
                if not getattr(self, "_warned_encoder", False):
                    self._warned_encoder = True
                    print(f"⚠️ 外部 encoder 编码失败，回退内置哈希编码: {e}", flush=True)

        import hashlib
        vec = np.zeros(self._EMBED_DIM, dtype=np.float32)
        text = text.strip().replace(" ", "")
        if not text:
            return vec
        grams = [text[i:i + 2] for i in range(len(text) - 1)] + list(text)
        for g in grams:
            # 确定性哈希：sha256 → 两个投影位置（跨进程可复现）
            h1 = int(hashlib.sha256(g.encode("utf-8")).hexdigest()[:8], 16) % self._EMBED_DIM
            h2 = int(hashlib.sha256((g + "#").encode("utf-8")).hexdigest()[:8], 16) % self._EMBED_DIM
            vec[h1] += 1.0
            vec[h2] -= 0.5
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec.astype(np.float32)  # 内部保持 float32，避免 float16 溢出/nan

    def store(
        self,
        exp_id: str,
        text: str,
        parent_ids: List[str] = None,
        source_url: str = "",
        source_type: str = "",
        source_year: int = 2020,
        scene_graph: Dict = None,
        emotion_vector: Dict = None,
    ):
        """存储一个新经验"""
        # 0. 重复 ID 检查 (fix: 防止静默覆盖)
        if exp_id in self.content:
            print(f"⚠️ store 重复 exp_id {exp_id}，已存在将覆盖（建议用 update）")
        # 1. 生成锚点embedding（内置编码器或外部encoder）
        anchor = self._encode(text)
        self._anchor_cache[exp_id] = anchor

        # 2. 创建经验对象 → 存内容库
        exp = Experience(
            exp_id=exp_id,
            text=text,
            source_url=source_url,
            source_type=source_type,
            source_year=source_year,
            scene_graph=scene_graph,
            anchor_embedding=anchor,
            emotion_vector=emotion_vector,
        )
        self.content.put(exp)

        # 3. 在坐标网中加节点+边
        self.network.add_experience(exp_id, parent_ids)

    def query(
        self,
        query_text: str = None,
        query_vector: np.ndarray = None,
        seed_ids: List[str] = None,
        k_hops: int = 2,
        top_k: int = 5,
        decay: float = 0.8,
        summary_chars: int = 100,
        touch_activated: bool = True,
        emotion_bias: str = None,
    ) -> Dict:
        """
        查询记忆库
        
        可以用三种方式触发查询：
        1. query_text: 用文本→编码→cosine匹配种子
        2. query_vector: 直接用向量匹配
        3. seed_ids: 手动指定种子节点
        
        Args:
            summary_chars: fused_text 每条摘要长度（默认100）
            touch_activated: 命中后更新时间戳（"最近想起的不容易忘"，默认开）
        
        Returns:
            {
                "activated": {exp_id: signal_strength},
                "contents": [Experience对象列表，按信号强度排序],
                "fused_text": 拼接后的文本摘要,
                "query_time_ms": 查询耗时
            }
        """
        import time
        t0 = time.time()
        
        # 1. 确定种子节点
        if seed_ids is None:
            if query_vector is None and query_text:
                query_vector = self._encode(query_text)
            # 主题boost用: 仅文本查询才更新, vector-only 查询置空防残留错配 (fix 2026-08-15)
            self._last_query_text = query_text or "" if query_text else ""
            if query_vector is not None:
                seed_ids = self._cosine_topk(query_vector, top_k)
            else:
                seed_ids = []
        
        # 2. 扩散激活
        activated = self.network.spread_activate(seed_ids, k_hops, decay)
        # 保底 (楚门 v1.6): 种子即使不在图中也要激活 (孤立新记忆也能被想起)
        for eid in seed_ids:
            if eid not in activated:
                activated[eid] = 0.9
        
        # 3. 取出内容
        contents = []
        for exp_id, signal in sorted(activated.items(), key=lambda x: -x[1]):
            exp = self.content.get(exp_id)
            if exp:
                contents.append((exp, signal))
        
        # 3.4 情绪偏置: 当前情绪匹配的记忆信号增强 (情绪一致性记忆, Bower 1981)
        if emotion_bias and contents:
            bias = emotion_bias.lower()
            boosted = []
            for exp, signal in contents:
                ev = exp.emotion_vector or {}
                dom = str(ev.get("dominant", "neutral")).lower()
                s = signal
                if dom == bias:
                    s *= 1.6   # 匹配情绪: 信号 ×1.6
                elif bias == "neutral" and dom == "neutral":
                    s *= 1.3
                boosted.append((exp, s))
            boosted.sort(key=lambda x: -x[1])
            contents = boosted

        # 3.45 真/假记忆出厂区分 (v4): imagination 记忆只浅层影响
        #    - 信号 ×0.15 (低权重, 不构成事实)
        #    - 若仍进入 top, 表达时必须带"我想的"标记 (engine._build_messages 处理)
        contents = [(exp, s * 0.15 if (exp.source_url or "") == "imagination" else s)
                    for exp, s in contents]
        contents.sort(key=lambda x: -x[1])

        # 3.46 最近激活降权 (V3.9.1): 防同一记忆连续命中 → 原句重复
        #    刚想起过的记忆短期降权 (10min内×0.45, 30min内×0.7), 强制检索多样性
        try:
            from datetime import datetime as _dt
            _now = _dt.now()
            _new_contents = []
            for _exp, _sig in contents:
                _w = 1.0
                if _exp.last_activated:
                    try:
                        _last = _dt.fromisoformat(_exp.last_activated)
                        _mins = (_now - _last).total_seconds() / 60.0
                        if _mins < 10:
                            _w = 0.45
                        elif _mins < 30:
                            _w = 0.7
                    except Exception:
                        pass
                _new_contents.append((_exp, _sig * _w))
            _new_contents.sort(key=lambda x: -x[1])
            contents = _new_contents
        except Exception:
            pass

        # 3.5 命中即“想起”：更新时间戳（时间感知衰减的数据源）
        if touch_activated:
            for exp, _sig in contents[:top_k]:
                self.touch(exp.exp_id)
        
        # 4. 融合文本（简单拼接前几条经验的文本摘要）
        fused_parts = []
        for exp, signal in contents[:3]:
            summary = exp.text[:summary_chars] + "..." if len(exp.text) > summary_chars else exp.text
            fused_parts.append(f"[{signal:.1f}] {summary}")
        fused_text = "\n".join(fused_parts)
        
        t1 = time.time()
        
        return {
            "activated": activated,
            "contents": contents,
            "fused_text": fused_text,
            "query_time_ms": round((t1 - t0) * 1000, 2),
        }
    
    def _cosine_topk(self, query_vec: np.ndarray, k: int) -> List[str]:
        """cosine相似度 → Top-K种子节点（只选有内容的经验节点，不选类型节点；矩阵向量化）"""
        content_ids = self.content.ids()  # 安全遍历，不直接访问内部字典
        if not content_ids:
            return []

        if not self._anchor_cache:
            # 没有锚点 → 不猜测语义，返回空（提示无锚点）
            print("⚠️ 锚点缓存为空，cosine 检索不可用（返回空种子）", flush=True)
            return []

        # 向量化：矩阵一次性点积（v2: 替换逐条 Python 循环）
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        anchors = np.stack([np.asarray(self._anchor_cache[eid], dtype=np.float32) for eid in content_ids])
        anchor_norms = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
        cosines = anchor_norms @ query_norm  # (N,)
        scores = {eid: float(c) for eid, c in zip(content_ids, cosines)}
        # 主题词典 boost (v2.1: 弥补哈希编码语义鸿沟, "宠物"→"金鱼"类远距离命中)
        q_topics = self._topics(getattr(self, "_last_query_text", ""))
        if q_topics:
            for i, eid in enumerate(content_ids):
                exp = self.content.get(eid)
                if exp is None:
                    continue
                m_topics = self._topics(exp.text)
                overlap = len(q_topics & m_topics)
                if overlap:
                    scores[eid] = float(cosines[i]) + 0.18 * overlap
        # 时间 boost (楚门世界 v1): query 问"今天/最近" → 今天的生活记忆绝对优先 (放在fab boost后, 碾压)
        qtext = getattr(self, "_last_query_text", "")
        if any(k in qtext for k in ("今天", "最近", "刚刚")):
            for i, eid in enumerate(content_ids):
                exp = self.content.get(eid)
                if exp is None:
                    continue
                t = (exp.text or "")[:12]
                sg = exp.scene_graph if isinstance(exp.scene_graph, dict) else {}
                is_today = sg.get("time") == "today" or any(
                    k in t for k in ("今天", "中午", "下午", "晚上"))
                if is_today:
                    scores[eid] = 2.0 + float(cosines[i]) * 0.1  # 绝对优先: 问今天先说今天
        # 遗忘衰减 (楚门 v1.5): 生活记忆按年龄衰减 — 久远的可以忘掉; 童年/关键记忆保留
        # 判定: 生活记忆 = scene_graph.time in (today, life); 童年/关键记忆不衰减
        import time as _time
        from datetime import datetime as _dt
        _now = _time.time()
        for i, eid in enumerate(content_ids):
            exp = self.content.get(eid)
            if exp is None:
                continue
            sg = exp.scene_graph if isinstance(exp.scene_graph, dict) else {}
            if sg.get("time") not in ("today", "life"):
                continue  # 童年/关键记忆不衰减
            try:
                created = _dt.fromisoformat(exp.created_at).timestamp() if exp.created_at else _now
            except Exception:
                created = _now
            age_days = (_now - created) / 86400
            if age_days > 365:
                decay = 0.05
            elif age_days > 90:
                decay = 0.2
            elif age_days > 30:
                decay = 0.5
            else:
                decay = 1.0
            if decay < 1.0:
                scores[eid] = float(scores[eid]) * decay
        # 成长记忆（AI父母编造的蠢珞珞童年）检索加权：短文本拼不过长文本，适度补偿 ×1.5 (v2.2: 原×3霸榜压过真实相关记忆, 已降)
        # 兼容 scene_graph 顶层 source 键；也 fallback 检查 Experience 对象
        for i, eid in enumerate(content_ids):
            exp = self.content.get(eid)
            if not exp:
                continue
            sg = exp.scene_graph if isinstance(exp.scene_graph, dict) else {}
            is_fab = sg.get("source") == "ai_parent_fabricated" or getattr(exp, "source_type", "") == "ai_parent_fabricated"
            if is_fab:
                # 今天的生活记忆(时间boost过)不再被 fab 抬高 (楚门v1, 否则童年记忆反超)
                t2 = (exp.text or "")[:12]
                is_today2 = sg.get("time") == "today" or any(
                    k in t2 for k in ("今天", "中午", "下午", "早上", "晚上"))
                if not is_today2:
                    scores[eid] = float(cosines[i]) * 1.5 + (scores[eid] - float(cosines[i]))

        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return sorted_ids[:k]

    def touch(self, exp_id: str):
        """激活节点：同步更新时间戳（网络图节点 + 内容库 Experience）——时间感知衰减的数据源"""
        import time as _time
        ts = _time.time()
        if exp_id in self.network:
            self.network.graph.nodes[exp_id]["last_activated_ts"] = ts
        exp = self.content.get(exp_id)
        if exp is not None:
            exp.last_activated = datetime.now().isoformat()

    def delete(self, exp_id: str):
        """统一删除：内容库 + 网络节点（含关联边）+ 锚点缓存，三处同步"""
        self.content.remove(exp_id)
        if exp_id in self.network:
            self.network.graph.remove_node(exp_id)  # remove_node 自动删除关联边
        self._anchor_cache.pop(exp_id, None)
    
    def reinforce(self, co_activated_pairs: List[Tuple[str, str]]):
        """强化同时激活的连接"""
        self.network.strengthen(co_activated_pairs)
    
    def decay(self):
        """全局记忆衰减"""
        self.network.weaken_all()
    
    def save(self, directory: str):
        """持久化到文件夹"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # 坐标网 → JSON边列表 + 节点属性（v2: 保存 created_at + last_activated_ts）
        edges = []
        node_attrs = {}
        for n, attrs in self.network.graph.nodes(data=True):
            node_attrs[n] = {
                "created_at": attrs.get("created_at", ""),
                "last_activated_ts": attrs.get("last_activated_ts", 0),
            }
        for u, v in self.network.graph.edges():
            edges.append({
                "from": u, "to": v,
                "weight": self.network.graph[u][v].get('weight', 0.5),
                "shared": self.network.graph[u][v].get('shared', False),
            })
        try:
            with open(path / "graph.json", "w", encoding="utf-8") as f:
                json.dump({"version": 2, "edges": edges, "nodes": node_attrs}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ graph.json 保存失败: {e}")
        
        # 内容库 → JSON
        contents = {}
        for exp_id in self.content.ids():
            exp = self.content.get(exp_id)
            if exp is None:
                continue
            contents[exp_id] = {
                "text": exp.text,
                "source_url": exp.source_url,
                "source_type": getattr(exp, "source_type", ""),
                "source_year": exp.source_year,
                "scene_graph": _make_jsonable(exp.scene_graph),
                "emotion_vector": _make_jsonable(exp.emotion_vector),
                "created_at": exp.created_at,
                "last_activated": exp.last_activated,
            }
        try:
            with open(path / "content.json", "w", encoding="utf-8") as f:
                json.dump(contents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ content.json 保存失败: {e}")
        
        # 锚点embedding → numpy
        if self._anchor_cache:
            # 保持顺序一致
            exp_ids = sorted(self._anchor_cache.keys())
            anchors = np.stack([self._anchor_cache[eid] for eid in exp_ids])
            try:
                np.save(path / "anchors.npy", anchors)
                with open(path / "anchor_ids.json", "w") as f:
                    json.dump(exp_ids, f)
            except Exception as e:
                print(f"⚠️ anchors 保存失败: {e}")
        
        print(f"✅ 记忆库已保存到 {directory} ({len(self)}经验, {len(edges)}条连接)")
    
    def load(self, directory: str):
        """从文件夹加载"""
        path = Path(directory)

        # 加载坐标网（v2 格式：{version, edges, nodes}；兼容 v1 纯列表）
        try:
            with open(path / "graph.json", "r", encoding="utf-8") as f:
                graph_data = json.load(f)
        except Exception as e:
            print(f"⚠️ graph.json 加载失败，回退空图: {e}")
            graph_data = []
        edges = graph_data if isinstance(graph_data, list) else graph_data.get("edges", [])
        node_attrs = graph_data.get("nodes", {}) if isinstance(graph_data, dict) else {}
        # 修复 (楚门 v1.6): 加载所有节点 — 孤立节点(无连接的新记忆)必须恢复, 否则 seed 无法激活
        for n, attrs in node_attrs.items():
            self.network.graph.add_node(n, **attrs)
        for edge in edges:
            self.network.graph.add_node(edge["from"], **node_attrs.get(edge["from"], {}))
            self.network.graph.add_node(edge["to"], **node_attrs.get(edge["to"], {}))
            self.network.graph.add_edge(
                edge["from"], edge["to"],
                weight=edge.get("weight", 0.5),
                shared=edge.get("shared", False),
            )

        # 加载内容库 (fix: 异常容错)
        try:
            with open(path / "content.json", "r", encoding="utf-8") as f:
                contents = json.load(f)
        except Exception as e:
            print(f"⚠️ content.json 加载失败，回退空库: {e}")
            contents = {}
        for exp_id, data in contents.items():
            exp = Experience(
                exp_id=exp_id,
                text=data["text"],
                source_url=data.get("source_url", ""),
                source_type=data.get("source_type", ""),
                source_year=data.get("source_year", 2020),
                scene_graph=data.get("scene_graph"),
                emotion_vector=data.get("emotion_vector"),
                created_at=data.get("created_at", ""),
                last_activated=data.get("last_activated", ""),
            )
            self.content.put(exp)

        # 加载锚点embedding + 回填到 Experience（v2 修复）
        anchor_file = path / "anchors.npy"
        id_file = path / "anchor_ids.json"
        if anchor_file.exists() and id_file.exists():
            try:
                anchors = np.load(anchor_file)
                with open(id_file, "r") as f:
                    exp_ids = json.load(f)
            except Exception as e:
                print(f"⚠️ anchors 加载失败: {e}")
                exp_ids, anchors = [], []
            for eid, anchor in zip(exp_ids, anchors):
                self._anchor_cache[eid] = anchor
                exp = self.content.get(eid)
                if exp is not None:
                    exp.anchor_embedding = np.asarray(anchor, dtype=np.float32)  # 回填

        print(f"✅ 记忆库已加载 ({len(self)}经验, {self.network.graph.number_of_edges()}条连接)")
    
    def stats(self) -> Dict:
        """统计信息"""
        return {
            "total_experiences": len(self.content),
            "total_connections": self.network.graph.number_of_edges(),
            "avg_degree": sum(d for _, d in self.network.graph.degree()) / max(len(self.network.graph), 1),
            "has_anchors": len(self._anchor_cache) > 0,
            "max_weight": max(
                (self.network.graph[u][v].get('weight', 0)
                 for u, v in self.network.graph.edges()),
                default=0
            ),
        }
    
    def __len__(self):
        return len(self.content)
    
    def __contains__(self, exp_id: str):
        return exp_id in self.content and exp_id in self.network

