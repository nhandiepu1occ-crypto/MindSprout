
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
embedder.py — 语义检索编码器 (P1, 2026-08-15)
复用 Humanizer 已加载的 3B 模型做 embedding (mean-pooling 末层 hidden, 2048维)
零额外显存: 与生成共用同一模型实例。降级: 无模型时 memory_bank 回退哈希编码。
"""
import numpy as np
import torch


class QwenEmbedder:
    """Qwen2.5 文本编码器: mean-pooling last hidden state (batch 编码)"""

    def __init__(self, humanizer):
        self.h = humanizer
        self._dim = None

    @property
    def dim(self):
        if self._dim is None:
            self._dim = self.h._model.config.hidden_size
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts):
        """→ np.ndarray (N, hidden_size) float32"""
        h = self.h
        h._ensure_loaded()
        tok = h._tokenizer
        model = h._model
        device = next(model.parameters()).device

        # 一次性 batch 编码 (无 padding 冲突: 用左 padding 或分别编码; 这里分批简单处理)
        embs = []
        with torch.no_grad():
            for t in texts:
                if not t or not t.strip():
                    embs.append(np.zeros(self.dim, dtype=np.float32))
                    continue
                inputs = tok(t, return_tensors="pt", padding=False).to(device)
                out = model(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[-1]  # (1, L, D)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                v = pooled[0].cpu().numpy().astype(np.float32)
                n = np.linalg.norm(v)
                embs.append(v / n if n > 1e-8 else v)
        return np.stack(embs)


def rebuild_anchors(memory_bank, embedder, out_dir):
    """全库重建 anchor embedding (2048维) → 覆盖 anchors.npy + anchor_ids.json"""
    import json
    from pathlib import Path
    ids = memory_bank.content.ids()
    texts = []
    for eid in ids:
        exp = memory_bank.content.get(eid)
        texts.append(exp.text if exp and exp.text else "")
    anchors = embedder.encode_batch(texts)
    for eid, a in zip(ids, anchors):
        memory_bank._anchor_cache[eid] = a
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "anchors.npy", anchors.astype(np.float32))
    (path / "anchor_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"✅ anchors 重建: {len(ids)} 条 × {anchors.shape[1]}维 → {path}")
    return len(ids)


class BgeEmbedder:
    """BGE-small-zh-v1.5 中文语义编码器 (CPU, 512维, ~400MB RAM)
    P1 v2: 专用语义模型, 替代 3B mean-pooling (抽象情感查询更准)
    """

    def __init__(self, model_path=None):
        import os
        from transformers import AutoModel, AutoTokenizer
        if not model_path:
            # 优先环境变量 MINSPROUT_BGE_PATH, 其次常见缓存路径, 最后自动下载
            cand = [
                os.environ.get("MINSPROUT_BGE_PATH", ""),
                os.path.expanduser(r"~/.cache/modelscope/hub/models/BAAI/bge-small-zh-v1___5"),
                os.path.expanduser(r"~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"),
            ]
            model_path = next((p for p in cand if p and os.path.isdir(p)), None)
            if not model_path:
                try:
                    from modelscope import snapshot_download
                    model_path = snapshot_download("BAAI/bge-small-zh-v1.5")
                except Exception:
                    raise FileNotFoundError(
                        "BGE 模型未找到。请设置 MINSPROUT_BGE_PATH 指向本地模型目录, "
                        "或先执行: pip install modelscope && python -c "
                        "\"from modelscope import snapshot_download; "
                        "print(snapshot_download('BAAI/bge-small-zh-v1.5'))\""
                    )
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self._dim = self.model.config.hidden_size
        self._device = "cpu"

    @property
    def dim(self):
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts):
        import torch
        embs = []
        with torch.no_grad():
            for t in texts:
                if not t or not t.strip():
                    embs.append(np.zeros(self._dim, dtype=np.float32))
                    continue
                inputs = self.tok(t, return_tensors="pt", padding=True, truncation=True, max_length=256)
                out = self.model(**inputs)
                hidden = out.last_hidden_state  # (1, L, D)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                v = pooled[0].numpy().astype(np.float32)
                n = np.linalg.norm(v)
                embs.append(v / n if n > 1e-8 else v)
        return np.stack(embs)

    def similarity(self, a: str, b: str) -> float:
        """文本相似度 (cosine) — 大脑语境关联用"""
        try:
            import numpy as np
            va = self.encode(a)
            vb = self.encode(b)
            if va is None or vb is None:
                return 0.0
            va = np.asarray(va, dtype=np.float32).reshape(-1)
            vb = np.asarray(vb, dtype=np.float32).reshape(-1)
            if va.shape != vb.shape:
                return 0.0
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1e-8 or nb < 1e-8:
                return 0.0
            return float(np.dot(va, vb) / (na * nb))
        except Exception:
            return 0.0
