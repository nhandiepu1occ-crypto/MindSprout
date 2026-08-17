
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
perception_monitor.py — 感知监测器 (感知系统 ③)

感知世界是最重要的 — 所以要监测它:
  1. 感知日志: 每次 enhance 记录 (时间/场景/动作/对象/器官/命中状态)
  2. 统计: 命中率 / 未命中对象清单 (→ 知识库热扩展)
  3. 报告: 每日/按需输出感知健康度

日志: self/perception_log.jsonl
"""
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(str(BASE))
LOG = BASE / "phase1" / "self" / "perception_log.jsonl"
CN_TZ = timezone(timedelta(hours=8))


def log_perception(scene: str, hits: list, enhanced: bool, source: str = ""):
    """记录一次感知事件"""
    rec = {
        "time": datetime.now(CN_TZ).isoformat(timespec="seconds"),
        "scene": scene[:120],
        "scene_hash": abs(hash(scene[:80])) % 100000,
        "hit": bool(hits),
        "objects": [h[2] for h in hits],
        "actions": [h[0] for h in hits],
        "organs": sorted({o for _, orgs, _, _ in hits for o in orgs}),
        "enhanced": enhanced,
        "source": source,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def report(days: int = 1, verbose: bool = True) -> dict:
    """感知健康报告: 命中率/未命中对象/器官分布"""
    if not LOG.exists():
        return {"total": 0, "hit_rate": 0, "miss_objects": [], "note": "无日志"}
    lines = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                lines.append(json.loads(line))
            except Exception:
                pass
    cutoff = datetime.now(CN_TZ) - timedelta(days=days)
    recent = [l for l in lines if datetime.fromisoformat(l["time"]).replace(tzinfo=CN_TZ) >= cutoff]
    if not recent:
        return {"total": 0, "hit_rate": 0, "miss_objects": [], "note": "该时段无记录"}

    hits = [l for l in recent if l["hit"]]
    rate = len(hits) / len(recent) * 100

    # 未命中对象: 从场景里提取名词 (简化: 场景里 2-4 字词频)
    miss_scenes = [l["scene"] for l in recent if not l["hit"]]
    candidates = []
    for sc in miss_scenes:
        for kw in re.findall(r"[\u4e00-\u9fa5]{2,4}", sc):
            # 排除常见虚词/动作词
            if kw in ["但是", "还是", "什么", "怎么", "觉得", "时候", "知道", "为什么",
                      "没有", "一个", "一下", "妈妈", "爸爸", "老师", "同学", "小雨", "大壮",
                      "我们", "你们", "他们", "自己", "真的", "然后", "突然", "昨天", "今天",
                      "明天", "学校", "回家", "上课", "下课", "放学", "吃饭", "睡觉", "作业",
                      "考试", "周末", "放假", "一起", "因为", "所以", "如果", "要是", "看见",
                      "听到", "闻到", "摸到", "吃到", "觉得", "感觉"]:
                continue
            candidates.append(kw)
    from collections import Counter
    top_miss = [w for w, _ in Counter(candidates).most_common(10)]

    # 器官分布
    organ_counter = Counter()
    for l in recent:
        for o in l.get("organs", []):
            organ_counter[o] += 1

    stats = {
        "total": len(recent),
        "hits": len(hits),
        "hit_rate": round(rate, 1),
        "top_miss_objects": top_miss,
        "organ_usage": dict(organ_counter.most_common()),
        "period": f"最近{days}天",
    }
    if verbose:
        print(f"📡 感知健康报告 ({stats['period']})")
        print(f"  感知事件: {stats['total']} | 命中: {stats['hits']} | 命中率: {stats['hit_rate']}%")
        print(f"  器官使用: {stats['organ_usage']}")
        print(f"  疑似未命中对象(候选): {stats['top_miss_objects']}")
    return stats


def suggest_new_objects(top_miss: list) -> list:
    """从未命中候选中挑出像对象的词 (2-3字, 出现在多个场景)"""
    out = []
    for w in top_miss:
        if 2 <= len(w) <= 3 and w not in out:
            out.append(w)
    return out


if __name__ == "__main__":
    report(days=7)
