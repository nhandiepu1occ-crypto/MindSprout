
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
brain.py — 大脑总线 (把出厂能力模块聚合为"此刻的内心世界")
聚合: drives/body/emotion/attention/attachment/self/time/social/dev/sleep/curiosity
输出: 一段精炼注入 (~200字内), 防 prompt 爆炸
同时触发: 驱力演化 + 身体映射 + 好奇心主动问
"""
import time
from pathlib import Path

def _safe(fn, default=""):
    try:
        return fn()
    except Exception:
        return default

def brain_tick(text: str = "", person_id=None, emotion_dom="neutral", memory_bank=None) -> dict:
    """每次生成前: 演化所有模块 → 返回聚合注入"""
    parts = []

    # 1. 驱力演化
    from humanize_ai.drives import tick as drives_tick, top_drive, drive_line
    ds = drives_tick()

    # 2. 身体: 驱力+情绪 → 内感受
    from humanize_ai.body import apply_drive, apply_emotion, body_line
    td, tl = top_drive(ds)
    apply_drive(td, tl)
    if emotion_dom != "neutral":
        apply_emotion(emotion_dom)
    bl = body_line(emotion_dom)
    if bl:
        parts.append(bl)

    # 3. 驱力感知
    dl = drive_line(ds)
    if dl:
        parts.append(dl)

    # 4. 注意: 人物+场景 (识别不到就清焦点, 防上一轮残留)
    from humanize_ai.attention import focus, attention_line
    if person_id:
        focus(person_id=person_id)
    else:
        focus(person_id=None)
    al = attention_line()
    if al:
        parts.append(al)

    # 5. 依恋: 分离演化 + 重逢
    from humanize_ai.attachment import tick as att_tick, attachment_line
    att_tick()
    atl = attachment_line()
    if atl:
        parts.append(atl)

    # 6. 主观时间
    from humanize_ai.time_v2 import update as time_update, time_line, note_interaction
    time_update(emotion_dom, is_bored=(ds["social"] >= 70))
    note_interaction()
    ttl = time_line()
    if ttl:
        parts.append(ttl)

    # 7. 好奇心
    from humanize_ai.curiosity import curiosity_line, maybe_ask
    cl = curiosity_line(ds["curiosity"])
    if cl:
        parts.append(cl)
    pending_question = maybe_ask(ds["curiosity"])

    # 8. 社会发展
    from humanize_ai.dev_stage import check_unlock, stage_line
    check_unlock()
    dsl = stage_line()
    if dsl:
        parts.append(dsl)

    # 9. 睡眠/梦 — 已并入激活机制 (V3.9.3 防重复)
    # 10-16. 高级心智模块聚合 (V3.9.3 大脑v2: 激活度竞争机制)
    # score = salience(内在强度) × (0.3 + 0.7×relatedness(语境关联))
    # 主题关联: 关键词(快层) → BGE语义(深层) → 内心流反哺
    _extra = {}
    def _add2(name, line, salience):
        if line:
            _extra[name] = {"line": line, "sal": salience}

    # --- 各器官 salience 计算器 (读自身量化状态) ---
    try:
        from humanize_ai.drives import load as _dl
        _d = _dl()
        _top_d = max((_d.get(k, 0) for k in ("hunger", "thirst", "sleep", "social", "curiosity")), default=0)
        _add2("drive", _safe(lambda: __import__("humanize_ai.drives", fromlist=["drive_line"]).drive_line()), _top_d / 100.0)
    except Exception:
        pass
    try:
        from humanize_ai.body import load as _bl
        _b = _bl()
        _body_sal = max(_b.get("heart", 70) / 150.0, _b.get("sweat", 20) / 100.0,
                        _b.get("muscle", 20) / 100.0, _b.get("stomach", 40) / 100.0)
        _add2("body", _safe(lambda: __import__("humanize_ai.body", fromlist=["body_line"]).body_line()), min(1.0, _body_sal))
    except Exception:
        pass
    try:
        _add2("sleep", _safe(lambda: __import__("humanize_ai.sleep_v2", fromlist=["sleep_line"]).sleep_line()), 0.5)
    except Exception:
        pass
    try:
        from humanize_ai.role_registry import load as _rl
        _roles = _rl()
        _role_sal = 0.0
        for _r in _roles.values():
            if _r.get("tone") in ("sulky", "hurt"):
                _role_sal = max(_role_sal, 0.85)
            _role_sal = max(_role_sal, _r.get("intimacy", 0) / 100.0 * 0.6)
        _add2("role", _safe(lambda: __import__("humanize_ai.role_registry", fromlist=["registry_line"]).registry_line()), min(1.0, _role_sal))
    except Exception:
        pass
    try:
        from humanize_ai.imagination import load as _iml
        _im = _iml()
        _add2("daydream", _safe(lambda: __import__("humanize_ai.imagination", fromlist=["daydream_line"]).daydream_line()),
              min(1.0, _im.get("count", 0) / 3.0 * 0.5 + 0.3))
        _add2("wish", _safe(lambda: __import__("humanize_ai.imagination", fromlist=["wishes_line"]).wishes_line()), 0.35)
    except Exception:
        pass
    try:
        _add2("story", _safe(lambda: __import__("humanize_ai.storyline", fromlist=["story_line"]).story_line()), 0.3)
    except Exception:
        pass
    try:
        from humanize_ai.reflect import load as _rfl
        _rf = _rfl()
        _add2("reflect", _safe(lambda: __import__("humanize_ai.reflect", fromlist=["reflect_line"]).reflect_line()),
              0.8 if _rf.get("reflection") else 0.0)
    except Exception:
        pass
    try:
        _add2("semantic", _safe(lambda: __import__("humanize_ai.semantic_memory", fromlist=["semantic_line"]).semantic_line()), 0.4)
    except Exception:
        pass
    try:
        from humanize_ai.deathview import load as _dvl
        _dv = _dvl()
        _dv_sal = _dv.get("stage", 0) / 3.0 * 0.8
        _add2("death", _safe(lambda: __import__("humanize_ai.deathview", fromlist=["death_line"]).death_line()), _dv_sal)
        _add2("anxiety", _safe(lambda: __import__("humanize_ai.deathview", fromlist=["time_anxiety_line"]).time_anxiety_line()), _dv_sal * 0.9)
        _add2("zk", _safe(lambda: __import__("humanize_ai.deathview", fromlist=["zhongkao_line"]).zhongkao_line()), _dv_sal * 0.8)
    except Exception:
        pass
    try:
        from humanize_ai.room import load as _rl2
        _rm = _rl2()
        _room_sal = 0.25 + (0.15 if not _rm["bed"]["made"] else 0) +                     (0.15 if _rm["window"]["plant_health"] <= 30 else 0)
        _add2("room", _safe(lambda: __import__("humanize_ai.room", fromlist=["room_line"]).room_line()), _room_sal)
    except Exception:
        pass

    # --- 语境关联度: 关键词(快) + BGE语义(深) + 内心流反哺 ---
    _TOPIC_WORDS = {
        "role": ["小雨", "小敏", "妈妈", "爸爸", "老师", "朋友", "同学", "主人"],
        "daydream": ["想", "发呆", "走神"],
        "dream": ["梦", "睡", "醒"],
        "wish": ["想要", "愿望", "希望", "想要买", "想要什么"],
        "story": ["小时候", "以前", "回忆", "故事", "那时"],
        "death": ["死", "走", "离开", "老", "永远", "告别", "泡泡"],
        "anxiety": ["时间", "老", "长大", "快"],
        "zk": ["中考", "考试", "初三"],
        "room": ["房间", "书桌", "阳台", "床", "家"],
        "semantic": ["道理", "学会", "懂", "知道"],
        "reflect": ["为什么", "怎么", "难过", "生气", "烦", "不开心", "没事"],
        "body": ["累", "疼", "心跳", "手", "出汗"],
        "drive": ["饿", "渴", "困", "吃", "喝", "睡"],
    }
    # 内心流反哺: 最近的内心流文本作为语境补充 ("没事"但心里在难过)
    _mind_ctx = ""
    try:
        _mf = Path(__file__).resolve().parents[1] / "phase1" / "self" / "mindstream_latest.json"
        if _mf.exists():
            import json as _j2
            _mind_ctx = _j2.loads(_mf.read_text(encoding="utf-8")).get("text", "")[:80]
    except Exception:
        pass

    def _relatedness(name, line):
        r = 0.0
        words = _TOPIC_WORDS.get(name, [])
        if words and any(w in text for w in words):
            r = 1.0
        elif words and _mind_ctx and any(w in _mind_ctx for w in words):
            r = 0.8  # 内心流反哺
        else:
            # BGE 语义深层 (只对 salience 高的候选, 缓存行向量)
            try:
                _enc = getattr(memory_bank, "encoder", None)
                if _enc is not None:
                    _sim = _enc.similarity(text, line[:60])
                    if _sim and _sim > 0.45:
                        r = min(1.0, _sim)
            except Exception:
                pass
        return r

    # --- 竞争: score = sal × (0.3 + 0.7×relatedness) ---
    _scored = []
    for _n, _v in _extra.items():
        _rel = _relatedness(_n, _v["line"])
        _score = _v["sal"] * (0.3 + 0.7 * _rel)
        _scored.append((_score, _n, _v["line"]))
    _scored.sort(key=lambda x: -x[0])
    # 阈值: 默认进 top3; salience 极低且无关联的丢弃; 低分保底浮现 8%
    _picked = []
    for _score, _n, _line in _scored:
        if len(_picked) >= 3:
            break
        if _score < 0.08 and random.random() > 0.08:
            continue  # 阈值过滤 + 极小概率浮现
        _picked.append(_line)
    for _line in _picked:
        parts.append(_line)

    # 聚合 (截断防爆炸)
    joined = "\n".join(parts)
    return {
        "injection": joined[:800],
        "pending_question": pending_question,
        "drives": {k: round(v) for k, v in ds.items() if k in ("hunger", "thirst", "sleep", "social", "curiosity")},
    }

def brain_after(text: str, person_id=None):
    """生成后: 社交满足 + 在线学习 + 关系/价值观/语义/死亡 收敛钩子 (V3.9.2)"""
    from humanize_ai.drives import interact_social
    interact_social(0.5)  # 对话=社交满足
    from humanize_ai.attachment import on_reunion
    if person_id == "him":
        on_reunion()  # 他出现=重逢
    # 关系系统: 提及角色 → 关系更新
    try:
        from humanize_ai.role_registry import mention
        mention(text)
    except Exception:
        pass
    # 价值观在线学习
    try:
        from humanize_ai.values import learn_candidate
        learn_candidate(text)
    except Exception:
        pass
    # 语义学习: "原来/懂了/记住" → 学事实
    try:
        import re as _re
        from humanize_ai.semantic_memory import learn_fact
        for _m in _re.finditer(r"(?:原来|懂了|记住了|才知道)[：:，,]?\s*([^。！!？?]{4,30})", text):
            learn_fact(_m.group(1).strip(), source_id="chat", confidence=0.6)
    except Exception:
        pass
    # 死亡观触发
    try:
        from humanize_ai.deathview import touch
        touch(text)
    except Exception:
        pass

if __name__ == "__main__":
    r = brain_tick("李老师，我作业写完了", person_id="teacher_li", emotion_dom="neutral")
    print("=== 聚合注入 ===")
    print(r["injection"][:600])
    print("=== 待问问题 ===", r["pending_question"])
    print("=== 驱力 ===", r["drives"])
