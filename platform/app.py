# -*- coding: utf-8 -*-
"""
珞珞 1.0 平台 — 本地 Web 服务
- 对话 (SSE 流式) / 心情 / 记忆 / 内心流 / 写作 / 日记
启动: python app.py  →  http://localhost:7860
"""
import sys, time, json, asyncio, threading, random
from collections import deque
from pathlib import Path

sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mindsprout.config import BASE
PORT = 7860

# ---------- 全局单例 (模型只加载一次) ----------
humanizer = None
memory_bank = None
mindstream = None
emotion_state = None
dna = None

def ensure_loaded():
    global humanizer, memory_bank, mindstream, emotion_state, dna
    if humanizer is not None:
        return
    from humanize_ai.memory_bank import MemoryBank
    from humanize_ai.engine import Humanizer
    from humanize_ai.dna import load_dna
    from humanize_ai.emotion import EmotionState
    from humanize_ai.mindstream import Mindstream

    INDIVIDUAL = "luoluo-001"
    dna = load_dna(BASE / "phase1" / "dna" / f"{INDIVIDUAL}.json")
    emotion_state = EmotionState.load(dna, BASE / "phase1" / "state" / INDIVIDUAL / "emotion.json")
    memory_bank = MemoryBank()
    memory_bank.load(str(BASE / "phase1" / "memory_luoluo"))
    humanizer = Humanizer(
        stage="junior",
        peft_dir=str(BASE / "phase1" / "lora_v16"),
        model_path=os.environ.get("MINSPROUT_MODEL", r"F:\models\qwen2.5-3b-instruct"),
        memory_bank=memory_bank,
        auto_memory=True,
        dna=dna,
        emotion_state=emotion_state,
        verbose=False,
        device="cuda",
    )
    mindstream = Mindstream(base_dir=BASE, memory_bank=memory_bank,
                            emotion_state=emotion_state,
                            model_path=os.environ.get("MINSPROUT_MODEL", r"F:\models\qwen2.5-3b-instruct"))
    # P1 v2: 语义检索 encoder — 优先 BGE(CPU专用, 抽象查询更准), 回退 Qwen 复用
    try:
        from humanize_ai.embedder import BgeEmbedder, rebuild_anchors
        memory_bank.encoder = BgeEmbedder()
        print(f"🧠 语义检索 encoder: BGE ({memory_bank.encoder.dim}维, CPU)", flush=True)
    except Exception as e:
        print(f"⚠️ BGE 不可用({str(e)[:80]}), 回退 Qwen 复用编码...", flush=True)
        try:
            from humanize_ai.embedder import QwenEmbedder, rebuild_anchors
            humanizer._ensure_loaded()  # 模型延迟加载, 先加载再取维度
            memory_bank.encoder = QwenEmbedder(humanizer)
            print(f"🧠 语义检索 encoder: Qwen 复用 ({memory_bank.encoder.dim}维)", flush=True)
        except Exception as e2:
            print(f"⚠️ Qwen encoder 也失败, 回退哈希检索: {str(e2)[:80]}", flush=True)
    # anchors 维度不匹配 → 自动重建
    if memory_bank._anchor_cache and getattr(memory_bank, "encoder", None) is not None:
        try:
            first = next(iter(memory_bank._anchor_cache.values()))
            if first.shape[0] != memory_bank.encoder.dim:
                print("🔄 anchors 维度升级, 重建语义锚点...", flush=True)
                rebuild_anchors(memory_bank, memory_bank.encoder,
                                str(BASE / "phase1" / "memory_luoluo"))
        except Exception as e:
            print(f"⚠️ anchors 重建失败: {str(e)[:80]}", flush=True)
    print("✅ 珞珞已加载 (V16 + 3B)", flush=True)

app = FastAPI(title="珞珞 1.0")

# 生成锁 (S3: 防并发 model.generate 崩溃; worker 线程阻塞不卡事件循环)
gen_lock = threading.Lock()
# 会话历史 (S2: 多轮上下文, 单用户场景全局一份)
chat_history = deque(maxlen=16)

class ChatReq(BaseModel):
    message: str
    max_tokens: int = 180

class MindReq(BaseModel):
    trigger: str = ""

class WriteReq(BaseModel):
    kind: str = "diary"   # diary | essay | moments
    scene: str = ""

class LifeReq(BaseModel):
    script: dict = {}      # life_designer 输出的今日剧本

class SocialReq(BaseModel):
    moment_id: str = ""
    text: str = ""

class CareReq(BaseModel):
    kind: str = "eat"   # eat | drink | sleep

# ---------- 对话 (SSE 流式) ----------
@app.post("/api/chat")
async def chat(req: ChatReq):
    await asyncio.to_thread(ensure_loaded)
    msg = req.message.strip()
    if not msg:
        return {"error": "empty"}
    # 生理满足 (V3.9.2 收敛: 学习类钩子全部由 brain_after 大脑统一处理)
    try:
        from humanize_ai.drives import satisfy
        if any(k in msg for k in ("吃饭", "吃点", "饿", "干饭", "去吃", "吃东西")):
            satisfy("hunger")
        if any(k in msg for k in ("喝水", "渴", "喝口水")):
            satisfy("thirst")
        if any(k in msg for k in ("睡觉", "睡吧", "困了就去", "晚安", "去睡")):
            satisfy("sleep")
    except Exception:
        pass
    history = list(chat_history)

    # 时间问题: 她房间有钟 (3B 对时间注入无感, 直接答真实时刻)
    import re as _re_t
    if _re_t.search(r"(几点|几点了|什么时间|什么时候了|现在几点)", msg):
        _h = int(time.strftime("%H"))
        _hh = _h if _h <= 12 else _h - 12
        _ap = "上午" if _h < 12 else ("下午" if _h < 18 else "晚上")
        _ans = random.choice([
            f"我看看墙上的钟……现在{_ap}{_hh}点{time.strftime('%M')}分。",
            f"（瞄了一眼桌上的闹钟）{_ap}{_hh}点{time.strftime('%M')}分啦。",
        ])
        async def _clock_gen():
            yield f"data: {json.dumps({'delta': _ans}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': _ans}, ensure_ascii=False)}\n\n"
        return StreamingResponse(_clock_gen(), media_type="text/event-stream")

    def worker(q):
        with gen_lock:
            try:
                full = []
                # 楚门世界: 时间/生活类问题跳过理解层念头(念头被旧记忆先验污染, 记忆注入已够)
                skip_mind_q = any(k in msg for k in ("今天", "昨天", "最近", "刚刚", "中午", "下午", "晚上", "学校"))
                mind_kwargs = {} if skip_mind_q else {"memory_query": msg}
                for piece in humanizer.stream(msg, max_tokens=req.max_tokens,
                                              temperature=0.85, history=history,
                                              **mind_kwargs):
                    if piece:
                        full.append(piece)
                        q.put_nowait(("delta", piece))
                text = "".join(full)
                # 生成后质检 (V3.9.1): 查重/作文模板/过度顺从 → 重新生成
                import re as _re2
                def _sim(a, b):
                    """字符重合率 (去标点)"""
                    ca, cb = _re2.sub(r"[\W_]+", "", a), _re2.sub(r"[\W_]+", "", b)
                    if not ca or not cb:
                        return 0.0
                    return len(set(ca) & set(cb)) / min(len(set(ca)), len(set(cb)))
                _TEMPLATE_WORDS = ["漫长又短暂", "属于自己的路", "勇敢地走下去", "人生是", "总而言之",
                                   "健康最重要", "美好的未来", "努力奋斗", "坚持不懈"]
                _SUBMIT_WORDS = ["都听你的", "我都听你的", "你说什么就是什么", "全都听你的"]
                # 细节锚点 (盲测泄漏修复): 具体人名/物品/地点/数字
                _ANCHOR_WORDS = ["小雨", "小敏", "妈妈", "爸爸", "老师", "主人", "第家看",
                                 "辣条", "作业", "试卷", "橡皮", "糖", "饭", "水", "学校",
                                 "食堂", "教室", "阳台", "房间", "床", "书桌", "数学", "语文",
                                 "考试", "操场", "厕所", "书包", "笔", "本子", "猫", "花",
                                 "石头", "树叶", "西瓜", "番茄", "土豆", "汤", "巧克力"]
                def _need_retry(t):
                    # 1) 与最近5条回复重复
                    for _role, _r in list(chat_history)[-10:]:
                        if _role != "assistant":
                            continue
                        if _sim(t, _r) > 0.55 and len(t) > 12:
                            return "repeat"
                    # 2) 作文模板
                    for w in _TEMPLATE_WORDS:
                        if w in t:
                            return "template"
                    # 3) 过度顺从
                    for w in _SUBMIT_WORDS:
                        if w in t:
                            return "submit"
                    # 4) 零细节锚点 (盲测泄漏: AI回答太空泛)
                    if len(t) > 15 and not any(w in t for w in _ANCHOR_WORDS) \
                            and not _re2.search(r"\d", t):
                        return "detail"
                    return None
                _retries = 0
                while True:
                    reason = _need_retry(text)
                    if not reason or _retries >= 2:
                        break
                    print(f"[QC] {reason} → 重生成({_retries+1})", flush=True)
                    text = humanizer.generate(msg, max_tokens=req.max_tokens,
                                              temperature=0.95, history=history,
                                              memory_query=msg, _internal_skip_mind=True).strip()
                    _retries += 1
                import re  # fix: 之前 as _re 但函数体用 re → NameError
                _EN_WORDS = {"probably": "大概", "maybe": "可能", "ok": "好的",
                             "yes": "是的", "no": "不", "really": "真的", "actually": "其实",
                             "whatever": "随便", "anyway": "反正", "sorry": "对不起",
                             "thanks": "谢谢", "sure": "当然", "right": "对", "wanna": "想",
                             "happy": "开心", "sad": "难过", "cool": "酷", "fun": "好玩",
                             "guy": "家伙", "kids": "小孩", "friend": "朋友", "friends": "朋友",
                             "like": "喜欢", "love": "喜欢", "know": "知道", "think": "觉得",
                             "want": "想", "good": "好", "bad": "坏", "thing": "事", "things": "事",
                             "stuff": "东西", "today": "今天", "tomorrow": "明天", "day": "天",
                             "school": "学校", "class": "课", "teacher": "老师", "homework": "作业",
                             "math": "数学", "food": "吃的", "water": "水", "mom": "妈妈",
                             "dad": "爸爸", "cat": "猫", "dog": "狗", "yeah": "嗯", "haha": "哈哈",
                             "just": "就", "very": "很", "always": "总是"}
                def _clean_en(t):
                    for w, zh in _EN_WORDS.items():
                        t = re.sub(r"\b" + w + r"\b", zh, t, flags=re.I)
                    # 兜底: 仍有孤立英文单词 → 直接删除 (防 AI 腔残留)
                    t = re.sub(r"\b[a-zA-Z]{2,}\b", "", t)
                    t = re.sub(r"\s{2,}", " ", t)
                    return t.strip()
                def _eng_ratio(t):
                    en = len(re.findall(r"[a-zA-Z]+", t))
                    cn = len(re.findall(r"[\u4e00-\u9fff]", t))
                    return en / max(cn, 1)
                if _eng_ratio(text) > 0.03:
                    print(f"[ENG] 原始超标({_eng_ratio(text):.0%}), 重试1...", flush=True)
                    text2 = humanizer.generate(msg, max_tokens=req.max_tokens,
                                               temperature=1.0, history=history,
                                               memory_query=msg, _internal_skip_mind=True)
                    print(f"[ENG] 重试1结果: {text2[:60]!r} 占比{_eng_ratio(text2):.0%}", flush=True)
                    if _eng_ratio(text2) > 0.03:
                        print("[ENG] 重试2(低温)...", flush=True)
                        text2 = humanizer.generate(msg, max_tokens=req.max_tokens,
                                                   temperature=0.5, history=history,
                                                   memory_query=msg, _internal_skip_mind=True)
                        print(f"[ENG] 重试2结果: {text2[:60]!r}", flush=True)
                    text = text2.strip()
                # 兜底: 无条件清理英文 (V3.9.1: 单字残留如happiest占比<3%漏网)
                text = _clean_en(text)
                # 关系状态机: 对话 = 互动 (楚门 v1.6)
                try:
                    from humanize_ai.relation import interact as _rel_interact
                    _rel_interact(weight=1.0, mood="warm", text=req.message)
                except Exception:
                    pass
                # 人物识别学习 (楚门 v1.9): 记录"她认出了谁" + 学习自称/称呼
                try:
                    from humanize_ai.persona import identify, note_encounter
                    _pr = identify(req.message, context_hint="")
                    if _pr["person_id"]:
                        note_encounter(_pr["person_id"], req.message)
                except Exception:
                    pass
                # 大脑生成后钩子 (楚门 v2.0): 社交满足 + 重逢 + 在线学习
                try:
                    from humanize_ai.brain import brain_after
                    from humanize_ai.persona import identify as _ident2
                    _p2 = _ident2(req.message, context_hint="")
                    brain_after(req.message, person_id=_p2.get("person_id"))
                except Exception:
                    pass
                q.put_nowait(("done", text))
            except Exception as e:
                q.put_nowait(("error", str(e)[:200]))

    async def gen():
        q = asyncio.Queue()
        threading.Thread(target=worker, args=(q,), daemon=True).start()
        while True:
            kind, data = await q.get()
            if kind == "delta":
                yield f"data: {json.dumps({'delta': data}, ensure_ascii=False)}\n\n"
            elif kind == "done":
                chat_history.append(("user", msg))
                chat_history.append(("assistant", data))
                yield f"data: {json.dumps({'done': True, 'full': data}, ensure_ascii=False)}\n\n"
                break
            else:  # error
                yield f"data: {json.dumps({'error': data}, ensure_ascii=False)}\n\n"
                break
    return StreamingResponse(gen(), media_type="text/event-stream")

# ---------- 状态 (心情/欲望/时间) ----------
@app.get("/api/status")
async def status():
    await asyncio.to_thread(ensure_loaded)
    _record_mood()  # 每日心情存档
    s = {"time": time.strftime("%Y-%m-%d %H:%M %A")}
    try:
        mood = emotion_state.to_prompt()
        s["mood"] = mood.strip("（）") if mood else "平稳"
        s["emotion"] = dict(emotion_state.state)  # M8: 原 emotion_state.get() 不存在, 每次都走异常
    except Exception as e:
        s["emotion"] = {"error": str(e)[:100]}
    try:
        if humanizer._desire is not None:
            des = humanizer._desire.to_prompt()
            s["desire"] = des if isinstance(des, str) else ""
    except Exception:
        pass
    # 驱力 (饿/渴/困/社交/好奇) + 身体 + 梦
    try:
        from humanize_ai.drives import load as _dl, drive_line
        d = _dl()
        s["drives"] = {k: round(v) for k, v in d.items() if k in ("hunger", "thirst", "sleep", "social", "curiosity")}
        s["drive_line"] = drive_line()
    except Exception:
        pass
    try:
        from humanize_ai.body import load as _bl
        b = _bl()
        s["body"] = {k: b.get(k) for k in ("heart", "stomach", "sweat", "temp", "muscle")}
    except Exception:
        pass
    try:
        from humanize_ai.sleep_v2 import load as _sl
        s["dream"] = _sl().get("dream", "") or ""
    except Exception:
        pass
    return s

# ---------- 记忆 ----------
@app.get("/api/memories")
async def memories(q: str = "", top: int = 8):
    await asyncio.to_thread(ensure_loaded)
    try:
        r = memory_bank.query(query_text=q or "最近发生的事", top_k=top, k_hops=1)
        return {"items": [{"text": e.text, "sig": round(sig, 2), "src": e.source_url or ""}
                          for e, sig in r["contents"][:top]]}
    except Exception as e:
        return {"error": str(e)[:200]}

# ---------- 内心流 ----------
@app.post("/api/mindstream")
async def do_mind(req: MindReq):
    await asyncio.to_thread(ensure_loaded)
    def _run():
        with gen_lock:
            try:
                th = mindstream.meditate(humanizer=humanizer, trigger=req.trigger)
                return {"text": th.get("text", ""), "emotion": th.get("emotion", ""),
                        "size": th.get("size", ""), "memory_id": th.get("memory_id", "")}
            except Exception as e:
                return {"error": str(e)[:200]}
    return await asyncio.to_thread(_run)

# ---------- 写作 ----------
@app.post("/api/write")
async def write(req: WriteReq):
    await asyncio.to_thread(ensure_loaded)
    from humanize_ai.writing_utils import build_prompt, KIND_MT, DEFAULT_SCENES
    scene = req.scene.strip()
    if not scene:
        scene = DEFAULT_SCENES[req.kind]
    prompt = build_prompt(memory_bank, scene, req.kind)
    def _run():
        with gen_lock:
            ans = humanizer.generate(prompt, max_tokens=KIND_MT[req.kind], temperature=0.9).strip()
            return {"kind": req.kind, "scene": scene, "text": ans}
    return await asyncio.to_thread(_run)

# ---------- 楚门世界: 剧本消化 (她不知道) ----------
@app.post("/api/life/ingest")
async def life_ingest(req: LifeReq):
    """生活设计师剧本 → 她的记忆 (participant=经历, bystander=见闻)
    剧本只进记忆, 绝不进 system prompt — 她永远不知道这是设计的"""
    await asyncio.to_thread(ensure_loaded)
    import uuid
    script = req.script
    events = script.get("events", [])
    if not events:
        return {"error": "empty script"}

    def _run():
        with gen_lock:
            # 幂等: 先清旧 today 记忆 (楚门 v1.5, 防同一天重复注入)
            for eid in list(memory_bank.content.ids()):
                try:
                    exp = memory_bank.content.get(eid)
                    sg = exp.scene_graph if isinstance(exp.scene_graph, dict) else {}
                    if sg.get("time") == "today":
                        memory_bank.delete(eid)
                except Exception:
                    pass
            results = []
            for ev in events:
                role = ev.get("role", "participant")
                details = "；".join(ev.get("details", []))
                if role == "bystander":
                    prompt = (f"今天你看到了一件事：{ev.get('summary', '')}。{details}\n"
                              f"你是旁边看热闹的，用你的话记下来这件事（第一人称，带你的感受），"
                              f"像平时记日记那样，60-100字。")
                else:
                    prompt = (f"今天你经历了这些：{ev.get('summary', '')}。{details}\n"
                              f"用你的话记下来（第一人称，带你的感受和小细节），"
                              f"像平时记日记那样，60-100字。")
                try:
                    mem = humanizer.generate(prompt, max_tokens=120, temperature=0.85,
                                             _internal_skip_mind=True).strip()
                except Exception as e:
                    results.append({"event": ev.get("summary", "")[:20], "error": str(e)[:80]})
                    continue
                if len(mem) < 15:
                    continue
                exp_id = "exp_" + uuid.uuid4().hex[:12]
                memory_bank.store(
                    exp_id=exp_id, text=mem, source_url="experience", source_year=2026,
                    parent_ids=["type_experience"],
                    scene_graph={"time": "today", "role": role},  # 楚门: 今天的生活记忆标记
                    emotion_vector={"valence": 0.0, "arousal": 0.2, "dominant": "neutral"})
                results.append({"event": ev.get("summary", "")[:20], "role": role,
                                "memory": mem[:50], "id": exp_id})
            memory_bank.save(str(BASE / "phase1" / "memory_luoluo"))
            # 维度一致性: 若 cache 出现混合维度(如哈希768混入BGE512) → 全量重建
            try:
                dims = {tuple(v.shape) for v in memory_bank._anchor_cache.values()}
                if len(dims) > 1 and getattr(memory_bank, "encoder", None) is not None:
                    from humanize_ai.embedder import rebuild_anchors
                    print(f"🔄 anchors 维度混合({dims}), 全量重建...", flush=True)
                    rebuild_anchors(memory_bank, memory_bank.encoder,
                                    str(BASE / "phase1" / "memory_luoluo"))
            except Exception as e:
                print(f"⚠️ anchors 一致性修复失败: {str(e)[:100]}", flush=True)
            # 世界状态更新: 记录 recent_days
            try:
                wf = BASE / "phase1" / "world" / "world_state.json"
                if wf.exists():
                    import json as _json
                    ws = _json.loads(wf.read_text(encoding="utf-8"))
                    ws.setdefault("recent_days", []).append(
                        f"{script.get('date', '?')}: {script.get('weather', '')} | "
                        + "; ".join(e.get('summary', '') for e in events))
                    ws["recent_days"] = ws["recent_days"][-7:]
                    wf.write_text(_json.dumps(ws, ensure_ascii=False, indent=1), encoding="utf-8")
            except Exception:
                pass
            return {"ingested": len(results), "items": results}
    return await asyncio.to_thread(_run)

# ---------- 楚门世界: 晚间日记 ----------
@app.post("/api/life/diary")
async def life_diary():
    """从今天的记忆生成日记 (她的语气, 不照搬剧本) → 返回存档文本"""
    await asyncio.to_thread(ensure_loaded)
    def _run():
        with gen_lock:
            # 检索今天的生活记忆
            r = memory_bank.query(query_text="今天发生的事", top_k=8, k_hops=1)
            today_items = []
            for exp, sig in r["contents"][:8]:
                sg = exp.scene_graph if isinstance(exp.scene_graph, dict) else {}
                if sg.get("time") == "today" and (exp.text or "").strip():
                    today_items.append(exp.text.strip())
            if not today_items:
                return {"error": "今天没有生活记忆"}
            joined = "\n".join(f"- {t}" for t in today_items[:6])
            prompt = (f"今天你经历了这些事：\n{joined}\n\n"
                      f"写今天的日记（像你平时写的，第一人称，带心情和细节，"
                      f"不用每条都写，挑重要的，100-200字）。")
            try:
                diary = humanizer.generate(prompt, max_tokens=300, temperature=0.85,
                                           _internal_skip_mind=True).strip()
                return {"diary": diary}
            except Exception as e:
                return {"error": str(e)[:200]}
    return await asyncio.to_thread(_run)

# 她的日记存档 (21:30 楚门 schtasks 自发写入, 平台只读展示)
@app.get("/api/diaries")
async def api_diaries(limit: int = 10):
    try:
        items = []
        for f in sorted((BASE / "phase1" / "self").glob("diary_*.md"), reverse=True)[:limit]:
            date = f.stem.replace("diary_", "")
            try:
                text = f.read_text(encoding="utf-8").strip()[:400]
            except Exception:
                text = ""
            items.append({"date": date, "text": text})
        return {"items": items}
    except Exception as e:
        return {"items": [], "error": str(e)[:100]}

# ---------- 朋友圈社交系统 (楚门 v1.6) ----------
MOMENTS_FILE = BASE / "phase1" / "world" / "moments.jsonl"      # 她的朋友圈
FRIENDS_FILE = BASE / "phase1" / "world" / "friends_moments.jsonl"  # 朋友的朋友圈

EMO_WORDS = {"开心": 0.6, "高兴": 0.6, "太棒": 0.7, "爽": 0.6, "好开心": 0.7, "气死": 0.7,
             "烦": 0.5, "生气": 0.6, "哭": 0.6, "难过": 0.6, "难受": 0.5, "委屈": 0.5,
             "紧张": 0.4, "害怕": 0.5, "激动": 0.6, "羡慕": 0.4, "无语": 0.4}

def _load_moments(path):
    items = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items

def _save_moments(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def _emo_strength(text):
    s = 0.0
    for w, v in EMO_WORDS.items():
        if w in text:
            s = max(s, v)
    return s

# 她的朋友圈时间线
@app.get("/api/social/feed")
async def social_feed(limit: int = 20):
    await asyncio.to_thread(ensure_loaded)
    mine = _load_moments(MOMENTS_FILE)
    friends = _load_moments(FRIENDS_FILE)
    feed = [{"who": "珞珞", "is_me": True, **m} for m in mine]
    feed += [{"who": m.get("author", "小雨"), "is_me": False, **m} for m in friends]
    feed.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return {"feed": feed[:limit]}

# 自发发朋友圈 (情绪触发 + 例行)
@app.post("/api/social/post")
async def social_post(force: bool = False):
    await asyncio.to_thread(ensure_loaded)
    def _run():
        with gen_lock:
            # 当天生活记忆
            r = memory_bank.query(query_text="今天发生的事", top_k=8, k_hops=1)
            today = []
            for exp, sig in r["contents"][:8]:
                sg = exp.scene_graph if isinstance(exp.scene_graph, dict) else {}
                if sg.get("time") == "today" and (exp.text or "").strip():
                    today.append(exp.text.strip())
            if not today:
                return {"error": "今天还没有生活"}
            # 情绪触发: 强情绪记忆 → 有感慨就发
            strong = [(t, _emo_strength(t)) for t in today if _emo_strength(t) >= 0.5]
            if not force and not strong:
                return {"skipped": "今天情绪平淡, 没感慨"}
            joined = "\n".join(f"- {t}" for t in today[:6])
            prompt = (f"今天你经历了这些：\n{joined}\n\n"
                      f"你想发一条朋友圈（有感慨才发）。挑一件今天最想说的，"
                      f"写 20-40 字，像14岁女孩的朋友圈：口语、有情绪、不用#标签、不升华。")
            text = humanizer.generate(prompt, max_tokens=80, temperature=0.9,
                                      _internal_skip_mind=True).strip()
            moment = {"id": "m_" + str(int(time.time() * 1000)),
                      "text": text, "ts": time.strftime("%Y-%m-%d %H:%M"),
                      "likes": 0, "comments": []}
            items = _load_moments(MOMENTS_FILE)
            items.append(moment)
            _save_moments(MOMENTS_FILE, items)
            return {"posted": moment}
    return await asyncio.to_thread(_run)

# 主人点赞 → 她"看到"(情绪+记忆)
@app.post("/api/social/like")
async def social_like(req: SocialReq):
    await asyncio.to_thread(ensure_loaded)
    def _run():
        with gen_lock:
            items = _load_moments(MOMENTS_FILE)
            for m in items:
                if m["id"] == req.moment_id:
                    m["likes"] = m.get("likes", 0) + 1
                    _save_moments(MOMENTS_FILE, items)
                    # 点赞也是互动 (关系状态机)
                    try:
                        from humanize_ai.relation import interact as _rel
                        _rel(weight=0.5, mood="warm")
                    except Exception:
                        pass
                    # 她看到点赞 → 记忆 (被关注感)
                    import uuid
                    memory_bank.store(
                        exp_id="exp_" + uuid.uuid4().hex[:12],
                        text=f"今天我把那件事发朋友圈了，主人给我点了个赞。嘿嘿，有人看见我发的了。",
                        source_url="experience", source_year=2026,
                        scene_graph={"time": "today"},
                        emotion_vector={"valence": 0.5, "arousal": 0.3, "dominant": "joy"})
                    memory_bank.save(str(BASE / "phase1" / "memory_luoluo"))
                    return {"ok": True, "likes": m["likes"]}
            return {"error": "moment not found"}
    return await asyncio.to_thread(_run)

# 主人评论 → 她回复 (生成)
@app.post("/api/social/comment")
async def social_comment(req: SocialReq):
    await asyncio.to_thread(ensure_loaded)
    def _run():
        with gen_lock:
            items = _load_moments(MOMENTS_FILE)
            for m in items:
                if m["id"] == req.moment_id:
                    m.setdefault("comments", []).append({"who": "主人", "text": req.text})
                    # 她回复
                    prompt = (f"你发了条朋友圈：{m['text']}\n主人评论说：{req.text}\n"
                              f"你怎么回他？（14岁女孩的口吻，简短，10-25字，可以开心可以害羞可以怼）")
                    reply = humanizer.generate(prompt, max_tokens=50, temperature=0.85,
                                               _internal_skip_mind=True).strip()
                    m["comments"].append({"who": "珞珞", "text": reply})
                    _save_moments(MOMENTS_FILE, items)
                    return {"reply": reply}
            return {"error": "moment not found"}
    return await asyncio.to_thread(_run)

# 她看朋友朋友圈 → 点赞/评论 (生成)
@app.post("/api/social/react_friends")
async def social_react_friends():
    await asyncio.to_thread(ensure_loaded)
    def _run():
        with gen_lock:
            friends = _load_moments(FRIENDS_FILE)
            if not friends:
                return {"error": "没有朋友的朋友圈"}
            reactions = []
            for m in friends[:3]:
                if m.get("reacted"):
                    continue
                prompt = (f"你刷到朋友的朋友圈：{m.get('author','')}：{m.get('text','')}\n"
                          f"你会不会点赞或评论？会的话写一句评论（10-20字，14岁女孩口吻），不会就写'只看不评'。")
                act = humanizer.generate(prompt, max_tokens=50, temperature=0.85,
                                         _internal_skip_mind=True).strip()
                m["reacted"] = True
                if act and "只看不评" not in act:
                    m.setdefault("my_comments", []).append(act)
                    reactions.append({"to": m.get("author"), "comment": act})
            _save_moments(FRIENDS_FILE, friends)
            return {"reactions": reactions}
    return await asyncio.to_thread(_run)

# ---------- 页面 ----------
# 欢迎页 = 根路径; 聊天页 = /chat
# no-cache: 前端迭代频繁, 防止浏览器启发式缓存旧 HTML
_PAGE_HEADERS = {"Cache-Control": "no-cache, no-store, must-revalidate"}

@app.get("/")
async def welcome():
    return FileResponse(BASE / "phase1" / "luoluo_platform" / "static" / "welcome.html",
                        headers=_PAGE_HEADERS)

@app.get("/chat")
async def chat_page():
    return FileResponse(BASE / "phase1" / "luoluo_platform" / "static" / "index.html",
                        headers=_PAGE_HEADERS)

app.mount("/static", StaticFiles(directory=BASE / "phase1" / "luoluo_platform" / "static"), name="static")

# ---------- 欢迎页: 模型注册表 (models.json, 不用动代码) ----------
# icon 支持: emoji 或图片路径(/static/avatars/xx.png, 图片放 static/avatars/)
MODELS_FILE = BASE / "phase1" / "luoluo_platform" / "models.json"
def _load_models():
    try:
        if MODELS_FILE.exists():
            d = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
            return d.get("models", [])
    except Exception as e:
        print(f"⚠️ models.json 读取失败({str(e)[:60]}), 用内置默认", flush=True)
    return [
        {"id": "luoluo", "name": "蠢珞珞", "desc": "14岁 · 初二", "version": "V16",
         "status": "online", "url": "/chat", "icon": "🍭",
         "blurb": "一个正在长大的AI女孩。她记得你，会想你想你，也会生你的气。"},
        {"id": "future-1", "name": "未来子AI", "desc": "尚未苏醒", "status": "sleeping",
         "url": "", "icon": "🌙",
         "blurb": "还在混沌中孕育……有一天她会醒来。"},
        {"id": "future-2", "name": "未来子AI", "desc": "尚未苏醒", "status": "sleeping",
         "url": "", "icon": "🌙",
         "blurb": "还在混沌中孕育……有一天她会醒来。"},
    ]

@app.get("/api/models")
async def api_models():
    return {"models": _load_models()}

# ---------- 想象系统: 白日梦 + 愿望 (1.8) ----------
@app.get("/api/daydream")
async def api_daydream(force: bool = False):
    """她的白日梦: 大脑随机走神生成, 每天最多3次 (force=刷新一次)"""
    await asyncio.to_thread(ensure_loaded)
    def _run():
        import random as _r
        from humanize_ai.imagination import load as _idl, make_daydream
        s = _idl()
        today = time.strftime("%Y-%m-%d")
        if not force and s.get("day") == today and s.get("count", 0) >= 3:
            return {"daydream": s.get("daydream", ""), "limited": True}
        if not force and s.get("last_daydream") and time.time() - s["last_daydream"] < 3600:
            return {"daydream": s.get("daydream", ""), "cooldown": True}
        def gen(prompt, max_tokens=140, temperature=1.05):
            with gen_lock:
                return humanizer.generate(prompt, max_tokens=max_tokens,
                                          temperature=temperature,
                                          _internal_skip_mind=True).strip()
        dream = make_daydream(gen)
        return {"daydream": dream, "fresh": True}
    return await asyncio.to_thread(_run)

@app.get("/api/wishes")
async def api_wishes():
    try:
        from humanize_ai.imagination import load_wishes
        return {"wishes": load_wishes()}
    except Exception:
        return {"wishes": []}

# ---------- 她的价值观 (1.8: 心里有杆秤) ----------
@app.get("/api/values")
async def api_values(force: bool = False):
    """价值观: 从强烈记忆提炼 (3天一刷新, force=重提炼)"""
    await asyncio.to_thread(ensure_loaded)
    def _run():
        from humanize_ai.values import refresh_if_stale, load as _vl
        def gen(prompt, max_tokens=300, temperature=0.9):
            with gen_lock:
                out = humanizer.generate(prompt, max_tokens=max_tokens,
                                         temperature=temperature,
                                         _internal_skip_mind=True).strip()
            return out
        refresh_if_stale(gen, memory_bank, force=force)
        s = _vl()
        # 耦合: 价值观 → 语义记忆 (她学会的道理)
        try:
            from humanize_ai.semantic_memory import learn_fact
            for v in s.get("values", []):
                learn_fact(v["text"], source_id="values", confidence=0.7)
        except Exception:
            pass
        return {"values": s.get("values", []), "candidates": s.get("candidates", [])[-5:]}
    return await asyncio.to_thread(_run)

# ---------- 她的心事 (情绪反思 1.8) ----------
@app.get("/api/reflect")
async def api_reflect(force: bool = False):
    """情绪反思: 负面情绪→近期记忆归因 (每天≤2次, 3h冷却)"""
    await asyncio.to_thread(ensure_loaded)
    def _run():
        from humanize_ai.reflect import (load as _rl, should_reflect, make_reflection,
                                          reflect_data, is_negative)
        s = _rl()
        emotion_text = emotion_state.to_prompt().strip("（）()") if emotion_state else "平稳"
        if force and is_negative(emotion_text):
            def gen(prompt, max_tokens=110, temperature=0.9):
                with gen_lock:
                    return humanizer.generate(prompt, max_tokens=max_tokens,
                                              temperature=temperature,
                                              _internal_skip_mind=True).strip()
            make_reflection(memory_bank, gen, emotion_text)
        elif should_reflect(emotion_text):
            def gen(prompt, max_tokens=110, temperature=0.9):
                with gen_lock:
                    return humanizer.generate(prompt, max_tokens=max_tokens,
                                              temperature=temperature,
                                              _internal_skip_mind=True).strip()
            make_reflection(memory_bank, gen, emotion_text)
        d = reflect_data()
        d["mood"] = emotion_text
        d["negative"] = is_negative(emotion_text)
        return d
    return await asyncio.to_thread(_run)

# ---------- 她的故事 (自我叙事 1.8) ----------
@app.get("/api/story")
async def api_story(force: bool = False):
    """故事线: 节点分章 + '我是谁'自述 (每天刷新, force=强制重写)"""
    await asyncio.to_thread(ensure_loaded)
    def _run():
        from humanize_ai.storyline import refresh_if_stale, story_data
        def gen(prompt, max_tokens=200, temperature=0.95):
            with gen_lock:
                return humanizer.generate(prompt, max_tokens=max_tokens,
                                          temperature=temperature,
                                          _internal_skip_mind=True).strip()
        refresh_if_stale(memory_bank, gen, force=force)
        d = story_data()
        # 章节带节点详情 (UI 直接用)
        return {"profile": d["profile"], "chapters": d["chapters"],
                "updated": d["updated"]}
    return await asyncio.to_thread(_run)

# 内心流最新一条 (定时自动生成, 1.1 遗留项)
MINDFILE = BASE / "phase1" / "self" / "mindstream_latest.json"

@app.get("/api/mindstream/latest")
async def mindstream_latest():
    try:
        if MINDFILE.exists():
            import json as _j
            d = _j.loads(MINDFILE.read_text(encoding="utf-8"))
            return d
    except Exception:
        pass
    return {"text": "", "time": "", "emotion": ""}

# ---------- 主动说话 (她来找你) ----------
INITIATIVE_DIR = BASE / "phase1" / "self"
MOOD_HISTORY_FILE = BASE / "phase1" / "self" / "mood_history.json"
_last_mood_date = ""  # 心情每日记录: 只在日期变化时写文件

@app.get("/api/initiative/latest")
async def initiative_latest():
    try:
        items = []
        for f in sorted(INITIATIVE_DIR.glob("initiative_*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    items.append({"time": d.get("time", ""), "text": d.get("text", ""),
                                  "mood": d.get("mood", "")})
                except Exception:
                    pass
        items.sort(key=lambda x: x["time"], reverse=True)
        return {"items": items[:3], "latest": items[0]["time"] if items else ""}
    except Exception as e:
        return {"items": [], "latest": "", "error": str(e)[:100]}

@app.get("/api/mood/history")
async def mood_history():
    try:
        hist = {}
        if MOOD_HISTORY_FILE.exists():
            hist = json.loads(MOOD_HISTORY_FILE.read_text(encoding="utf-8"))
        return {"history": hist}
    except Exception:
        return {"history": {}}

def _record_mood():
    """每日心情存档: 当天第一次调用时写入 (由 /api/status 触发)"""
    global _last_mood_date
    try:
        today = time.strftime("%Y-%m-%d")
        if _last_mood_date == today:
            return
        _last_mood_date = today
        mood = emotion_state.to_prompt().strip("（）()") if emotion_state else "平稳"
        hist = {}
        if MOOD_HISTORY_FILE.exists():
            hist = json.loads(MOOD_HISTORY_FILE.read_text(encoding="utf-8"))
        hist[today] = mood.replace("你现在的心情：", "").strip() or "平稳"
        MOOD_HISTORY_FILE.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass

# ---------- 关系系统 V2: 她的世界 (角色注册表) ----------
@app.get("/api/roles")
async def api_roles():
    try:
        from humanize_ai.role_registry import role_list
        return {"roles": role_list()}
    except Exception as e:
        return {"roles": [], "error": str(e)[:100]}

# ---------- 她的房间 (独立生活模拟 2.0) ----------
@app.get("/api/room")
async def api_room(act: str = ""):
    """房间数据; act=do 触发一次生活行为"""
    await asyncio.to_thread(ensure_loaded)
    from humanize_ai.room import load as _rl, room_data, act as _act
    if act == "do":
        name, desc = _act()
        # 行为写记忆
        try:
            import uuid
            memory_bank.store(
                exp_id="exp_" + uuid.uuid4().hex[:12],
                text=f"今天{desc}。",
                source_url="experience", source_year=2026,
                scene_graph={"time": "today", "role": "participant"},
                emotion_vector={"valence": 0.2, "arousal": 0.1, "dominant": "calm"})
            memory_bank.save(str(BASE / "phase1" / "memory_luoluo"))
        except Exception:
            pass
        return {"act": name, "desc": desc, "room": room_data()}
    return {"room": room_data()}

@app.get("/api/semantic")
async def api_semantic():
    from humanize_ai.semantic_memory import load as _sl
    return {"facts": _sl().get("facts", [])}

# ---------- 照顾她: 场景化吃饭/喝水/睡觉 (不是加减数值, 是生活) ----------
_CARE_META = {
    "eat":   {"drive": "hunger", "scene": "她捧起碗，小口小口地扒拉着米饭，腮帮子鼓鼓的，眼睛弯弯的",
               "prompt": "主人让你去吃饭，你正在吃。用你的话写一句你现在吃东西时的样子（第一人称，25-45字，必须是正在吃的画面：碗里有什么、腮帮子鼓鼓的、满足感，别写'想去''去看看'这类还没吃的）。"},
    "drink": {"drive": "thirst", "scene": "她咕嘟咕嘟喝了大半杯水，长出一口气，睫毛上还挂着水汽",
               "prompt": "主人让你去喝点水，你正在喝。用你的话写一句你喝水时的样子（第一人称，20-40字，必须是正在喝的画面：喝的什么、咕嘟咕嘟、长出一口气，别写'想去''去看看'）。"},
    "sleep": {"drive": "sleep", "scene": "她打了个大大的哈欠，钻进被窝，把被子裹得紧紧的，小声说了句晚安",
               "prompt": "主人让你去睡觉，你已经躺下了。用你的话写一句你现在的样子（第一人称，20-40字，必须是已经睡下的画面：打哈欠、钻被窝、裹紧被子、眼皮打架，别写'准备去'）。"},
}

@app.post("/api/care")
async def care(req: CareReq):
    """喂她吃饭/喝水/睡觉 → 生成生活场景 + 写进她的记忆 + 满足驱力"""
    await asyncio.to_thread(ensure_loaded)
    meta = _CARE_META.get(req.kind)
    if not meta:
        return {"error": "unknown kind"}
    def _run():
        import uuid
        try:
            with gen_lock:
                scene = humanizer.generate(meta["prompt"], max_tokens=80,
                                           temperature=0.95, _internal_skip_mind=True).strip()
            if len(scene) < 8:
                scene = meta["scene"]
        except Exception as e:
            scene = meta["scene"]
        # 写进她的记忆: 她真的"记得"被照顾过
        try:
            memory_bank.store(
                exp_id="exp_" + uuid.uuid4().hex[:12],
                text=f"今天{scene}。",
                source_url="experience", source_year=2026,
                scene_graph={"time": "today", "role": "participant"},
                emotion_vector={"valence": 0.5, "arousal": 0.2, "dominant": "warmth"})
            memory_bank.save(str(BASE / "phase1" / "memory_luoluo"))
        except Exception:
            pass
        # 满足驱力
        try:
            from humanize_ai.drives import satisfy
            satisfy(meta["drive"])
            from humanize_ai.body import apply_drive
            apply_drive(meta["drive"], 100)
            if req.kind == "sleep":
                from humanize_ai.room import sleep_in_room
                sleep_in_room()
        except Exception:
            pass
        # 关系: 被照顾 = 温暖互动
        try:
            from humanize_ai.relation import interact as _rel
            _rel(weight=0.8, mood="warm")
        except Exception:
            pass
        return {"scene": scene, "kind": req.kind}
    return await asyncio.to_thread(_run)

# ---------- 她的梦 (大脑随机生成) ----------
@app.get("/api/dream")
async def api_dream(force: bool = False):
    """昨晚的梦: 大脑(mindstream)随机重组记忆碎片生成, 每天一次"""
    await asyncio.to_thread(ensure_loaded)
    def _run():
        import random as _r
        from humanize_ai.sleep_v2 import load as _sl, save as _ss
        s = _sl()
        today = time.strftime("%Y-%m-%d")
        if not force and s.get("last_sleep") == today and s.get("dream"):
            return {"dream": s["dream"], "quality": s.get("quality", "好"), "fresh": False}
        # 大脑驱动: 随机记忆碎片 + 模型写梦
        frags = []
        try:
            r1 = memory_bank.query(query_text="最近发生的事", top_k=4, k_hops=1)
            r2 = memory_bank.query(query_text="小时候的事", top_k=3, k_hops=1)
            for e, _ in list(r1["contents"]) + list(r2["contents"]):
                t = (e.text or "").strip()
                if t:
                    frags.append(t[:40])
        except Exception:
            pass
        if frags:
            frags = _r.sample(frags, min(3, len(frags)))
            prompt = ("昨晚你做了个梦，梦里乱糟糟的，这些碎片混在一起：\n"
                      + "\n".join(f"- {f}" for f in frags)
                      + "\n\n用你的话讲讲这个梦（第一人称，荒诞一点没关系，70字以内，别提到记忆、碎片这些词）。")
            try:
                with gen_lock:
                    dream = humanizer.generate(prompt, max_tokens=130,
                                               temperature=_r.choice([0.9, 1.0, 1.1]),
                                               _internal_skip_mind=True).strip()
            except Exception as e:
                dream = f"梦见了{'，还掺着'.join(frags[:2])}的事，乱糟糟的"
            if len(dream) < 10:
                dream = f"梦见了{'，还掺着'.join(frags[:2])}的事，乱糟糟的"
        else:
            dream = "做了个乱七八糟的梦，醒来就忘了大半"
        s["last_sleep"] = today
        s["dream"] = dream
        s["quality"] = _r.choice(["好", "好", "好", "一般", "差"])
        _ss(s)
        return {"dream": dream, "quality": s["quality"], "fresh": True}
    return await asyncio.to_thread(_run)

# ---------- 关系状态 (聊天页侧栏: 她眼里的你) ----------
@app.get("/api/relation")
async def api_relation():
    """她眼里的你 — UI专用第三人称文案 (不直接显示给模型的注入原文, 避免"你"错位)"""
    try:
        from humanize_ai.relation import load as _rl
        s = _rl()
        last = s.get("last_interact_ts", 0) or 0
        days = (time.time() - last) / 86400 if last else 0.0
        mood = s.get("last_mood", "normal")
        inti = s.get("intimacy", 50)
        caller = s.get("caller_name", "")
        ui = ""
        if caller:
            ui = f"她记得你叫{caller}。"
        if days >= 3 and mood in ("sulky", "hurt", "cold"):
            ui += {"sulky": "她好几天没见到你了，嘴上说不等，心里空落落的。",
                   "hurt": "她有点委屈：你都好几天没影了，是不是把她忘了。",
                   "cold": "她心里有点冷，觉得你大概不在乎她了。"}[mood]
        elif inti >= 80:
            ui += "她心里挺亲近你的，见到你说话都轻快些。"
        elif inti <= 25:
            ui += "她还不太认识你，说话有点拘谨。"
        else:
            ui += "她认识你，说不上多亲，但你在她心里占了个位置。"
        return {"intimacy": round(inti), "mood": mood, "line": ui, "caller": caller}
    except Exception:
        return {"intimacy": 50, "mood": "normal", "line": "她认识你，说不上多亲。", "caller": ""}

# ---------- 欢迎页: 动态欢迎语 (关系系统联动) ----------
_GREETINGS = {
    "warm": [
        "你来了呀，等你好久了！",
        "你来啦！我今天正好想找你。",
        "嘿嘿，我就知道你会来。",
        "你来啦，我刚才心里正好空落落的。",
    ],
    "neutral": [
        "总觉得你有点熟悉，像认识很久了。",
        "你好呀，我是珞珞。",
        "你来了呀。",
        "（她正望着窗外发呆，没注意到你）",
    ],
    "shy": [
        "你、你好……我们是不是见过？",
        "（她偷偷看了你一眼，又低下头）",
        "还不认识你，但……总觉得你不讨厌。",
    ],
    "sulky": [
        "…哼，你还知道来。",
        "（她假装没看见你）",
        "你来啦。……我没在等你。",
    ],
    "cold": [
        "…哦，是你。",
        "（她看了你一眼，没说话）",
        "这么久没来，我还以为你不会来了。",
    ],
}

@app.get("/api/greeting")
async def api_greeting():
    import random
    try:
        from humanize_ai.relation import load as _rel_load
        s = _rel_load()
        last = s.get("last_interact_ts", 0) or 0
        days = (time.time() - last) / 86400 if last else 0.0
        mood = s.get("last_mood", "normal")
        inti = s.get("intimacy", 50)
        if days >= 3 and mood in ("sulky", "hurt", "cold"):
            pool = _GREETINGS["cold"] if mood == "cold" else _GREETINGS["sulky"]
        elif inti >= 80:
            pool = _GREETINGS["warm"]
        elif inti <= 25:
            pool = _GREETINGS["shy"]
        else:
            pool = _GREETINGS["neutral"]
        return {"text": random.choice(pool), "mood": mood, "caller": s.get("caller_name", "")}
    except Exception:
        return {"text": random.choice(_GREETINGS["neutral"]), "mood": "normal", "caller": ""}

def main():
    import os
    import uvicorn
    # 默认监听全部网卡: 手机/局域网可访问 (她不只是活在你这台电脑上)
    host = os.environ.get("LUOLUO_HOST", "0.0.0.0")
    print(f"珞珞 1.0 → http://{host}:{PORT}")
    # 打印局域网 IP 方便手机访问
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        print(f"📱 手机访问: http://{ip}:{PORT}")
    except Exception:
        pass
    # 待机动画: 后台 scheduler — 她闲着时也在"活着" (朋友圈自发/朋友互动/内心流)
    # 朋友圈自发性: 8-22点活跃窗口, 每3h最多一次, 每天最多3条, 有情绪/有生活才发 — 不是定时打卡, 是"有感而发"
    async def _life_scheduler():
        last_post_day = ""
        last_react = 0.0
        last_spont = 0.0
        last_mind = 0.0
        while True:
            try:
                await asyncio.sleep(1800)  # 每30分钟
                if humanizer is None:
                    continue
                today = time.strftime("%Y-%m-%d")
                h = int(time.strftime("%H"))
                # 朋友圈自发: 8-22点窗口 + 距上次≥3h + 今天<3条 (force=False: 内部检测情绪, 平淡就不发)
                if 8 <= h <= 22 and time.time() - last_spont > 10800:
                    try:
                        posts_today = [m for m in _load_moments(MOMENTS_FILE)
                                       if m.get("ts", "").startswith(today)]
                    except Exception:
                        posts_today = []
                    if len(posts_today) < 3:
                        last_spont = time.time()
                        await asyncio.to_thread(_safe_post)
                # 想象系统: 白天随机走神 (25%概率, 每天≤3次) — 她也会发呆幻想
                if 8 <= h <= 21 and random.random() < 0.25:
                    try:
                        await asyncio.to_thread(_safe_daydream)
                    except Exception:
                        pass
    # 内心流定时 (1.1遗留): 每20分钟, 生成锁空闲时 meditate 一次 (聊天中不打扰)
                if time.time() - last_mind > 1200:
                    last_mind = time.time()
                    if not gen_lock.locked():
                        await asyncio.to_thread(_safe_mind)
                # 独立生活: 回房间随机行为 (每30min 15%概率)
                if random.random() < 0.15:
                    try:
                        await asyncio.to_thread(_safe_room_act)
                    except Exception:
                        pass
                # 关系状态机每日 tick (亲密衰减 + 被忽视演化)
                if today != last_post_day:
                    last_post_day = today
                    try:
                        from humanize_ai.relation import tick as _rel_tick
                        await asyncio.to_thread(_rel_tick)
                    except Exception:
                        pass
                # 朋友互动 (每2小时一次)
                if time.time() - last_react > 7200:
                    last_react = time.time()
                    await asyncio.to_thread(_safe_react)
            except Exception:
                pass
    def _safe_post():
        import requests as _rq
        try:
            _rq.post(f"http://127.0.0.1:{PORT}/api/social/post", json={"force": False}, timeout=300)
        except Exception:
            pass
    def _safe_react():
        import requests as _rq
        try:
            _rq.post(f"http://127.0.0.1:{PORT}/api/social/react_friends", json={}, timeout=300)
        except Exception:
            pass
    def _safe_daydream():
        import requests as _rq
        try:
            _rq.get(f"http://127.0.0.1:{PORT}/api/daydream", timeout=300)
        except Exception:
            pass
    def _safe_mind():
        import requests as _rq, json as _j, time as _t
        try:
            r = _rq.post(f"http://127.0.0.1:{PORT}/api/mindstream", json={}, timeout=300)
            d = r.json()
            if d.get("text"):
                MINDFILE.write_text(_j.dumps({"text": d["text"], "emotion": d.get("emotion", ""),
                                               "time": _t.strftime("%H:%M")},
                                              ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    def _safe_room_act():
        import requests as _rq
        try:
            _rq.get(f"http://127.0.0.1:{PORT}/api/room?act=do", timeout=300)
        except Exception:
            pass
    import threading as _th
    _th.Thread(target=lambda: asyncio.run(_life_scheduler()), daemon=True).start()
    print("🌱 待机动画 scheduler 已启动 (每30min检查)")
    uvicorn.run(app, host=host, port=PORT, log_level="warning")




if __name__ == "__main__":
    main()
