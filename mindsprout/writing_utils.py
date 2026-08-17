
from mindsprout.config import BASE
# -*- coding: utf-8 -*-
"""
writing_utils.py — 写作管线公共工具 (M6: 消除 app.py 与 writing_pipeline.py 重复)
纯函数, 不加载模型。记忆注入 / 人设锚 / 强化prompt / 英文检测。
"""
import re
import random

EN_CHECK = re.compile(r"[a-zA-Z]+")          # P1: 含单字母英文词
CN_CHECK = re.compile(r"[\u4e00-\u9fff]")

def eng_ratio(t: str) -> float:
    en = len(EN_CHECK.findall(t or ""))
    cn = len(CN_CHECK.findall(t or ""))
    return en / max(cn, 1)

# 人设锚: 防成人语料泄漏 + 时间锚定 (加在日记/作文 prompt 前)
ANCHOR = ("（你是初二女生，14岁。你写的是你自己：上学、写作业、朋友、爸妈、零花钱这些事。"
          "你不上班、没有工作、没有客户、不加班。你爸是老师，你妈在一家公司做文秘——不许把她写成医生、"
          "语文老师或其他职业。不许出现英文。\n今天是平常的一天，你在写今天的事。你记忆里那些经历都是过去发生的，"
          "别把它们写成今天的事，别把'考完的试'写成'明天要考'。）\n")

# 强化 prompt: 直接正文/禁借口/不抖机灵
STRONG = ("你趴在桌上，笔都拿起来了，开始写。直接写正文！第一句就写具体的事，"
          "不许写'忘了''不想写''写不出来''算了'这类话。结尾不要总结道理，不要抖机灵，"
          "写到不想写了就停，像随手写的。")

KIND_MT = {"diary": 260, "essay": 420, "moments": 100}
KIND_MIN = {"diary": 80, "essay": 150, "moments": 20}

def mem_block(memory_bank, scene, k=6):
    """检索相关记忆 → '你的一些经历（过去的事）：…' 前缀 (去重+随机3条)"""
    try:
        r = memory_bank.query(query_text=scene, top_k=k, k_hops=1)
        mems = [e.text for e, s in r["contents"][:k] if getattr(e, "text", "")]
        seen, picked = set(), []
        random.shuffle(mems)
        for m in mems:
            key = re.sub(r"[^\u4e00-\u9fff]", "", m)[:12]
            if key and key not in seen:
                seen.add(key)
                picked.append(m)
            if len(picked) >= 3:
                break
        if picked:
            head = "你的一些经历（这些都是你过去经历的事，不是今天发生的）：\n"
            return head + "\n".join(f"- {m}" for m in picked) + "\n\n"
    except Exception:
        pass
    return ""

def build_prompt(memory_bank, scene, kind):
    """组装完整写作 prompt"""
    return mem_block(memory_bank, scene) + ANCHOR + scene + STRONG

DEFAULT_SCENES = {
    "diary": "你趴在书桌前写今天的日记。",
    "essay": "语文老师布置了作文《我的周末》，要求300字。",
    "moments": "你拍了张窗外的照片想发朋友圈。",
}
