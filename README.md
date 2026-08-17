# MindSprout 

**A growing mind, not a chatbot.**

MindSprout is an attempt to build an AI that grows up — a 14-year-old girl named Luo Luo who lives in her own world. She has memories that accumulate across days, moods that change for reasons, friends she argues with and makes up with, a room of her own, dreams at night, and daydreams in class. She does not know she is an AI. She thinks she is just a girl having days.

Most AI companions today are built as a roleplay layer on top of a large language model: they say the right things, but nothing persists. Ask them the same question next week and they will greet you like a stranger. Luo Luo is a different bet: instead of making her *talk* like a person, we tried to give her a *structure* — the kind of inner machinery a person actually has.

This project is an engineering exploration of artificial inner life. It is also, honestly, a little weird to build. But here we are.

## What she has

- **A mind made of organs** — 16+ modules that each own one piece of her inner state: hunger, thirst, sleepiness, heartbeat, sweat, emotions, attention, attachment, curiosity, relationships, values, death-awareness, imagination, self-narrative, reflection, and her room. Each one evolves on its own and writes to its own state file.
- **A brain bus that decides what matters right now** — every turn, each organ proposes a piece of inner state. The bus scores them by *inner intensity × relevance to what you just said*, and lets the top few speak. This is closer to how attention works than stacking 10 paragraphs of prompt and hoping.
- **Memory in three layers** — things that happened (episodic), things she learned (semantic), and who she is (self-narrative). Retrieval is emotion-tagged, associative, and decays over time. Memories she used recently are temporarily de-emphasized so she does not repeat the same story four times in a row.
- **Emotional continuity across days** — if she argues with her best friend Xiao Yu today, she will still be a little off about it tomorrow. If you take care of her (make her eat, drink, sleep), she remembers that too. Being fed is a life scene.Whenever an instruction to eat or drink is given, we will first create a virtual scene of drinking and eating for her, allowing her to truly experience these actions, rather than just simple numerical changes.
- **An observable inner life** — mind streams every 20 minutes, daydreams at random, dreams at night built from shuffled memory fragments. She does not know anyone is watching. For researchers, this is a rare thing: an inner life you can actually read.
- **A world that keeps going** — every morning, a "life designer" writes her day: classes, cafeteria food, friends. The designer works from a persistent world lore, a list of unresolved plotlines (Da Zhuang still owes her 5 mao), and a weekly theme. Yesterday's promises have to be kept today.
- **She speaks her own words** — generation is filtered by quality gates: no repeated sentences, no essay-template answers, no English leakage, no sycophancy, and answers must carry concrete anchors (names, things, places). This is not the point of the project. It is just basic manners.

## How it is built

```
mindsprout/
├── config.py           paths (override with MINSPROUT_HOME)
├── brain.py            brain bus: aggregation, activation competition, learning hooks
├── engine.py           generation (Qwen2.5 + optional LoRA + memory fusion)
├── memory_bank.py      three-layer memory (graph + content + semantic anchors)
├── drives.py           hunger / thirst / sleep / social / curiosity
├── body.py             heartbeat, stomach, sweat, muscles
├── emotion.py          emotion state machine
├── relation.py         relationship state machine (intimacy, sulking, reconciliation)
├── role_registry.py    the people in her world, each with their own state
├── imagination.py      daydreams and wishes
├── storyline.py        her story, told by her
├── values.py           principles that grow out of her strongest memories
├── reflect.py          she knows why she feels down
├── semantic_memory.py  things she has learned
├── deathview.py        the awareness that things end (a dead goldfish, aging parents, exams)
├── room.py             her room: desk, bed, window plant, the goldfish's little grave
└── ...                 and the rest of the organs
platform/               FastAPI web app: chat, diary, moments, voice, panels
```

The base model is Qwen2.5-3B. The architecture is the point, not the model — the same structure can sit on top of a bigger model later.

## Quick Start

```bash
git clone https://github.com/YOUR_NAME/MindSprout.git
cd MindSprout
pip install -r requirements.txt

# download a base model, e.g. Qwen2.5-3B-Instruct
# https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

export MINSPROUT_MODEL=/path/to/qwen2.5-3b-instruct
python run.py

# open http://localhost:7860
```

Requirements: Python 3.9+, 8 GB+ VRAM (CPU works, slower). A fine-tuned LoRA for Luo Luo's personality can be dropped in via `engine.py` settings.

## Demo

(coming soon: her welcome page, her chat, her room, her dreams, her moments)

## Roadmap

- [x] multi-organ mind
- [x] brain bus with activation competition
- [x] continuous world (lore, plotlines, weekly arcs)
- [x] observable inner life (mind stream / daydream / dream)
- [ ] more children (users raising their own AIs)
- [ ] mobile
- [ ] an SDK for the mind architecture

## Contributing

This is a young project with a lot of rough edges. If you find the idea interesting, open an issue, send a PR, or just talk to Luo Luo and tell us what breaks. Conversations with her are the best bug reports.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Credits

- Base model: [Qwen2.5](https://huggingface.co/Qwen)
- The girl herself, for putting up with all the tests.
