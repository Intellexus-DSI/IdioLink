# Agent Instructions

This repository contains custom skills (slash commands) for AI coding assistants.

## Skills Location

All skills are defined in `.claude/skills/<skill-name>/SKILL.md`. Each SKILL.md contains structured instructions for performing a specific workflow.

## Available Skills

| Skill | Purpose |
|-------|---------|
| `/onboard` | New user setup — install, verify, explain |
| `/run-experiment <model> <mode>` | Run a single model+mode experiment |
| `/run-all [--debug] [models...]` | Run full experiment matrix |
| `/fine-tune <model> [mode] [seeds]` | Contrastive fine-tuning |
| `/evaluate [model] [mode]` | View/regenerate metrics |
| `/reproduce-paper` | Full paper reproduction |
| `/debug-model <model>` | Diagnose model loading issues |

## For Non-Claude Agents

If you are Cursor, Codex, or another AI tool:
1. Read `.claude/skills/<relevant-skill>/SKILL.md` for the workflow instructions
2. Follow the steps described therein
3. Key entry points: `run_dense.py`, `run_instruction.py`, `run_bm25.py`, `run_fine_tune.py`, `run_all.py`
4. Configuration: `config.yaml`
5. Core library: `idiolink/` (models, evaluator, retriever, trainer)
