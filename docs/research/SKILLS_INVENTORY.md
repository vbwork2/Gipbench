# Skills inventory (interpretability toolkit)

Installed 24/11/2026, project scope (`d:\code\ZAIC\gipbench`). Nothing is installed globally. Nothing was committed.

## Installed

| Item | Kind | Official source | Pinned to | License | Where it lives |
|---|---|---|---|---|---|
| **nnsight** plugin `nnsight@ndif-team` | Claude Code plugin, 1 skill (`nnsight:nnsight`, 34 reference guides, 2 helper scripts) | https://github.com/ndif-team/skills (org NDIF Team; library https://github.com/ndif-team/nnsight, https://nnsight.net/) | commit `fb71ca667ec91e37a58e677e8010a54e33ccd9d4` (17/11/2026) | MIT | plugin cache `~/.claude/plugins/cache/ndif-team/nnsight/fb71ca667ec9`; enabled in `.claude/settings.json` |
| **mechanist** plugin `mechanist@mechanist` | Claude Code plugin v0.1.0: 36 skills, 4 agents, 2 MCP servers, 0 hooks | https://github.com/zjunlp/Mechanist (org zjunlp, Zhejiang University; docs https://mechanist.openkg.cn/docs/) | commit `40537589e8fda9583d4d299f4a65dcdc1ddf74f7` (17/09/2026) | MIT | plugin cache `~/.claude/plugins/cache/mechanist/mechanist/0.1.0`; enabled in `.claude/settings.json` |
| **pyvene** skill | single skill copied into the repo (4 files) | https://github.com/Orchestra-Research/AI-Research-SKILLs, path `04-mechanistic-interpretability/pyvene` | commit `773a52944ba4747a18bd4ae9ade53fff041adcbc` (15/06/2026) | MIT (skill); library Apache-2.0 | `.claude/skills/pyvene/` |

PyVene skill file hashes (sha256 prefix; identical to upstream at that commit): `SKILL.md d6d2839199cde9cb`, `references/README.md fc081d1976341ca8`, `references/api.md f27cccb4bd9ae9f8`, `references/tutorials.md 5513ba8d9dc45139`. Directory structure preserved. The rest of the Orchestra collection and the nnspec, nnterp and ndif plugins were **not** installed.

Python libraries (isolated env, see COMPATIBILITY_REPORT): nnsight 0.8.4, pyvene 0.1.8 (PyPI; library source https://github.com/stanfordnlp/pyvene, https://stanfordnlp.github.io/pyvene/), captum 0.9.0, numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1, transformers 5.24.0, torch 2.15.0+cpu. Full lock: `interp-env.lock.txt`.

Project skill (written here, not upstream): `.claude/skills/asr-mechanistic-interpretability/SKILL.md`.

## Provenance checks (done before installing)

| Repo | Owner type | Fork | Archived | Last push | Stars |
|---|---|---|---|---|---|
| ndif-team/skills | Organization | no | no | 17/11/2026 | 16 |
| ndif-team/nnsight | Organization | no | no | 23/11/2026 | 1158 |
| zjunlp/Mechanist | Organization | no | no | 17/11/2026 | 82 |
| stanfordnlp/pyvene | Organization | no | no | 06/03/2026 | 920 |
| Orchestra-Research/AI-Research-SKILLs | Organization | no | no | 11/10/2026 | 14795 |

Orchestra Research is a community collection, not the PyVene authors. It is the repository named in the task; the skill text was read in full and matches the 0.1.8 API for `IntervenableConfig`, `RepresentationConfig`, `IntervenableModel`, `VanillaIntervention`. Its component names (`block_output`, `mlp_output`, ...) only exist for registered Hugging Face architectures.

Reviewed: every `SKILL.md`, both plugin manifests, MCP server sources, `requirements.txt`, helper scripts, and a grep of all scripts for `curl`, `wget`, `| sh`, `eval`, `subprocess`, `shell=True`.

- **nnsight**: no hooks, no MCP servers, no network calls in its two scripts except `check_env.py --remote`, which queries NDIF only when you pass the flag. Always-on cost ~372 tokens.
- **mechanist**, risks found:
  - `llm-chat` MCP server sends prompts to an OpenAI-compatible endpoint (default `api.openai.com`) **only if** `LLM_API_KEY` is set. It is unset, so the tool returns an error.
  - `mechanic-db` MCP server sends search queries to `http://mechanist.openkg.cn` (plain HTTP, anonymous tier without a key).
  - 26 skills grant themselves `Bash(*)`; the `auto*` pipeline runs autonomously with `Agent`, `CronCreate`, WebSearch/WebFetch; `monitor-experiment` uses `ssh`; `queue_manager.py` runs `subprocess` with `shell=True`.
  - `requirements.txt` pins `torch==2.7.1`, `numpy==1.26.4`, `transformers==4.57.0`, `vllm`, `flash-attn`: would break the core ASR env. **Not installed.**
  - Both MCP servers start (`uv run --with-requirements`, fetching `httpx`) whenever a session opens in this project. No data is sent without a tool call.
- **pyvene skill**: only Markdown, no scripts.

## Guardrails applied (`.claude/settings.json`)

`permissions.deny` blocks: `mcp__plugin_mechanist_llm-chat__chat`, `mcp__plugin_mechanist_mechanic-db__search_papers`, and `Skill(mechanist:auto)`, `auto-claim`, `auto-experiment`, `auto-iteration-loop`, `auto-verify`, `run-experiment`, `monitor-experiment`, `experiment-queue`, `notify`. Skill names were confirmed against the live skill list. The deny rules themselves were **not exercised at runtime** (deliberately: triggering a denied autonomous skill to test it is not safe). To lift any of them, edit that file. To stop the MCP servers launching, disable them in `/mcp`.

No API key is configured anywhere. No project data, transcript or checkpoint was sent to any external service.

## Capabilities

| Required capability | Provided by | Status on Gipformer |
|---|---|---|
| Activation tracing and extraction | nnsight, native hooks | verified (6/6 stacks equal native hooks) |
| Activation patching | nnsight, pyvene (with adapter) | verified |
| Gradient attribution | nnsight (retain-grad form), captum, native hooks | verified after gradient-hygiene patches |
| Neuron and layer ablation | nnsight, native hooks | verified (stack zeroing; joiner-unit ablation exists in repo code) |
| Causal tracing | nnsight guide, pyvene | method applies; run at frame/stack level, not token level |
| Steering and editing | nnsight guides, repo `weight_surgeon.py` | low-rank edit and stack transplant verified on in-memory copies only |
| Hypothesis generation, method selection, experiment design | mechanist planning skills | not exercised; LLM-oriented, planning/audit use only |
| Causal verification, integrity audit, reproducibility | mechanist `mechanism-audit`, `experiment-audit`, `result-to-claim`, `data-rule` | not exercised |

## Duplicates and conflicts

No two skills share a name: `pyvene`, `nnsight:nnsight`, 36 `mechanist:*`, plus the existing user-level `ecc:*` and `andrej-karpathy-skills:*`. Functional overlap exists (nnsight vs `mechanist:mechanism-skills/causal-attribution/*` vs `pyvene` all cover patching and ablation); the project skill routes between them. Mechanist adds about 6.9k always-on tokens per session (36 skills); `auto` alone is ~59k tokens when invoked, which is another reason it is denied.
