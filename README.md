# SDYJ Multi Agents

**一个自验证 + 可回放 + 可基准测试的多智能体研究框架。**

[English](README_EN.md) | 中文

[![CI](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml/badge.svg)](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sdyj-multi-agents?color=blue)](https://pypi.org/project/sdyj-multi-agents/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/hwfengcs/SDYJ_Multi_Agents?style=social)](https://github.com/hwfengcs/SDYJ_Multi_Agents)

> SDYJ 把开放式研究请求转换成一个可控的 LangGraph 工作流：意图识别 → 计划生成 → 人工审核 → 多源检索 → 证据驱动的报告合成。每次运行都写入 Trace v2 事件时间线、可 deterministic replay、记录每次 LLM 调用的 token 与美元成本，并且通过 benchmark 阈值 gate 把质量回归挡在 CI 之外。目标是 **agent 工程化运营**，不是又一个 LangGraph hello-world demo。

## 为什么是 SDYJ

大多数开源 agent 框架只能跑通"happy path"。一旦工具返回为空、LLM 编造了引用、计划跑偏 —— 没办法追查、没办法回放、没办法做回归测试。SDYJ 的设计基础假设是：**agent 的运营本质上是评测问题，不是 prompt 问题**：

- 每个节点、LLM 调用、工具调用、路由决策、报告指标都成为结构化 trace 事件。
- 每次运行都可以从录制的 I/O 重放 —— 不需要真实 API 调用就能复现失败。
- 每个版本都用 hard scenario benchmark 阈值卡住 —— 质量回归会让 CI 红，无法上线。
- 每个检索结果被规范化为去重后的 `E1/E2/...` 证据项，与报告里的论点一一对应。

## 与同类项目对比

| 能力                              | SDYJ Multi Agents | GPT Researcher | AutoGen | LangGraph 官方示例 |
|-----------------------------------|:-:|:-:|:-:|:-:|
| LangGraph 状态机                  | ✅ | ❌ | ⚠️ | ✅ |
| 人在环路计划审核                  | ✅ | ❌ | ⚠️ | ⚠️ |
| 证据驱动的 `E1/E2/...` ID         | ✅ | ⚠️ | ❌ | ❌ |
| Trace v2 事件时间线               | ✅ | ❌ | ⚠️ | ❌ |
| 从 trace 做 deterministic replay  | ✅ | ❌ | ❌ | ❌ |
| 每次调用的 token + USD cost       | ✅ | ⚠️ | ⚠️ | ❌ |
| benchmark 阈值 gate (`--fail-under`) | ✅ | ❌ | ❌ | ❌ |
| 4 个 LLM provider 统一抽象层      | ✅ | ✅ | ✅ | ⚠️ |
| 自验证修订 loop                   | ✅ v0.6 alpha | ❌ | ❌ | ❌ |
| 公开 benchmark 跑分（GAIA 等）    | 🚧 v0.6 | ⚠️ | ⚠️ | ❌ |

✅ 一等公民 · ⚠️ 部分支持 / 需自己写 · ❌ 不提供 · 🚧 进行中

## 快速开始（60 秒）

```bash
conda env create -f environment.yml
conda activate sdyj
cp .env.example .env                    # Windows PowerShell: copy .env.example .env
sdyj research "RAG Agent 如何做可靠性评估？"
```

在 `.env` 中填入 `DEEPSEEK_API_KEY` 与 `TAVILY_API_KEY` 后即可运行。完整环境说明见 [docs/conda-setup.md](docs/conda-setup.md)。

不带 query 进入交互式菜单：

```bash
sdyj
```

## 架构概览

```text
用户查询
    │
    ▼
Coordinator ─ 意图识别 / 状态初始化
    │
    ▼
Planner ─ 生成结构化研究计划 ─────────────┐
    │                                     │
    ▼                                     │ 修订
Human Review ─ 批准或反馈修改 ────────────┘
    │ 批准
    ▼
Researcher ─ Tavily / arXiv / MCP 检索（迭代）
    │
    ▼
Rapporteur ─ Markdown / HTML / JSON 报告
    │
    ▼
Verifier ─ critique + revise loop
    │
    ▼
Trace v2 bundle → outputs/runs/<run-id>/
```

完整设计见 [docs/architecture.md](docs/architecture.md)。

## v0.6 新进展（进行中）

- **每次 LLM 调用的 token 与美元成本跟踪** —— 见 [`SDYJ_Agents/utils/cost.py`](SDYJ_Agents/utils/cost.py)。CLI `inspect-run` 会展示每次调用的成本表，trace.metrics 累计总额。
- **provider 无关的 usage 捕获**：OpenAI、Claude、DeepSeek、Gemini 都会暴露 `last_usage`，无论 provider 都能算 cost。
- **Verifier loop** —— 由独立 critic agent 检查报告与证据是否一致，不达标会触发有上限的 Rapporteur 修订。
- **Reflexive Researcher** —— 当一批查询返回为空、失败或相关性低时，agent 会重写查询并重试一次。
- **中途计划修订** —— 完成足够子任务后，Planner 会基于已收集证据调整剩余计划。
- **并行工具执行** —— 单个 task 内的 `(query, source)` 检索可以按并发上限同时运行。
- **结构化输出链路** —— Planner、Rapporteur 信息组织、Researcher 反思、Verifier 优先使用 provider 原生 JSON mode。
- **Streamlit Web UI MVP** —— 本地运行：`streamlit run streamlit_app.py` 或 `streamlit run SDYJ_Agents/web/app.py`。
- **PyPI 发布流程** 用 Trusted Publishers —— 见 [docs/release-process.md](docs/release-process.md)。
- **公开 benchmark 跑分** —— GAIA Level 1 子集与 AssistantBench，包括 v0.5-vs-v0.6 ablation。*即将到来。*
- **Hugging Face Spaces 在线 Demo 部署**。*即将到来。*
- **真正的 MCP 集成** —— 支持官方 `mcp` Python SDK 的 stdio / streamable HTTP transport，同时保留旧 HTTP shim 作为 fallback。

完整 v0.6 计划见 [`docs/release-notes/v0.6.md`](docs/release-notes/v0.6.md) 与 [ROADMAP.md](ROADMAP.md)。

## Trace、回放、检查

每次运行会写入 `outputs/runs/<run-id>/` 下的 bundle，并保留兼容路径 `outputs/traces/<run-id>.json`：

```bash
sdyj inspect-run                       # 最新一次运行的摘要 + 工具调用 + LLM 成本表
sdyj inspect-run <run-id> --timeline   # 完整事件时间线
sdyj runs list
sdyj replay <run-id>                   # 用录制的 LLM/tool I/O 重放，不调用真实 API
sdyj diff-runs <run-a> <run-b>
```

也可以在浏览器打开 `SDYJ_Agents/web/trace_viewer.html`，拖入
`trace.json` 或 `events.jsonl` 做客户端过滤和事件详情检查。

详见 [docs/trace-replay.md](docs/trace-replay.md)。

## Benchmark

离线 benchmark 使用确定性 hard scenarios + canned evidence，不需要真实 API key，能直接接入 CI：

```bash
sdyj list-scenarios
sdyj benchmark run --max-scenarios 1 --max-iterations 2
sdyj benchmark run --fail-under 0.75            # CI 回归 gate
sdyj benchmark run --determinism-repeats 2      # 离线 determinism 检查
```

真实 DeepSeek 评测把模型推理与确定性检索分开：

```bash
sdyj benchmark run \
  --live \
  --provider deepseek \
  --model deepseek-v4-flash \
  --scenario agent_reliability_hard \
  --max-iterations 2
```

加 `--live-search` 同时启用真实检索。详见 [docs/benchmark.md](docs/benchmark.md)。

## 项目结构

```text
SDYJ_Agents/
  agents/       # Coordinator / Planner / Researcher / Rapporteur / Verifier
  cli/          # argparse CLI 与交互菜单
  llm/          # provider 无关的 LLM 抽象层（OpenAI / Claude / Gemini / DeepSeek）
  prompts/      # Jinja 提示词模板
  tools/        # Tavily、arXiv、MCP 适配器
  workflow/     # LangGraph 图 / 状态 / 节点
  utils/        # config、logging、evidence、tracing、cost
  evaluation/   # benchmark 场景、指标、runner
docs/           # 架构、trace/replay、benchmark、发布流程
examples/       # 可复现的小示例
tests/          # 用 fake LLM/search 的单元测试
```

## 配置

```bash
copy .env.example .env
```

至少配置一个 LLM API key。推荐 DeepSeek（最便宜）：

```bash
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

其他 provider 使用各自官方变量名 —— `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GOOGLE_API_KEY`。旧的 `CLAUDE_API_KEY` 和 `GEMINI_API_KEY` 仍兼容。

MCP 是可选配置。stdio、streamable HTTP 与旧 HTTP fallback 的配置方式见
[docs/mcp.md](docs/mcp.md)。

## 输出格式

- Markdown：适合版本管理、二次编辑、报告草稿。
- HTML：适合直接分享与演示。
- JSON：适合下游自动化、回归评测、系统集成。

样例：[examples/sample_report.md](examples/sample_report.md) · [examples/sample_trace.json](examples/sample_trace.json) · [examples/eval_summary.json](examples/eval_summary.json)。

## 开发

```bash
conda env update -n sdyj -f environment.yml --prune
conda activate sdyj
pytest
ruff check SDYJ_Agents tests
```

Conda 是本项目默认开发环境。单元测试使用 fake LLM 与 fake search，不需要真实 API key。

## Roadmap

| 里程碑 | 状态 |
|-------|------|
| v0.1 — 工程基线（CI、tests、license、docs） | ✅ |
| v0.2 — 证据驱动报告 + 去重 source ID | ✅ |
| v0.3 — Agent 可观测性（Trace v2、latency、错误率） | ✅ |
| v0.4 — 评测套件（hard scenarios + 报告质量指标） | ✅ |
| v0.5 — Trace v2、deterministic replay、benchmark gate、JSON 输出 | ✅ |
| **v0.6 — self-verifying loop、cost tracking、公开 benchmark、Web UI、MCP** | 🚧 |
| v0.7 — partial replay、OpenTelemetry 导出、插件式检索注册 | ⏳ |

完整路线图见 [ROADMAP.md](ROADMAP.md)。

## 贡献

欢迎提交 Issue 和 PR，开发流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

MIT，见 [LICENSE](LICENSE)。
