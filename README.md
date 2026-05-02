# SDYJ Multi Agents

[English](README_EN.md) | 中文

[![CI](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml/badge.svg)](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

SDYJ Multi Agents 是一个基于 LangGraph 的多智能体研究框架。它把开放式研究任务拆成“意图识别 -> 计划生成 -> 人工审核 -> 多源检索 -> 综合报告”的可控工作流，并重点提供 Trace v2、deterministic replay、证据追踪和可作为 CI gate 的 benchmark。

## 为什么值得关注

- **多 Agent 工作流**：Coordinator、Planner、Researcher、Rapporteur 分工协作。
- **Human-in-the-loop**：执行研究前先展示计划，用户可批准或反馈修改。
- **多模型适配**：支持 DeepSeek、OpenAI、Claude、Gemini，统一 LLM 抽象层。
- **多源检索**：集成 Tavily、arXiv，并预留 MCP 工具适配接口。
- **证据驱动报告**：检索结果会规范化为 `E1/E2/...` 证据项，自动去重并保留 URL、domain、query、发布时间和相关性分数。
- **Trace v2 可观测性**：每次运行都会生成事件时间线，记录节点、LLM 调用、工具调用、routing decision、latency、错误、报告指标和 replay cache。
- **Deterministic Replay**：可用已记录的 LLM/tool I/O 重放一次历史运行，复现失败路径，不调用真实 API。
- **实用 Benchmark**：内置困难 Agent 场景，支持阈值 gate、summary compare、trace completeness 和离线确定性检查。
- **可复现工程**：提供 Python 包配置、CLI 入口、单元测试、CI、示例输出和架构文档。

## 架构概览

```text
User Query
    |
    v
Coordinator -- classify intent / initialize state
    |
    v
Planner -- build structured research plan
    |
    v
Human Review -- approve or request changes
    |
    v
Researcher -- Tavily / arXiv / MCP retrieval
    |
    v
Rapporteur -- synthesize Markdown or HTML report
```

更完整的设计说明见 [docs/architecture.md](docs/architecture.md)。

## 快速开始

### 1. 安装

```bash
git clone https://github.com/hwfengcs/SDYJ_Multi_Agents.git
cd SDYJ_Multi_Agents
python -m pip install -e ".[dev]"
```

也可以使用传统方式：

```bash
python -m pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
copy .env.example .env
```

在 `.env` 中填入至少一个 LLM API Key。推荐先用 DeepSeek：

```bash
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

Claude 和 Gemini 使用官方变量名：

```bash
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

项目仍兼容旧变量名 `CLAUDE_API_KEY` 和 `GEMINI_API_KEY`。

### 3. 运行

```bash
python main.py config-info
python main.py list-models deepseek
python main.py research "总结一下 AI Agent 评测方法的最新趋势"
```

安装为包后也可以使用：

```bash
sdyj research "对比 LangGraph、AutoGen 和 CrewAI 的设计取舍"
```

常用参数：

```bash
python main.py research \
  --provider deepseek \
  --model deepseek-v4-flash \
  --max-iterations 3 \
  --output-format markdown \
  --auto-approve \
  "RAG Agent 如何做可靠性评估？"
```

不带 query 会进入交互式菜单：

```bash
python main.py
```

### 4. Trace / Replay

每次研究和评测都会写入 run bundle：

```text
outputs/runs/<run-id>/
  trace.json
  events.jsonl
  state.final.json
  report.md|html|json
```

同时保留兼容路径 `outputs/traces/<run-id>.json`。

```bash
python main.py inspect-run
python main.py inspect-run <run-id> --timeline
python main.py runs list
python main.py replay <run-id>
python main.py diff-runs <run-a> <run-b>
```

详见 [docs/trace-replay.md](docs/trace-replay.md)。

### 5. Benchmark

离线 benchmark 不需要真实 API key，使用固定困难场景和 canned evidence，适合 CI 和回归测试：

```bash
python main.py list-scenarios
python main.py benchmark run --max-scenarios 1 --max-iterations 2
python main.py benchmark run --fail-under 0.75
python main.py benchmark run --determinism-repeats 2
```

真实 DeepSeek 评测会调用 `.env` 中的 `DEEPSEEK_API_KEY`，默认仍使用 canned evidence，方便把变量集中在模型推理质量上：

```bash
python main.py benchmark run \
  --live \
  --provider deepseek \
  --model deepseek-v4-flash \
  --scenario agent_reliability_hard \
  --max-iterations 2
```

如果需要同时评估真实搜索链路，可以加 `--live-search`。

旧命令 `python main.py eval ...` 仍然可用。详见 [docs/benchmark.md](docs/benchmark.md)。

## 项目结构

```text
SDYJ_Agents/
  agents/       # Coordinator / Planner / Researcher / Rapporteur
  cli/          # argparse CLI and interactive menu
  llm/          # provider-agnostic LLM wrappers
  prompts/      # Jinja prompt templates
  tools/        # Tavily, arXiv, MCP adapters
  workflow/     # LangGraph graph and state
  utils/        # config, logging, evidence, tracing
  evaluation/   # benchmark scenarios, metrics, runner
docs/           # architecture, trace/replay, benchmark, roadmap
examples/       # small reproducible examples
tests/          # unit tests with fake LLM/search
```

## 输出格式

- Markdown：适合版本管理、二次编辑、论文/报告草稿。
- HTML：适合直接分享和演示。
- JSON：适合下游自动化、回归评测和系统集成。

生成文件默认写入 `outputs/`，该目录已被 git 忽略：

- `outputs/research_report_*.md|html|json`：研究报告。
- `outputs/runs/<run-id>/`：Trace v2 run bundle。
- `outputs/traces/*.json`：兼容旧路径的运行轨迹。
- `outputs/eval_reports/`：评测报告和汇总。

仓库展示样例见 [examples/sample_report.md](examples/sample_report.md)、[examples/sample_trace.json](examples/sample_trace.json) 和 [examples/eval_summary.json](examples/eval_summary.json)。

## 测试

```bash
pytest
ruff check SDYJ_Agents tests
```

测试默认使用 fake LLM 和 fake search，不需要真实 API key。

## Roadmap

v0.5 已完成：

- Trace v2 事件流和 run bundle。✅
- Deterministic replay。✅
- Benchmark gate、阈值、summary compare 和 trace completeness。✅
- JSON 输出格式。✅

下一步重点是 partial replay、外部 benchmark suite、工具注册机制、OpenTelemetry export 和 Web UI。

完整路线图见 [ROADMAP.md](ROADMAP.md)。

## 贡献

欢迎提交 Issue 和 PR。开发流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

MIT. See [LICENSE](LICENSE).
