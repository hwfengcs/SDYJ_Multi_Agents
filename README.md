# SDYJ Multi Agents

[English](README_EN.md) | 中文

[![CI](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml/badge.svg)](https://github.com/hwfengcs/SDYJ_Multi_Agents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

SDYJ Multi Agents 是一个基于 LangGraph 的多智能体深度研究系统。它把一次开放式研究任务拆成“意图识别 -> 计划生成 -> 人工审核 -> 多源检索 -> 综合报告”的可追踪工作流，适合展示 AI Agent 工程中的任务规划、工具调用、人机协作、报告合成和多模型适配能力。

## 为什么值得关注

- **多 Agent 工作流**：Coordinator、Planner、Researcher、Rapporteur 分工协作。
- **Human-in-the-loop**：执行研究前先展示计划，用户可批准或反馈修改。
- **多模型适配**：支持 DeepSeek、OpenAI、Claude、Gemini，统一 LLM 抽象层。
- **多源检索**：集成 Tavily、arXiv，并预留 MCP 工具适配接口。
- **可复现工程**：提供 Python 包配置、CLI 入口、单元测试、CI、示例输出和架构文档。
- **面向 Agent 岗位的信号**：关注 trace、evaluation、source grounding、tool reliability 等真实工程问题。

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
LLM_MODEL=deepseek-chat
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
  --model deepseek-chat \
  --max-iterations 3 \
  --output-format markdown \
  --auto-approve \
  "RAG Agent 如何做可靠性评估？"
```

不带 query 会进入交互式菜单：

```bash
python main.py
```

## 项目结构

```text
SDYJ_Agents/
  agents/       # Coordinator / Planner / Researcher / Rapporteur
  cli/          # argparse CLI and interactive menu
  llm/          # provider-agnostic LLM wrappers
  prompts/      # Jinja prompt templates
  tools/        # Tavily, arXiv, MCP adapters
  workflow/     # LangGraph graph and state
  utils/        # config and logging
docs/           # architecture, evaluation, roadmap context
examples/       # small reproducible examples
tests/          # unit tests with fake LLM/search
```

## 输出格式

- Markdown：适合版本管理、二次编辑、论文/报告草稿。
- HTML：适合直接分享和演示。

生成文件默认写入 `outputs/`，该目录已被 git 忽略。仓库展示样例见 [examples/sample_report.md](examples/sample_report.md)。

## 测试

```bash
pytest
ruff check SDYJ_Agents tests
```

测试默认使用 fake LLM 和 fake search，不需要真实 API key。

## Roadmap

短期目标：

- 增强 report evidence schema，让每个结论绑定来源、query 和置信度。
- 增加 trace/cost/latency 指标，便于评测 Agent 运行质量。
- 完善 MCP adapter，使外部工具接入更标准。
- 补充端到端 demo 和 benchmark case。

完整路线图见 [ROADMAP.md](ROADMAP.md)。

## 适合展示的能力

这个项目可以作为 AI Agent 算法工程师实习申请材料，重点讲：

- 如何用 LangGraph 设计可控的 Agent 状态机。
- 如何在人机协作中降低错误计划的执行成本。
- 如何把搜索结果组织为可追踪 evidence，再交给报告生成器。
- 如何用 mock、单元测试和 CI 让 LLM 项目具备工程可信度。

## 贡献

欢迎提交 Issue 和 PR。开发流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

MIT. See [LICENSE](LICENSE).
