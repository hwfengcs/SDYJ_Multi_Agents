# SDYJ 深度研究助手 🔬✨

中文文档 | [English](README_EN.md)

> 🎯 **让 AI 帮你做深度研究！** 只需要提一个问题，系统会自动规划、搜索、整理，最后生成一份完整的研究报告。

## 💡 这是什么？
小白初上手多agent系统的第一个项目！

想象一下，你有一个由多个 AI 助手组成的研究团队：
- **协调器** 👔：理解你的需求，分配任务
- **规划师** 📋：制定详细的研究计划
- **研究员** 🔍：在网上搜索各种资料
- **报告员** 📝：把所有信息整理成专业报告

这个项目就是让这些 AI 助手协同工作，帮你完成深度研究任务！

## ✨ 能做什么？

✅ **零基础也能用**：只需要一行命令就能开始
✅ **多种 AI 模型**：支持 GPT、Claude、Gemini、DeepSeek 等
✅ **全网搜索**：自动搜索网页、学术论文等多个来源
✅ **人工审核**：研究计划会先让你确认,不满意可以修改
✅ **多种报告格式**：支持 Markdown 和 HTML 两种输出格式
✅ **自动生成报告**：输出漂亮的专业研究报告
✅ **实时进度**：看着 AI 一步步完成研究

## 🎬 使用场景

- 📚 **学术研究**："总结一下 Transformer 架构的最新进展"
- 💼 **行业调研**："分析 2024 年 AI 行业的发展趋势"
- 🌍 **知识学习**："量子计算是什么？有哪些应用场景？"
- 📊 **竞品分析**："对比主流的大语言模型的优缺点"

## 📋 系统架构

```
用户查询
    ↓
┌─────────────┐
│ Coordinator │ ← 入口点
│  协调器     │
└──────┬──────┘
       ↓
┌─────────────┐
│   Planner   │ ← 创建研究计划
│   规划器    │
└──────┬──────┘
       ↓
  [用户审核]
       ↓
┌─────────────┐
│ Researcher  │ ← 收集信息
│  研究员     │
└──────┬──────┘
       ↓
┌─────────────┐
│ Rapporteur  │ ← 生成报告
│  报告员     │
└──────┬──────┘
       ↓
 Markdown/HTML 报告
```

## 🚀 快速开始（3分钟上手）

### 步骤一：安装

```bash
# 1. 下载项目
git clone <repository-url>
cd SDYJ_deep_reasearch

# 2. 安装依赖（就这一行）
pip install -r requirements.txt
```

> 💡 **提示**：需要 Python 3.10+，用 `python --version` 检查版本

### 步骤二：配置 API 密钥

创建 `.env` 文件（在项目根目录），填入以下内容：

```bash
# 推荐使用 DeepSeek（便宜好用）
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxxxxxx          # 👈 在这里填你的密钥
TAVILY_API_KEY=tvly-xxxxxxxx          # 👈 在这里填你的密钥
```

**如何获取 API 密钥？**
- 🔑 **DeepSeek**：访问 [platform.deepseek.com](https://platform.deepseek.com/)，注册后在控制台创建
- 🔑 **Tavily**：访问 [tavily.com](https://tavily.com/)，免费注册即可获得

**也支持其他 AI 模型：**
<details>
<summary>点击查看 OpenAI / Claude / Gemini 配置</summary>

```bash
# 使用 OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxx

# 使用 Claude
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx

# 使用 Gemini
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIzaxxxxxxxx
```
</details>

### 步骤三：开始使用！

**方式1：交互式菜单（推荐新手）**
```bash
python main.py
```
然后按提示操作即可，非常简单！

**方式2：直接提问**
```bash
python main.py research "量子计算是什么？"
```

**方式3：高级用法**
```bash
# 自定义输出文件和迭代次数
python main.py research \
  --max-iterations 3 \
  --output 我的报告.md \
  "分析人工智能的发展趋势"

# 生成 HTML 格式报告
python main.py research \
  --output-format html \
  "量子计算的应用前景"

# 跳过人工审核，全自动运行
python main.py research --auto-approve "区块链技术应用场景"
```

## 📋 常用命令

| 命令 | 说明 |
|------|------|
| `python main.py` | 打开交互式菜单 |
| `python main.py research "问题"` | 直接开始研究 |
| `python main.py config-info` | 查看当前配置 |
| `python main.py list-models deepseek` | 查看可用模型 |

## ❓ 常见问题

<details>
<summary><b>Q: 我没有编程基础，能用吗？</b></summary>

完全可以！只需要：
1. 安装 Python
2. 复制粘贴几行命令
3. 填入 API 密钥
4. 运行 `python main.py` 就能用了
</details>

<details>
<summary><b>Q: API 密钥要花钱吗？</b></summary>

- **Tavily**：免费版每月有1000次搜索额度，够用
- **DeepSeek**：非常便宜，1块钱能用很久
- **其他模型**：OpenAI/Claude/Gemini 价格较高，但也有免费额度
</details>

<details>
<summary><b>Q: 报告会保存在哪里？</b></summary>

默认保存在 `outputs/` 文件夹：
- Markdown 格式：`research_report_日期时间.md`
- HTML 格式：`research_report_日期时间.html`

可通过 `--output-format` 参数选择格式，或在交互式菜单中配置。
</details>

<details>
<summary><b>Q: 可以用中文提问吗？</b></summary>

当然可以！支持中英文混合，AI 会自动识别并处理。
</details>

<details>
<summary><b>Q: 研究过程可以中断吗？</b></summary>

可以。按 `Ctrl+C` 中断，下次重新运行即可。
</details>

<details>
<summary><b>Q: 如何切换不同的 AI 模型？</b></summary>

方法1：修改 `.env` 文件中的 `LLM_PROVIDER` 和对应的 API 密钥
方法2：运行时使用参数 `--llm-provider openai --llm-model gpt-4`
</details>

<details>
<summary><b>Q: Markdown 和 HTML 格式有什么区别？</b></summary>

- **Markdown (.md)**：纯文本格式，适合版本控制、文档编辑，可用 Typora、VS Code 等工具打开
- **HTML (.html)**：网页格式，包含精美样式，可直接用浏览器打开，适合分享和演示

选择格式的方式：
1. 命令行参数：`--output-format html` 或 `--output-format markdown`
2. 交互式菜单：选择"配置设置" → 输出格式
3. 配置文件：在 `config.json` 中设置 `"output_format": "html"`
</details>

## 🎯 工作原理

简单来说，这个系统就像一个研究团队的流水线：

```
你的问题 ➜ 协调器分析 ➜ 规划师制定计划 ➜ 你审核确认
         ➜ 研究员搜索资料 ➜ 报告员整理成文 ➜ 生成研究报告 ✅
```

每个环节都有专门的 AI "角色"负责，互相配合完成任务。

## 🛠️ 高级配置

<details>
<summary>点击查看完整的环境变量配置</summary>

创建 `.env` 文件，可配置以下选项：

```bash
# === LLM 配置 ===
LLM_PROVIDER=deepseek          # AI 提供商
LLM_MODEL=deepseek-chat        # 模型名称（可选）
LLM_TEMPERATURE=0.7            # 创造性（0-1，越高越随机）

# === API 密钥 ===
DEEPSEEK_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=AIza-xxx
TAVILY_API_KEY=tvly-xxx

# === 搜索配置 ===
MCP_SERVER_URL=http://...      # MCP 服务器（可选）

# === 工作流配置 ===
MAX_ITERATIONS=5               # 最大搜索轮数
AUTO_APPROVE_PLAN=false        # 是否自动批准计划
OUTPUT_DIR=./outputs           # 报告保存位置
OUTPUT_FORMAT=markdown         # 报告格式（markdown 或 html）
```
</details>

## 💻 给开发者：Python API

如果你想在自己的 Python 程序中使用这个系统：

```python
from SDYJ_Agents.utils.config import load_config_from_env
from SDYJ_Agents.llm.factory import LLMFactory
from SDYJ_Agents.agents.coordinator import Coordinator
from SDYJ_Agents.agents.planner import Planner
from SDYJ_Agents.agents.researcher import Researcher
from SDYJ_Agents.agents.rapporteur import Rapporteur
from SDYJ_Agents.workflow.graph import ResearchWorkflow

# 加载配置
config = load_config_from_env()

# 创建 LLM
llm = LLMFactory.create_llm(
    provider=config.llm.provider,
    api_key=config.llm.api_key,
    model=config.llm.model
)

# 初始化智能体
coordinator = Coordinator(llm)
planner = Planner(llm)
researcher = Researcher(llm, tavily_api_key=config.search.tavily_api_key)
rapporteur = Rapporteur(llm)

# 创建并运行工作流
workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur)
final_state = workflow.run("你的研究问题")
print(final_state['final_report'])
```

## 📚 技术架构（给开发者）

<details>
<summary>点击查看技术细节</summary>

**核心技术栈：**
- 🧠 **LangGraph**：工作流编排框架
- 🔗 **LangChain**：LLM 应用开发框架
- 🔍 **Tavily**：网页搜索 API
- 📄 **arXiv**：学术论文搜索

**支持的 LLM：**

| 提供商 | 模型 | API 密钥变量 |
|--------|------|-------------|
| DeepSeek | deepseek-chat, deepseek-coder | `DEEPSEEK_API_KEY` |
| OpenAI | gpt-4, gpt-3.5-turbo | `OPENAI_API_KEY` |
| Claude | claude-3-5-sonnet, claude-3-opus | `ANTHROPIC_API_KEY` |
| Gemini | gemini-pro, gemini-ultra | `GOOGLE_API_KEY` |

**项目结构：**
```
SDYJ_deep_reasearch/
├── main.py                 # 程序入口
├── SDYJ_Agents/
│   ├── agents/            # 四个智能体实现
│   ├── llm/              # LLM 抽象层
│   ├── tools/            # 搜索工具
│   ├── workflow/         # LangGraph 工作流
│   ├── prompts/          # 提示词模板
│   └── utils/            # 配置和日志
└── outputs/              # 生成的报告
```
</details>

## 🤝 贡献与反馈

- 💬 遇到问题？[提交 Issue](../../issues)
- 🌟 觉得好用？给个 Star 支持一下
- 🔧 想改进？欢迎提交 Pull Request

## 📄 开源协议

本项目采用 MIT 协议开源，可自由使用和修改。

## 🙏 致谢

感谢以下开源项目：
- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流框架
- [LangChain](https://github.com/langchain-ai/langchain) - LLM 应用框架
- [Tavily](https://tavily.com/) - 搜索服务
- [arXiv](https://arxiv.org/) - 学术论文

---

<div align="center">

### 🌟 让 AI 帮你做研究，从此告别信息焦虑！

**有问题？看看[常见问题](#-常见问题)或者[提交 Issue](../../issues)**

</div>
