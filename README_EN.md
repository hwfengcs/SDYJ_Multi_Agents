# SDYJ Deep Research Assistant 🔬✨

[中文文档](README.md) | English

> 🎯 **Let AI do deep research for you!** Just ask a question, and the system will automatically plan, search, organize, and generate a complete research report.

## 💡 What is this?
My first project with a multi-agent system as a beginner!

Imagine you have a research team made up of multiple AI assistants:
- **Coordinator** 👔: Understands your needs and assigns tasks
- **Planner** 📋: Creates detailed research plans
- **Researcher** 🔍: Searches the web for information
- **Reporter** 📝: Organizes everything into a professional report

This project makes these AI assistants work together to help you complete deep research tasks!

## ✨ What can it do?

✅ **Beginner-friendly**: Just one command to get started
✅ **Multiple AI models**: Supports GPT, Claude, Gemini, DeepSeek, and more
✅ **Multi-source search**: Automatically searches web pages, academic papers, and more
✅ **Human review**: Research plan requires your approval before proceeding
✅ **Multiple report formats**: Supports both Markdown and HTML output formats
✅ **Auto-generated reports**: Outputs beautiful professional research reports
✅ **Real-time progress**: Watch AI complete research step by step

## 🎬 Use Cases

- 📚 **Academic Research**: "Summarize the latest developments in Transformer architecture"
- 💼 **Industry Analysis**: "Analyze AI industry trends in 2024"
- 🌍 **Learning**: "What is quantum computing? What are its applications?"
- 📊 **Competitive Analysis**: "Compare the pros and cons of mainstream LLMs"

## 🚀 Quick Start (3 Minutes)

### Step 1: Installation

```bash
# 1. Clone the project
git clone <repository-url>
cd SDYJ_deep_reasearch

# 2. Install dependencies (just one command)
pip install -r requirements.txt
```

> 💡 **Tip**: Requires Python 3.10+, check with `python --version`

### Step 2: Configure API Keys

Create a `.env` file in the project root with the following content:

```bash
# Recommended: DeepSeek (affordable and powerful)
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxxxxxx          # 👈 Fill in your key here
TAVILY_API_KEY=tvly-xxxxxxxx          # 👈 Fill in your key here
```

**How to get API keys?**
- 🔑 **DeepSeek**: Visit [platform.deepseek.com](https://platform.deepseek.com/), register and create in console
- 🔑 **Tavily**: Visit [tavily.com](https://tavily.com/), free registration available

**Also supports other AI models:**
<details>
<summary>Click to see OpenAI / Claude / Gemini configuration</summary>

```bash
# Using OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxx

# Using Claude
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx

# Using Gemini
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIzaxxxxxxxx
```
</details>

### Step 3: Start Using!

**Method 1: Interactive Menu (Recommended for beginners)**
```bash
python main.py
```
Follow the prompts - it's that simple!

**Method 2: Direct Question**
```bash
python main.py research "What is quantum computing?"
```

**Method 3: Advanced Usage**
```bash
# Customize output file and iterations
python main.py research \
  --max-iterations 3 \
  --output my_report.md \
  "Analyze AI development trends"

# Generate HTML format report
python main.py research \
  --output-format html \
  "Quantum computing applications"

# Skip human review, fully automatic
python main.py research --auto-approve "Blockchain technology applications"
```

## 📋 Common Commands

| Command | Description |
|---------|-------------|
| `python main.py` | Open interactive menu |
| `python main.py research "question"` | Start research directly |
| `python main.py config-info` | View current configuration |
| `python main.py list-models deepseek` | List available models |

## ❓ FAQ

<details>
<summary><b>Q: I have no programming experience, can I use this?</b></summary>

Absolutely! You just need to:
1. Install Python
2. Copy and paste a few commands
3. Fill in your API keys
4. Run `python main.py` and you're ready to go!
</details>

<details>
<summary><b>Q: Do API keys cost money?</b></summary>

- **Tavily**: Free tier includes 1000 searches per month, which is plenty
- **DeepSeek**: Very affordable, ~$1 lasts a long time
- **Others**: OpenAI/Claude/Gemini are pricier but offer free credits
</details>

<details>
<summary><b>Q: Where are reports saved?</b></summary>

Reports are saved in the `outputs/` folder:
- Markdown format: `research_report_YYYYMMDD_HHMMSS.md`
- HTML format: `research_report_YYYYMMDD_HHMMSS.html`

You can choose the format using `--output-format` parameter or configure it in the interactive menu.
</details>

<details>
<summary><b>Q: Can I ask questions in other languages?</b></summary>

Yes! The system supports multiple languages including Chinese, English, and more.
</details>

<details>
<summary><b>Q: Can I interrupt the research process?</b></summary>

Yes. Press `Ctrl+C` to interrupt, and simply run again when ready.
</details>

<details>
<summary><b>Q: How do I switch between AI models?</b></summary>

Method 1: Edit `LLM_PROVIDER` in the `.env` file and set the corresponding API key
Method 2: Use parameters at runtime: `--llm-provider openai --llm-model gpt-4`
</details>

<details>
<summary><b>Q: What's the difference between Markdown and HTML formats?</b></summary>

- **Markdown (.md)**: Plain text format, ideal for version control and documentation, opens with Typora, VS Code, etc.
- **HTML (.html)**: Web page format with beautiful styling, opens directly in browsers, perfect for sharing and presentations

How to choose the format:
1. Command-line parameter: `--output-format html` or `--output-format markdown`
2. Interactive menu: Select "Configure settings" → Output format
3. Configuration file: Set `"output_format": "html"` in `config.json`
</details>

## 🎯 How It Works

Simply put, the system works like a research team pipeline:

```
Your Question ➜ Coordinator analyzes ➜ Planner creates plan ➜ You review
             ➜ Researcher searches ➜ Reporter organizes ➜ Research report ✅
```

Each step has a dedicated AI "role" that collaborates to complete the task.

## 🛠️ Advanced Configuration

<details>
<summary>Click to view complete environment configuration</summary>

Create a `.env` file with the following options:

```bash
# === LLM Configuration ===
LLM_PROVIDER=deepseek          # AI provider
LLM_MODEL=deepseek-chat        # Model name (optional)
LLM_TEMPERATURE=0.7            # Creativity (0-1, higher = more random)

# === API Keys ===
DEEPSEEK_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=AIza-xxx
TAVILY_API_KEY=tvly-xxx

# === Search Configuration ===
MCP_SERVER_URL=http://...      # MCP server (optional)

# === Workflow Settings ===
MAX_ITERATIONS=5               # Maximum search rounds
AUTO_APPROVE_PLAN=false        # Auto-approve plans
OUTPUT_DIR=./outputs           # Report save location
OUTPUT_FORMAT=markdown         # Report format (markdown or html)
```
</details>

## 💻 For Developers: Python API

If you want to integrate this system into your own Python programs:

```python
from SDYJ_Agents.utils.config import load_config_from_env
from SDYJ_Agents.llm.factory import LLMFactory
from SDYJ_Agents.agents.coordinator import Coordinator
from SDYJ_Agents.agents.planner import Planner
from SDYJ_Agents.agents.researcher import Researcher
from SDYJ_Agents.agents.rapporteur import Rapporteur
from SDYJ_Agents.workflow.graph import ResearchWorkflow

# Load configuration
config = load_config_from_env()

# Create LLM
llm = LLMFactory.create_llm(
    provider=config.llm.provider,
    api_key=config.llm.api_key,
    model=config.llm.model
)

# Initialize agents
coordinator = Coordinator(llm)
planner = Planner(llm)
researcher = Researcher(llm, tavily_api_key=config.search.tavily_api_key)
rapporteur = Rapporteur(llm)

# Create and run workflow
workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur)
final_state = workflow.run("Your research question")
print(final_state['final_report'])
```

## 📚 Technical Architecture (For Developers)

<details>
<summary>Click to view technical details</summary>

**Core Tech Stack:**
- 🧠 **LangGraph**: Workflow orchestration framework
- 🔗 **LangChain**: LLM application development framework
- 🔍 **Tavily**: Web search API
- 📄 **arXiv**: Academic paper search

**Supported LLMs:**

| Provider | Models | API Key Variable |
|----------|--------|------------------|
| DeepSeek | deepseek-chat, deepseek-coder | `DEEPSEEK_API_KEY` |
| OpenAI | gpt-4, gpt-3.5-turbo | `OPENAI_API_KEY` |
| Claude | claude-3-5-sonnet, claude-3-opus | `ANTHROPIC_API_KEY` |
| Gemini | gemini-pro, gemini-ultra | `GOOGLE_API_KEY` |

**Project Structure:**
```
SDYJ_deep_reasearch/
├── main.py                 # Program entry point
├── SDYJ_Agents/
│   ├── agents/            # Four agent implementations
│   ├── llm/              # LLM abstraction layer
│   ├── tools/            # Search tools
│   ├── workflow/         # LangGraph workflow
│   ├── prompts/          # Prompt templates
│   └── utils/            # Configuration and logging
└── outputs/              # Generated reports
```
</details>

## 🤝 Contributing & Feedback

- 💬 Have issues? [Submit an Issue](../../issues)
- 🌟 Find it useful? Give us a Star!
- 🔧 Want to improve? Pull Requests are welcome!

## 📄 License

This project is open-sourced under the MIT License.

## 🙏 Acknowledgments

Thanks to these open-source projects:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow framework
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [Tavily](https://tavily.com/) - Search service
- [arXiv](https://arxiv.org/) - Academic papers

---

<div align="center">

### 🌟 Let AI do your research, say goodbye to information overload!

**Questions? Check the [FAQ](#-faq) or [Submit an Issue](../../issues)**

</div>
