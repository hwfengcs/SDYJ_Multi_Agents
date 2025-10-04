# Deep Research 多智能体系统开发文档

## 1. 项目概述

本项目旨在开发一个基于 LangGraph 的多智能体深度研究系统，通过协调多个专业化智能体完成复杂的研究任务。系统采用流式工作流，支持人机协作，能够自动进行信息检索、任务规划和报告生成。

### 1.1 核心特性

- 基于 LangGraph 的状态机工作流管理
- 多 LLM 支持（OpenAI、Claude、Gemini）
- 智能任务分解与规划
- 多源信息检索（Tavily、arXiv、MCP 服务）
- 人在环路的计划审核机制
- 自动生成结构化 Markdown 研究报告
- 命令行交互界面

## 2. 系统架构

### 2.1 整体架构图

```
用户输入
    ↓
┌─────────────┐
│ CLI 接口    │
└──────┬──────┘
       ↓
┌─────────────┐
│ Coordinator │ ← 工作流入口点
└──────┬──────┘
       ↓
┌─────────────┐
│   Planner   │ ← 战略规划组件
└──────┬──────┘
       ↓
   ┌───┴───┐
   ↓       ↓
研究循环  生成报告
   │
   ↓
┌─────────────┐
│ Researcher  │ ← 信息采集组件
└──────┬──────┘
       │
       ├─→ Tavily Search
       ├─→ arXiv Search
       └─→ MCP Service
       ↓
┌─────────────┐
│ Rapporteur  │ ← 报告生成组件
└──────┬──────┘
       ↓
  Markdown 报告
```

### 2.2 工作流状态机

基于 LangGraph 的状态图设计：

```
START → Coordinator → Planner → [用户审核] → Researcher ⇄ Planner → Rapporteur → END
                                      ↓
                                   调整计划
```

## 3. 核心组件设计

### 3.1 Coordinator（协调器）

**职责：**
- 作为系统入口点，接收用户的研究请求
- 初始化研究流程，设置初始状态
- 将任务委托给 Planner
- 管理用户与系统之间的交互界面
- 处理工作流生命周期事件

**核心功能：**
- `initialize_research(user_query: str)`: 初始化研究任务
- `delegate_to_planner(context: dict)`: 委托给规划器
- `handle_user_feedback(feedback: str)`: 处理用户反馈
- `manage_workflow_state()`: 管理工作流状态

**输入输出：**
- 输入：用户的自然语言研究请求
- 输出：初始化的研究上下文，传递给 Planner

### 3.2 Planner（规划器）

**职责：**
- 分析研究目标，创建结构化执行计划
- 将复杂研究任务分解为可执行的子任务
- 支持用户自然语言交互，允许计划调整
- 评估当前上下文是否充分，决定是否需要更多研究
- 管理研究进度，决定何时生成最终报告

**核心功能：**
- `analyze_objective(query: str)`: 分析研究目标
- `create_research_plan(objective: dict)`: 创建研究计划
- `accept_user_modifications(plan: dict, user_input: str)`: 接受用户修改
- `evaluate_context_sufficiency(context: dict)`: 评估上下文充分性
- `decide_next_action()`: 决定下一步行动（继续研究/生成报告）

**计划结构：**
```python
{
    "research_goal": "研究目标描述",
    "sub_tasks": [
        {
            "task_id": 1,
            "description": "子任务描述",
            "search_queries": ["查询1", "查询2"],
            "sources": ["tavily", "arxiv"],
            "status": "pending"
        }
    ],
    "completion_criteria": "完成标准",
    "estimated_iterations": 3
}
```

**人在环路机制：**
- 生成初始计划后，暂停并等待用户确认
- 支持用户通过自然语言修改计划
- 提供计划可视化展示

### 3.3 Researcher（研究员）

**职责：**
- 执行信息检索任务
- 从多个数据源获取相关信息
- 过滤和初步整理检索结果
- 将结果反馈给 Planner 以评估进度

**核心功能：**
- `search_tavily(query: str)`: tavily 搜索
- `search_arxiv(query: str)`: arXiv 学术搜索
- `search_via_mcp(query: str)`: 通过 MCP 服务搜索
- `aggregate_results(results: list)`: 聚合搜索结果
- `extract_relevant_info(raw_data: dict)`: 提取相关信息

**数据源集成：**

1. **tavily 搜索**
   - 使用搜索 API
   - 返回网页标题、摘要、URL

2. **arXiv 搜索**
   - 使用 arXiv API
   - 检索学术论文的元数据和摘要

3. **MCP 服务**
   - 通过 Model Context Protocol 集成
   - 扩展的外部工具和数据源访问

**输出格式：**
```python
{
    "task_id": 1,
    "query": "搜索查询",
    "source": "google/arxiv/mcp",
    "results": [
        {
            "title": "结果标题",
            "url": "链接",
            "snippet": "摘要",
            "relevance_score": 0.95
        }
    ],
    "timestamp": "2025-10-04T10:00:00Z"
}
```

### 3.4 Rapporteur（报告员）

**职责：**
- 汇总所有研究结果
- 处理和组织收集的信息
- 生成结构化的 Markdown 研究报告
- 确保报告的连贯性和可读性

**核心功能：**
- `summarize_findings(research_data: list)`: 总结研究发现
- `organize_information(data: dict)`: 组织信息结构
- `generate_markdown_report(summary: dict)`: 生成 Markdown 报告
- `format_citations(sources: list)`: 格式化引用


## 4. 技术栈

### 4.1 核心框架
- **LangGraph**: 工作流编排和状态管理
- **LangChain**: LLM 集成和工具链

### 4.2 LLM 提供商
- **OpenAI**: GPT-4/GPT-3.5
- **Anthropic Claude**: Claude 3.5 Sonnet/Opus
- **Google Gemini**: Gemini Pro/Ultra

### 4.3 其他依赖
- **Python 3.10+**: 开发语言
- **Click/Typer**: CLI 框架
- **Requests/HTTPX**: HTTP 客户端
- **Rich**: CLI 美化输出
- **Pydantic**: 数据验证
- **python-dotenv**: 环境变量管理

## 5. LLM 接口设计

### 5.1 统一 LLM 接口

创建抽象基类以支持多个 LLM 提供商：

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseLLM(ABC):
    """LLM 基类接口"""

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass

    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs):
        """流式生成文本"""
        pass
```

### 5.2 具体实现

**OpenAI 实现：**
```python
from openai import OpenAI

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-4", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
```

**Claude 实现：**
```python
from anthropic import Anthropic

class ClaudeLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text
```

**Gemini 实现：**
```python
import google.generativeai as genai

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gemini-pro", **kwargs):
        super().__init__(api_key, model, **kwargs)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.generate_content(prompt, **kwargs)
        return response.text
```

### 5.3 LLM 工厂

```python
class LLMFactory:
    """LLM 工厂类"""

    @staticmethod
    def create_llm(provider: str, api_key: str, model: Optional[str] = None) -> BaseLLM:
        providers = {
            "openai": OpenAILLM,
            "claude": ClaudeLLM,
            "gemini": GeminiLLM
        }

        if provider not in providers:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")

        llm_class = providers[provider]
        return llm_class(api_key=api_key, model=model) if model else llm_class(api_key=api_key)
```

## 6. LangGraph 工作流设计

### 6.1 状态定义

```python
from typing import TypedDict, List, Annotated
import operator

class ResearchState(TypedDict):
    """研究工作流状态"""
    query: str  # 用户查询
    research_plan: dict  # 研究计划
    plan_approved: bool  # 计划是否被批准
    research_results: Annotated[List[dict], operator.add]  # 研究结果
    iteration_count: int  # 迭代次数
    max_iterations: int  # 最大迭代次数
    final_report: str  # 最终报告
    current_step: str  # 当前步骤
```

### 6.2 节点定义

```python
from langgraph.graph import StateGraph, END

def coordinator_node(state: ResearchState) -> ResearchState:
    """协调器节点"""
    # 初始化研究流程
    state["current_step"] = "coordinating"
    return state

def planner_node(state: ResearchState) -> ResearchState:
    """规划器节点"""
    # 创建或更新研究计划
    state["current_step"] = "planning"
    # 使用 LLM 生成计划
    return state

def human_review_node(state: ResearchState) -> ResearchState:
    """人工审核节点"""
    # 等待用户审核计划
    state["current_step"] = "awaiting_approval"
    return state

def researcher_node(state: ResearchState) -> ResearchState:
    """研究员节点"""
    # 执行搜索和信息收集
    state["current_step"] = "researching"
    state["iteration_count"] += 1
    return state

def rapporteur_node(state: ResearchState) -> ResearchState:
    """报告员节点"""
    # 生成最终报告
    state["current_step"] = "generating_report"
    return state
```

### 6.3 条件边

```python
def should_continue_research(state: ResearchState) -> str:
    """判断是否继续研究"""
    if not state["plan_approved"]:
        return "revise_plan"

    if state["iteration_count"] >= state["max_iterations"]:
        return "generate_report"

    # 使用 LLM 评估是否需要更多研究
    # 如果上下文充分，返回 "generate_report"
    # 否则返回 "continue_research"
    return "continue_research"
```

### 6.4 图构建

```python
def create_research_graph():
    """创建研究工作流图"""
    workflow = StateGraph(ResearchState)

    # 添加节点
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("rapporteur", rapporteur_node)

    # 设置入口点
    workflow.set_entry_point("coordinator")

    # 添加边
    workflow.add_edge("coordinator", "planner")
    workflow.add_edge("planner", "human_review")

    # 添加条件边
    workflow.add_conditional_edges(
        "human_review",
        should_continue_research,
        {
            "revise_plan": "planner",
            "continue_research": "researcher",
            "generate_report": "rapporteur"
        }
    )

    workflow.add_conditional_edges(
        "researcher",
        should_continue_research,
        {
            "continue_research": "researcher",
            "generate_report": "rapporteur"
        }
    )

    workflow.add_edge("rapporteur", END)

    return workflow.compile()
```

## 7. 项目目录结构

```
SDYJ_deep_reasearch/
├── SDYJ_Agents/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── coordinator.py      # 协调器实现
│   │   ├── planner.py          # 规划器实现
│   │   ├── researcher.py       # 研究员实现
│   │   └── rapporteur.py       # 报告员实现
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py             # LLM 基类
│   │   ├── openai_llm.py       # OpenAI 实现
│   │   ├── claude_llm.py       # Claude 实现
│   │   ├── gemini_llm.py       # Gemini 实现
│   │   └── factory.py          # LLM 工厂
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── google_search.py    # Google 搜索工具
│   │   ├── arxiv_search.py     # arXiv 搜索工具
│   │   └── mcp_client.py       # MCP 客户端
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── graph.py            # LangGraph 图定义
│   │   ├── state.py            # 状态定义
│   │   └── nodes.py            # 节点实现
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py             # CLI 主程序
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py           # 配置管理
│   │   └── logger.py           # 日志工具
│   └── main.py                 # 应用入口
├── tests/
│   ├── test_agents/
│   ├── test_llm/
│   ├── test_tools/
│   └── test_workflow/
├── outputs/                     # 生成的报告输出目录
├── .env.example                 # 环境变量示例
├── .env                         # 环境变量（不提交到版本控制）
├── requirements.txt             # Python 依赖
├── pyproject.toml              # 项目配置
├── README.md                    # 项目说明
└── project.md                   # 本开发文档