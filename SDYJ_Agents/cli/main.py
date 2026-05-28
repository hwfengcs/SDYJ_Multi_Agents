"""
CLI Main Program

This module provides the command-line interface for the SDYJ research system.
Refactored to follow the example.py structure with argparse and config persistence.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from datetime import datetime

from dotenv import load_dotenv
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..evaluation import run_evaluation
from ..evaluation.scenarios import list_scenarios
from ..utils.config import load_config_from_env
from ..utils.logger import setup_logger
from ..utils.tracing import (
    InstrumentedLLM,
    create_run_trace,
    diff_traces,
    iter_timeline_events,
    latest_trace_path,
    load_trace,
    merge_trace_state,
    save_trace,
)
from ..replay import can_deterministically_replay, run_deterministic_replay
from ..llm.factory import LLMFactory
from ..agents.coordinator import Coordinator
from ..agents.planner import Planner
from ..agents.researcher import Researcher
from ..agents.rapporteur import Rapporteur
from ..workflow.graph import ResearchWorkflow

console = Console()
error_console = Console(stderr=True)


@dataclass
class CLIConfig:
    """CLI运行时配置"""
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    max_iterations: int = 5
    auto_approve: bool = False
    output_dir: str = "./outputs"
    show_steps: bool = False
    output_format: str = "markdown"  # "markdown", "html", or "json"


# 配置文件路径
CONFIG_FILE = Path(__file__).parent.parent.parent / "config.json"

PROVIDER_DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-flash",
    "openai": "gpt-4o-mini",
    "claude": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-1.5-pro",
}

PROVIDER_API_KEY_ENVS = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "claude": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}

PROVIDER_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "claude": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
    "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
}

ACCENT = "bright_white"
SUBTLE = "white"
MUTED = "bright_black"
SUCCESS = "green"
WARNING = "yellow"
ERROR = "red"


def load_config_from_file() -> Dict[str, Any]:
    """从配置文件加载设置"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[yellow][WARN] 配置文件加载失败：{e}，使用默认设置[/yellow]")
    return {}


def save_config_to_file(config: CLIConfig) -> None:
    """保存配置到文件"""
    try:
        config_data = {
            "provider": config.provider,
            "model": config.model,
            "max_iterations": config.max_iterations,
            "auto_approve": config.auto_approve,
            "output_dir": config.output_dir,
            "show_steps": config.show_steps,
            "output_format": config.output_format,
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        status_line("保存", "配置已写入 config.json", SUCCESS)
    except Exception as e:
        status_line("错误", f"配置保存失败：{e}", ERROR)


def get_api_key_for_provider(provider: str) -> str | None:
    """根据提供商获取对应的 API 密钥"""
    for env_var in PROVIDER_API_KEY_ENVS.get(provider.lower(), ()):
        api_key = os.getenv(env_var)
        if api_key:
            return api_key
    return None


def yes_no(value: bool) -> str:
    """Format a boolean value for compact CLI displays."""
    return "是" if value else "否"


def status_line(label: str, message: str, style: str = ACCENT) -> None:
    """Print a compact, modern status line."""
    console.print(f"[bold white]{label:<6}[/] [{style}]{message}[/]")


def compact_text(value: Any, limit: int = 140) -> str:
    """Collapse whitespace and keep terminal copy readable."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    width = 0
    output = []
    for char in text:
        char_width = 2 if ord(char) > 127 else 1
        if width + char_width > limit:
            return f"{''.join(output).rstrip()}..."
        output.append(char)
        width += char_width
    if output:
        return text
    return ""


def compact_list(values: list[Any], limit: int = 3) -> str:
    """Render a short list without vertical wrapping noise."""
    items = [compact_text(value, 48) for value in values if str(value or "").strip()]
    if not items:
        return "N/A"
    visible = items[:limit]
    suffix = f"；另有 {len(items) - limit} 条" if len(items) > limit else ""
    return "；".join(visible) + suffix


def display_status(status: str | None) -> str:
    """Translate internal task status for users."""
    return {
        "pending": "待执行",
        "in_progress": "进行中",
        "completed": "已完成",
    }.get(str(status or "pending"), str(status or "待执行"))


def render_config_panel(config: CLIConfig, title: str = "当前配置") -> Panel:
    """Render runtime configuration as a compact table."""
    key_status = "已配置" if get_api_key_for_provider(config.provider) else "未配置"
    env_names = ", ".join(PROVIDER_API_KEY_ENVS.get(config.provider, ()))

    table = Table.grid(expand=True)
    table.add_column(style=SUBTLE, ratio=1)
    table.add_column(style="white", ratio=2)
    table.add_row("提供商", f"[bold]{config.provider}[/]")
    table.add_row("模型", config.model)
    table.add_row("最大迭代", str(config.max_iterations))
    table.add_row("自动批准", yes_no(config.auto_approve))
    table.add_row("输出格式", config.output_format.upper())
    table.add_row("输出目录", config.output_dir)
    table.add_row("显示步骤", yes_no(config.show_steps))
    table.add_row("API Key", f"{key_status} [{SUBTLE}]({env_names})[/]")

    return Panel(
        table,
        title=f"[bold {ACCENT}]{title}[/]",
        border_style=ACCENT,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def render_plan_panel(plan: Dict[str, Any]) -> Panel:
    """Render a research plan as readable vertical summaries."""
    lines = [
        Text("研究目标", style="bold white"),
        Text(compact_text(plan.get("research_goal", "N/A"), 180), style="white"),
        Text(""),
        Text("完成标准", style="bold white"),
        Text(compact_text(plan.get("completion_criteria", "N/A"), 180), style="white"),
        Text(""),
        Text(f"预计迭代：{plan.get('estimated_iterations', 'N/A')}", style="white"),
        Text(""),
        Text("任务摘要", style="bold white"),
    ]

    for task in plan.get("sub_tasks", []):
        task_id = task.get("task_id", "?")
        priority = task.get("priority", "N/A")
        sources = ", ".join(task.get("sources", [])) or "N/A"
        status = display_status(task.get("status"))
        lines.extend(
            [
                Text(""),
                Text(f"{task_id}. {compact_text(task.get('description', ''), 150)}", style="bold white"),
                Text(f"   优先级：{priority}    来源：{sources}    状态：{status}", style="white"),
                Text(f"   查询：{compact_list(task.get('search_queries', []), limit=2)}", style="white"),
            ]
        )

    return Panel(
        Group(*lines),
        title="[bold white]研究计划[/]",
        subtitle=f"[{MUTED}]批准后开始检索；需要调整就选择微调[/]",
        border_style="white",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def render_task_panel(task: Dict[str, Any], iteration: int, max_iterations: int) -> Panel:
    """Render the active research task."""
    return Panel(
        Group(
            Text(f"进度：{iteration}/{max_iterations}", style="bold white"),
            Text(compact_text(task.get("description", "未知任务"), 160), style="white"),
            Text(f"来源：{', '.join(task.get('sources', [])) or 'N/A'}", style="white"),
            Text(f"查询：{compact_list(task.get('search_queries', []), limit=2)}", style="white"),
        ),
        title="[bold white]正在检索[/]",
        border_style="white",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def clean_summary_text(text: str) -> str:
    """Remove markup that is noisy in a short CLI summary."""
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[E\d+\]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[#*_`>|-]+", " ", text)
    return compact_text(text, 66)


def extract_markdown_section(report: str, heading: str) -> str:
    """Extract a markdown section body by heading text."""
    pattern = rf"(?is)^##\s+{re.escape(heading)}\s*(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, report or "", flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def summarize_report_for_cli(report: str, output_format: str, limit: int = 5) -> list[str]:
    """Create a concise terminal summary from the generated report artifact."""
    candidates: list[str] = []

    if output_format == "json":
        try:
            payload = json.loads(report)
            if payload.get("summary"):
                candidates.append(payload["summary"])
            for item in payload.get("key_findings", [])[: limit + 1]:
                if isinstance(item, dict) and item.get("claim"):
                    candidates.append(item["claim"])
        except json.JSONDecodeError:
            candidates.append(report)
    elif output_format == "html":
        text = re.sub(r"</(p|li|h\d)>", "\n", report, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        candidates.extend(line for line in text.splitlines() if line.strip())
    else:
        executive = extract_markdown_section(report, "执行摘要")
        findings = extract_markdown_section(report, "核心发现")
        if executive:
            candidates.extend(line for line in executive.splitlines() if line.strip())
        if findings:
            candidates.extend(line for line in findings.splitlines() if line.strip())
        if not candidates:
            candidates.extend(line for line in (report or "").splitlines() if line.strip())

    summary = []
    seen = set()
    for item in candidates:
        cleaned = clean_summary_text(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        summary.append(cleaned)
        if len(summary) >= limit:
            break
    return summary or ["报告已生成，建议打开文件查看完整内容。"]


def render_completion_panel(
    report: str,
    output_format: str,
    report_path: Path,
    trace_path: Path | None,
) -> Panel:
    """Render the only terminal output users need after report generation."""
    lines = [
        Text("本次研究已完成。", style="bold green"),
        Text(""),
        Text("精华总结", style="bold white"),
    ]
    for index, item in enumerate(summarize_report_for_cli(report, output_format), 1):
        lines.append(Text(f"{index}. {item}", style="white"))

    lines.extend(
        [
            Text(""),
            Text("完整报告已保存，请打开下面的文件查看细节：", style="bold yellow"),
            Text(str(report_path), style="bold white"),
        ]
    )
    if trace_path:
        lines.append(Text(f"Trace：{trace_path}", style=MUTED))

    return Panel(
        Group(*lines),
        title="[bold green]研究完成[/]",
        border_style=SUCCESS,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def print_separator(char: str = "─", length: int = 70) -> None:
    """打印分隔线"""
    console.print(f"[{SUBTLE}]{char * length}[/]")


def print_header(text: str) -> None:
    """打印标题"""
    header = Group(
        Align.center(Text(text, style="bold white")),
        Align.center(Text("LangGraph Multi-Agent Research Console", style=ACCENT)),
    )
    console.print(Panel(
        header,
        border_style=ACCENT,
        box=box.ROUNDED,
        padding=(1, 4),
    ))


def print_welcome() -> None:
    """打印欢迎界面"""
    console.print("\n")
    print_header("SDYJ 深度研究系统")

    # 显示配置文件状态
    if CONFIG_FILE.exists():
        status_line("配置", f"已加载 {CONFIG_FILE.name}", SUCCESS)
    else:
        status_line("配置", "使用默认配置 max_iterations=5, auto_approve=False")
    console.print()


def print_menu() -> None:
    """打印主菜单"""
    table = Table.grid(expand=True)
    table.add_column("key", style=f"bold {ACCENT}", width=4, justify="center", no_wrap=True)
    table.add_column("action", style="white", ratio=2)
    table.add_column("hint", style=SUBTLE, ratio=3)
    table.add_row("1", "执行研究任务", "生成计划、审批、检索并输出报告")
    table.add_row("2", "查看可用模型", "按 provider 浏览模型名称")
    table.add_row("3", "配置设置", "调整 provider、模型、输出与审批")
    table.add_row("4", "查看当前配置", "检查运行参数和 API Key 状态")
    table.add_row("5", "退出程序", "结束当前 CLI 会话")
    console.print(Panel(
        table,
        title=f"[bold {ACCENT}]主菜单[/]",
        border_style=SUBTLE,
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def show_models(provider: str) -> None:
    """显示可用模型列表"""
    table = Table(
        title=f"{provider.upper()} 可用模型",
        box=box.ROUNDED,
        border_style=ACCENT,
        header_style=f"bold {ACCENT}",
        expand=True,
    )
    table.add_column("#", justify="right", width=4, style=SUBTLE)
    table.add_column("Model", style="white")
    table.add_column("默认", justify="center", width=8)

    default_model = PROVIDER_DEFAULT_MODELS.get(provider)
    for index, model in enumerate(PROVIDER_MODELS.get(provider, []), 1):
        table.add_row(str(index), model, "是" if model == default_model else "")
    console.print(table)


def print_config_info(config: CLIConfig) -> None:
    """显示当前 CLI 配置和密钥状态。"""
    console.print(render_config_panel(config))


def configure_settings(config: CLIConfig) -> None:
    """配置设置"""
    console.print(render_config_panel(config, title="配置设置"))
    status_line("提示", "直接回车保留当前值")
    console.print()

    config_changed = False

    # 修改提供商
    provider_input = input(f"LLM 提供商 (deepseek/openai/claude/gemini) [{config.provider}]: ").strip().lower()
    if provider_input and provider_input in ["deepseek", "openai", "claude", "gemini"]:
        if provider_input != config.provider:
            # 检查 API 密钥
            new_api_key = get_api_key_for_provider(provider_input)
            if not new_api_key:
                console.print(f"[red][ERR] 未找到 {provider_input.upper()}_API_KEY 环境变量[/red]")
                console.print(f"[yellow]请在 .env 文件中配置 {provider_input.upper()}_API_KEY[/yellow]")
            else:
                config.provider = provider_input
                # 自动更新默认模型
                config.model = PROVIDER_DEFAULT_MODELS.get(provider_input, config.model)
                config_changed = True
                console.print(f"[green][OK] 已更新提供商为 {provider_input}，模型自动调整为 {config.model}[/green]")
    elif provider_input and provider_input not in ["deepseek", "openai", "claude", "gemini"]:
        console.print("[red][ERR] 无效的提供商[/red]")

    # 修改模型
    model_input = input(f"模型名称 [{config.model}]: ").strip()
    if model_input:
        config.model = model_input
        config_changed = True
        console.print(f"[green][OK] 已更新模型为 {model_input}[/green]")

    # 修改最大迭代次数
    try:
        max_iter_input = input(f"最大迭代次数 [{config.max_iterations}]: ").strip()
        if max_iter_input:
            new_max_iter = int(max_iter_input)
            if new_max_iter > 0:
                config.max_iterations = new_max_iter
                config_changed = True
                console.print(f"[green][OK] 已更新最大迭代次数为 {new_max_iter}[/green]")
            else:
                console.print("[red][ERR] 最大迭代次数必须大于 0[/red]")
    except ValueError:
        console.print("[red][ERR] 无效的数字[/red]")

    # 修改自动批准
    auto_approve_input = input(f"自动批准计划 (y/n) [{'y' if config.auto_approve else 'n'}]: ").strip().lower()
    if auto_approve_input in ['y', 'yes', '是']:
        if not config.auto_approve:
            config.auto_approve = True
            config_changed = True
        console.print("[green][OK] 已启用自动批准[/green]")
    elif auto_approve_input in ['n', 'no', '否']:
        if config.auto_approve:
            config.auto_approve = False
            config_changed = True
        console.print("[green][OK] 已禁用自动批准[/green]")

    # 修改输出目录
    output_dir_input = input(f"输出目录 [{config.output_dir}]: ").strip()
    if output_dir_input:
        config.output_dir = output_dir_input
        config_changed = True
        console.print(f"[green][OK] 已更新输出目录为 {output_dir_input}[/green]")

    # 修改输出格式
    output_format_input = input(f"输出格式 (markdown/html/json) [{config.output_format}]: ").strip().lower()
    if output_format_input in ['markdown', 'md', 'html', 'json']:
        # 规范化格式名称
        normalized_format = 'markdown' if output_format_input in ['markdown', 'md'] else output_format_input
        if normalized_format != config.output_format:
            config.output_format = normalized_format
            config_changed = True
            console.print(f"[green][OK] 已更新输出格式为 {normalized_format.upper()}[/green]")
    elif output_format_input:
        console.print("[red][ERR] 无效的输出格式，请选择 markdown、html 或 json[/red]")

    # 修改显示步骤
    show_steps_input = input(f"显示步骤 (y/n) [{'y' if config.show_steps else 'n'}]: ").strip().lower()
    if show_steps_input in ['y', 'yes', '是']:
        if not config.show_steps:
            config.show_steps = True
            config_changed = True
        console.print("[green][OK] 已启用显示步骤[/green]")
    elif show_steps_input in ['n', 'no', '否']:
        if config.show_steps:
            config.show_steps = False
            config_changed = True
        console.print("[green][OK] 已禁用显示步骤[/green]")

    # 保存配置
    if config_changed:
        console.print()
        save_choice = input("是否保存为永久配置？(y/n) [y]: ").strip().lower()
        if save_choice in ['', 'y', 'yes', '是']:
            save_config_to_file(config)

    print_separator("-")


def human_approval_callback(state: Dict[str, Any]) -> Tuple[bool, str]:
    """
    人在闭环审批回调函数

    Args:
        state: 当前工作流状态

    Returns:
        (approved: bool, feedback: str) - 是否批准和用户反馈
    """
    console.print()
    actions = Table.grid(expand=True)
    actions.add_column("key", style=f"bold {WARNING}", width=4, justify="center")
    actions.add_column("action", style="white", ratio=2)
    actions.add_column("hint", style=SUBTLE, ratio=3)
    actions.add_row("1", "批准计划", "立即开始执行检索")
    actions.add_row("2", "微调计划", "输入反馈后重新生成计划")
    actions.add_row("3", "取消任务", "退出本次研究流程")
    console.print(Panel(
        actions,
        title=f"[bold {WARNING}]等待决策[/]",
        border_style=WARNING,
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    choice = input("请选择操作 [1/2/3]: ").strip()

    if choice == "1":
        # 批准计划
        status_line("审批", "计划已批准，开始研究", SUCCESS)
        return True, None

    elif choice == "2":
        # 拒绝并提供反馈
        console.print(f"\n[{WARNING}]请提供修改意见[/]")
        console.print(f"[{SUBTLE}]例如：聚焦历史、减少检索源、增加近三年发展等[/]\n")

        feedback = input("> ").strip()

        if not feedback:
            status_line("反馈", "未提供具体内容，将重新优化计划", WARNING)
            feedback = "请重新优化研究计划"

        status_line("反馈", "已收到，正在重新制定计划")
        return False, feedback

    elif choice == "3":
        # 取消任务
        status_line("取消", "任务已取消", WARNING)
        raise KeyboardInterrupt("用户取消任务")

    else:
        # 无效选择，默认拒绝
        status_line("错误", "无效选择，请重新决策", ERROR)
        return human_approval_callback(state)


def execute_research(config: CLIConfig, query: str = None) -> None:
    """执行研究任务"""
    console.print(Panel(
        "输入一个开放式研究问题，系统会先生成计划，审批后再检索并生成报告。",
        title=f"[bold {ACCENT}]执行研究任务[/]",
        border_style=SUBTLE,
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    trace = None

    if not query:
        query = input("研究问题 > ").strip()

    if not query:
        status_line("错误", "研究问题不能为空", ERROR)
        return

    logger = None
    try:
        # Setup logger
        logger = setup_logger()

        # Load config from env after applying CLI overrides.
        status_line("准备", "加载配置")
        os.environ['LLM_PROVIDER'] = config.provider
        env_cfg = load_config_from_env()
        env_cfg.llm.model = config.model
        env_cfg.workflow.max_iterations = config.max_iterations
        env_cfg.workflow.auto_approve_plan = config.auto_approve

        trace = create_run_trace(
            query=query,
            provider=config.provider,
            model=config.model,
            mode="research",
        )

        # Create LLM
        status_line("模型", f"初始化 {config.provider.upper()} / {config.model}")
        base_llm = LLMFactory.create_llm(
            provider=env_cfg.llm.provider,
            api_key=env_cfg.llm.api_key,
            model=env_cfg.llm.model
        )
        llm = InstrumentedLLM(base_llm, trace)

        # Create agents
        status_line("智能体", "初始化 Coordinator / Planner / Researcher / Rapporteur")
        coordinator = Coordinator(llm)
        planner = Planner(llm)
        researcher = Researcher(
            llm=llm,
            tavily_api_key=env_cfg.search.tavily_api_key,
            mcp_server_url=env_cfg.search.mcp_server_url,
            mcp_api_key=env_cfg.search.mcp_api_key
        )
        rapporteur = Rapporteur(llm)

        # Create workflow
        status_line("工作流", "装配 LangGraph 状态机")
        workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur)

        # Run workflow
        console.print(Panel(
            query,
            title=f"[bold {SUCCESS}]开始研究[/]",
            border_style=SUCCESS,
            box=box.ROUNDED,
            padding=(1, 2),
        ))

        current_state = None
        last_plan_signature = None
        last_render_key = None

        # Always use stream_interactive to handle interrupts properly
        stream_iter = workflow.stream_interactive(
            query,
            config.max_iterations,
            auto_approve=config.auto_approve,
            human_approval_callback=human_approval_callback if not config.auto_approve else None,
            output_format=config.output_format,
            trace=trace
        )

        for state_update in stream_iter:
            # Debug: check what we got
            if config.show_steps:
                console.print(f"[dim]state_update type: {type(state_update)}[/dim]")

            for node_name, state in state_update.items():
                # Debug: check state type
                if config.show_steps:
                    console.print(f"[dim]node: {node_name}, state type: {type(state)}[/dim]")

                # Handle both dict and tuple states
                if isinstance(state, tuple):
                    # LangGraph might return (values, next_node) tuple
                    if len(state) >= 1:
                        current_state = state[0] if isinstance(state[0], dict) else state
                    else:
                        continue
                else:
                    current_state = state

                # Check if current_state is a dict
                if not isinstance(current_state, dict):
                    if config.show_steps:
                        console.print(f"[yellow]Warning: state is not dict: {type(current_state)}[/yellow]")
                    continue

                step = current_state.get('current_step', 'unknown')

                if config.show_steps:
                    console.print(f"[magenta]步骤：{step}[/magenta]")

                # Check for simple response (greeting/inappropriate query)
                if current_state.get('simple_response'):
                    console.print(f"\n{current_state['simple_response']}\n")
                    current_state = current_state  # Store for later
                    continue

                # Display step updates
                if step == 'planning':
                    plan = current_state.get('research_plan')
                    if plan:
                        plan_signature = json.dumps(plan, sort_keys=True, ensure_ascii=False)
                        if plan_signature != last_plan_signature:
                            console.print(render_plan_panel(plan))
                            last_plan_signature = plan_signature
                    elif last_render_key != ("planning", "pending"):
                        status_line("规划", "正在创建研究计划")
                        last_render_key = ("planning", "pending")

                elif step == 'awaiting_approval':
                    if config.auto_approve and last_render_key != ("approval", "auto"):
                        status_line("审批", "计划已自动批准", SUCCESS)
                        last_render_key = ("approval", "auto")
                    # Interactive approval is handled by the callback in stream_interactive

                elif step == 'researching':
                    task = current_state.get('current_task', {})
                    iteration = current_state.get('iteration_count', 0)
                    render_key = ("researching", iteration, task.get("task_id"))
                    if render_key != last_render_key:
                        console.print(render_task_panel(task, iteration, config.max_iterations))
                        last_render_key = render_key

                elif step == 'generating_report':
                    if last_render_key != ("report", "generating"):
                        status_line("报告", "正在生成最终报告")
                        last_render_key = ("report", "generating")

        # Get final report
        # Check completion status
        if current_state and current_state.get('final_report'):
            report = current_state['final_report']

            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine file extension based on output format
            if current_state.get('output_format') == 'html':
                file_extension = 'html'
            elif current_state.get('output_format') == 'json':
                file_extension = 'json'
            else:
                file_extension = 'md'
            output_path = output_dir / f"research_report_{timestamp}.{file_extension}"

            rapporteur.save_report(report, str(output_path))
            trace = merge_trace_state(trace, current_state.get("trace"))
            trace_path = save_trace(
                trace,
                config.output_dir,
                final_state=current_state,
                report=report,
                report_extension=file_extension,
            )
            console.print()
            console.print(render_completion_panel(
                report=report,
                output_format=current_state.get('output_format', config.output_format),
                report_path=output_path,
                trace_path=trace_path,
            ))

        elif current_state and current_state.get('simple_response'):
            # Simple query was handled, no need to show error
            trace = merge_trace_state(trace, current_state.get("trace"))
            trace_path = save_trace(trace, config.output_dir, final_state=current_state)
            if trace_path:
                status_line("Trace", f"已保存至 {trace_path}", SUCCESS)
        else:
            status_line("错误", "研究未成功完成", ERROR)
            if current_state and isinstance(current_state, dict):
                trace = merge_trace_state(trace, current_state.get("trace"))
            trace_path = save_trace(trace, config.output_dir, final_state=current_state)
            if trace_path:
                status_line("Trace", f"失败轨迹已保存至 {trace_path}", WARNING)

        print_separator("-")

    except KeyboardInterrupt:
        status_line("中断", "任务已被用户中断", WARNING)
        trace_path = save_trace(trace, config.output_dir) if trace else None
        if trace_path:
            status_line("Trace", f"中断轨迹已保存至 {trace_path}", WARNING)
        print_separator("-")
    except Exception as e:
        status_line("错误", f"发生错误：{e}", ERROR)
        if logger:
            logger.exception("Research error")
        trace_path = save_trace(trace, config.output_dir) if trace else None
        if trace_path:
            status_line("Trace", f"错误轨迹已保存至 {trace_path}", WARNING)
        print_separator("-")


def interactive_mode(config: CLIConfig) -> int:
    """交互式菜单模式"""
    print_welcome()

    try:
        while True:
            try:
                print_menu()
                choice = input("请选择操作 (1-5): ").strip()

                if choice == "1":
                    # 执行研究任务
                    execute_research(config)

                elif choice == "2":
                    # 查看可用模型
                    providers = Table.grid(expand=True)
                    providers.add_column("key", style=f"bold {ACCENT}", width=4, justify="center")
                    providers.add_column("provider", style="white")
                    providers.add_row("1", "DeepSeek")
                    providers.add_row("2", "OpenAI")
                    providers.add_row("3", "Claude")
                    providers.add_row("4", "Gemini")
                    console.print(Panel(
                        providers,
                        title=f"[bold {ACCENT}]选择 LLM 提供商[/]",
                        border_style=SUBTLE,
                        box=box.ROUNDED,
                        padding=(1, 2),
                    ))

                    provider_choice = input("选择提供商 [1-4]: ").strip()
                    provider_map = {'1': 'deepseek', '2': 'openai', '3': 'claude', '4': 'gemini'}
                    provider = provider_map.get(provider_choice)

                    if provider:
                        show_models(provider)
                    else:
                        status_line("错误", "无效的选择", ERROR)

                elif choice == "3":
                    # 配置设置
                    configure_settings(config)

                elif choice == "4":
                    # 查看当前配置
                    print_config_info(config)

                elif choice == "5":
                    # 退出程序
                    status_line("退出", "感谢使用 SDYJ 深度研究系统", WARNING)
                    return 0

                else:
                    status_line("错误", "无效的选择，请输入 1-5", ERROR)

            except KeyboardInterrupt:
                status_line("退出", "感谢使用", WARNING)
                return 0
            except EOFError:
                status_line("退出", "感谢使用", WARNING)
                return 0
            except Exception as e:
                status_line("错误", f"发生错误：{e}", ERROR)

    except Exception as e:
        status_line("错误", f"系统错误：{e}", ERROR)
        return 1


def run_single_task(config: CLIConfig, query: str) -> int:
    """运行单个任务（命令行模式）"""
    try:
        execute_research(config, query)
        return 0
    except Exception as e:
        error_console.print(f"[red][ERR] 错误：{e}[/red]")
        return 1


def inspect_run(
    run_id: str | None = None,
    output_dir: str = "./outputs",
    timeline: bool = False,
    event_id: str | None = None,
) -> int:
    """Inspect a persisted run trace."""
    try:
        if not run_id:
            latest = latest_trace_path(output_dir)
            if not latest:
                error_console.print("[red][ERR] 未找到任何 trace 文件[/red]")
                return 1
            trace = load_trace(str(latest), output_dir)
        else:
            trace = load_trace(run_id, output_dir)

        console.print(Panel.fit(
            f"[bold cyan]Run {trace.get('run_id')}[/bold cyan]\n"
            f"mode={trace.get('mode')} provider={trace.get('provider')} model={trace.get('model')}\n"
            f"scenario={trace.get('scenario_id') or 'N/A'}",
            title="SDYJ Trace",
            border_style="cyan",
        ))
        console.print(f"[bold]Query:[/bold] {trace.get('query')}\n")

        summary = Table(title="Trace Summary")
        summary.add_column("Metric")
        summary.add_column("Value", justify="right")
        summary.add_row("Nodes", str(len(trace.get("nodes", []))))
        summary.add_row("LLM calls", str(len(trace.get("llm_calls", []))))
        summary.add_row("Tool calls", str(len(trace.get("tool_calls", []))))
        summary.add_row("Errors", str(len(trace.get("errors", []))))
        for key, value in (trace.get("metrics") or {}).items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            summary.add_row(key, str(value))
        console.print(summary)

        tools = Table(title="Tool Calls")
        tools.add_column("Source")
        tools.add_column("Results", justify="right")
        tools.add_column("Latency ms", justify="right")
        tools.add_column("Error")
        for call in trace.get("tool_calls", []):
            tools.add_row(
                str(call.get("source")),
                str(call.get("result_count")),
                str(call.get("latency_ms")),
                str(call.get("error") or ""),
            )
        if trace.get("tool_calls"):
            console.print(tools)

        if timeline:
            events = iter_timeline_events(trace)
            if event_id:
                events = [event for event in events if event.get("event_id") == event_id]
            timeline_table = Table(title="Timeline")
            timeline_table.add_column("Seq", justify="right")
            timeline_table.add_column("Event")
            timeline_table.add_column("Name")
            timeline_table.add_column("Node")
            timeline_table.add_column("Status")
            timeline_table.add_column("Latency", justify="right")
            timeline_table.add_column("Details")
            for event in events:
                details = event.get("metadata") or {}
                if event.get("error"):
                    details = {**details, "error": event.get("error")}
                timeline_table.add_row(
                    str(event.get("seq") or ""),
                    str(event.get("event_type") or ""),
                    str(event.get("name") or ""),
                    str(event.get("node") or ""),
                    str(event.get("status") or ""),
                    "" if event.get("latency_ms") is None else str(event.get("latency_ms")),
                    json.dumps(details, ensure_ascii=False)[:120],
                )
            console.print(timeline_table)

        if trace.get("errors"):
            console.print("[bold red]Errors[/bold red]")
            for error in trace["errors"]:
                console.print(f"- {error.get('where')}: {error.get('error')}")
        return 0
    except Exception as e:
        error_console.print(f"[red][ERR] Trace 读取失败：{e}[/red]")
        return 1


def replay_run(run_id: str, output_dir: str = "./outputs") -> int:
    """Replay a persisted run with recorded I/O."""
    try:
        trace = load_trace(run_id, output_dir)
        ok, reason = can_deterministically_replay(trace)
        if not ok:
            error_console.print(f"[red][ERR] 无法 deterministic replay：{reason}[/red]")
            return 2
        result = run_deterministic_replay(trace, output_dir=output_dir)
        console.print(Panel.fit(
            f"[bold cyan]Replay completed[/bold cyan]\n"
            f"source={result['source_run_id']}\n"
            f"replay={result['replay_run_id']}",
            title="SDYJ Replay",
            border_style="cyan",
        ))
        console.print(f"[green][OK] replay trace: {result['trace_path']}[/green]")
        return 0
    except Exception as e:
        error_console.print(f"[red][ERR] Replay 失败：{e}[/red]")
        return 1


def diff_runs(left: str, right: str, output_dir: str = "./outputs", as_json: bool = False) -> int:
    """Compare two persisted traces."""
    try:
        left_trace = load_trace(left, output_dir)
        right_trace = load_trace(right, output_dir)
        diff = diff_traces(left_trace, right_trace)
        if as_json:
            console.print(json.dumps(diff, indent=2, ensure_ascii=False))
            return 0

        table = Table(title=f"Trace Diff: {diff['left_run_id']} -> {diff['right_run_id']}")
        table.add_column("Metric")
        table.add_column("Left")
        table.add_column("Right")
        table.add_column("Delta")
        for row in diff["rows"]:
            if not row["changed"] and row["metric"] not in {"run_id"}:
                continue
            table.add_row(
                row["metric"],
                str(row["left"]),
                str(row["right"]),
                str(row["delta"]),
            )
        console.print(table)
        return 0
    except Exception as e:
        error_console.print(f"[red][ERR] Diff 失败：{e}[/red]")
        return 1


def compare_benchmark_summaries(
    baseline_path: str,
    candidate_path: str,
    as_json: bool = False,
) -> int:
    """Compare two benchmark summary JSON files."""
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        with open(candidate_path, "r", encoding="utf-8") as f:
            candidate = json.load(f)

        baseline_by_id = {item["scenario_id"]: item for item in baseline.get("results", [])}
        rows = []
        for item in candidate.get("results", []):
            scenario_id = item["scenario_id"]
            old = baseline_by_id.get(scenario_id, {})
            old_score = old.get("metrics", {}).get("overall_score")
            new_score = item.get("metrics", {}).get("overall_score")
            delta = (
                round(new_score - old_score, 4)
                if isinstance(old_score, (int, float)) and isinstance(new_score, (int, float))
                else None
            )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "baseline_score": old_score,
                    "candidate_score": new_score,
                    "delta": delta,
                    "regressed": isinstance(delta, (int, float)) and delta < -0.02,
                }
            )
        payload = {
            "baseline": baseline_path,
            "candidate": candidate_path,
            "baseline_average": baseline.get("average_score"),
            "candidate_average": candidate.get("average_score"),
            "rows": rows,
            "passed": not any(row["regressed"] for row in rows),
        }
        if as_json:
            console.print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0 if payload["passed"] else 3

        table = Table(title="Benchmark Summary Diff")
        table.add_column("Scenario")
        table.add_column("Baseline", justify="right")
        table.add_column("Candidate", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Status")
        for row in rows:
            table.add_row(
                row["scenario_id"],
                str(row["baseline_score"]),
                str(row["candidate_score"]),
                str(row["delta"]),
                "REGRESSION" if row["regressed"] else "ok",
            )
        console.print(table)
        return 0 if payload["passed"] else 3
    except Exception as e:
        error_console.print(f"[red][ERR] Benchmark compare 失败：{e}[/red]")
        return 1


def list_runs(output_dir: str = "./outputs", limit: int = 20) -> int:
    """List recent run bundles."""
    try:
        runs_dir = Path(output_dir) / "runs"
        if not runs_dir.exists():
            error_console.print("[red][ERR] 未找到 runs 目录[/red]")
            return 1
        traces = sorted(
            runs_dir.glob("*/trace.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:limit]
        table = Table(title="Recent Runs")
        table.add_column("Run ID")
        table.add_column("Mode")
        table.add_column("Provider")
        table.add_column("Model")
        table.add_column("Errors", justify="right")
        table.add_column("Created")
        for path in traces:
            trace = load_trace(str(path), output_dir)
            table.add_row(
                str(trace.get("run_id")),
                str(trace.get("mode")),
                str(trace.get("provider")),
                str(trace.get("model")),
                str(len(trace.get("errors", []))),
                str(trace.get("created_at")),
            )
        console.print(table)
        return 0
    except Exception as e:
        error_console.print(f"[red][ERR] Runs 读取失败：{e}[/red]")
        return 1


def _parse_threshold_overrides(values: list[str] | None) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"threshold 必须使用 metric=value 格式：{value}")
        metric, raw_threshold = value.split("=", 1)
        overrides[metric.strip()] = float(raw_threshold)
    return overrides


def execute_evaluation(args: argparse.Namespace) -> int:
    """Run evaluation scenarios and print a compact dashboard."""
    try:
        summary = run_evaluation(
            live=args.live,
            provider=args.provider,
            model=args.model,
            scenario_ids=args.scenario,
            max_scenarios=args.max_scenarios,
            live_search=args.live_search,
            max_iterations=args.max_iterations,
            output_format=args.output_format,
            output_dir=args.output_dir,
            fail_under=args.fail_under,
            threshold_overrides=_parse_threshold_overrides(args.threshold),
            compare_summary_path=args.compare_summary,
            determinism_repeats=args.determinism_repeats,
        )

        table = Table(title="SDYJ Evaluation")
        table.add_column("Scenario")
        table.add_column("Score", justify="right")
        table.add_column("Plan", justify="right")
        table.add_column("Citations", justify="right")
        table.add_column("Tool OK", justify="right")
        table.add_column("Trace")
        for item in summary["results"]:
            metrics = item["metrics"]
            table.add_row(
                item["scenario_id"],
                f"{metrics['overall_score']:.4f}",
                f"{metrics['plan_coverage']:.2f}",
                f"{metrics['citation_id_coverage']:.2f}",
                f"{metrics['tool_success_rate']:.2f}",
                item["run_id"],
            )
        console.print(table)
        console.print(f"[green][OK] 评测摘要已保存至：{summary['summary_path']}[/green]")
        for item in summary["results"]:
            if item.get("report_path"):
                console.print(f"[dim]report: {item['report_path']}[/dim]")
            if item.get("trace_path"):
                console.print(f"[dim]trace: {item['trace_path']}[/dim]")
        if not summary.get("passed", True):
            console.print("[red][FAIL] Benchmark gate 未通过[/red]")
            for failed in summary.get("failed_scenarios", []):
                console.print(f"[red]- {failed['scenario_id']}: {failed['failed_thresholds']}[/red]")
            return 3
        return 0
    except Exception as e:
        error_console.print(f"[red][ERR] 评测失败：{e}[/red]")
        return 1


def _add_runtime_options(parser: argparse.ArgumentParser, saved_config: Dict[str, Any]) -> None:
    """Add shared options used by research and interactive modes."""
    parser.add_argument(
        "--provider",
        default=saved_config.get("provider", "deepseek"),
        choices=["deepseek", "openai", "claude", "gemini"],
        help="LLM 提供商（默认：deepseek）"
    )
    parser.add_argument(
        "--model",
        default=saved_config.get("model"),
        help="模型名称（默认根据提供商选择）"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=saved_config.get("max_iterations", 5),
        help="最大研究迭代次数（默认：5）"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=saved_config.get("auto_approve", False),
        help="自动批准研究计划"
    )
    parser.add_argument(
        "--output-dir",
        default=saved_config.get("output_dir", "./outputs"),
        help="报告输出目录（默认：./outputs）"
    )
    parser.add_argument(
        "--output-format",
        default=saved_config.get("output_format", "markdown"),
        choices=["markdown", "html", "json"],
        help="报告输出格式（默认：markdown）"
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        default=saved_config.get("show_steps", False),
        help="显示详细执行步骤"
    )


def _create_config_from_args(args: argparse.Namespace) -> CLIConfig:
    """Create CLIConfig from parsed args and provider defaults."""
    if not getattr(args, "model", None):
        args.model = PROVIDER_DEFAULT_MODELS.get(args.provider, "deepseek-v4-flash")

    return CLIConfig(
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
        auto_approve=args.auto_approve,
        output_dir=args.output_dir,
        show_steps=args.show_steps,
        output_format=args.output_format,
    )


def parse_args(argv: Any) -> argparse.Namespace:
    """解析命令行参数，兼容直接传 query 和显式子命令。"""
    saved_config = load_config_from_file()
    argv = list(argv or [])

    root_parser = argparse.ArgumentParser(
        description="SDYJ 深度研究系统 - 基于 LangGraph 的多智能体研究系统",
        epilog=(
            "示例：\n"
            "  python main.py research \"Transformer 架构最新进展\"\n"
            "  python main.py \"Transformer 架构最新进展\"\n"
            "  python main.py list-models deepseek\n"
            "  python main.py eval --max-scenarios 1\n"
            "  python main.py inspect-run\n"
            "  python main.py config-info"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    root_parser.add_argument(
        "--version",
        action="version",
        version="SDYJ Deep Research System 0.5.0"
    )

    if argv and argv[0] in {"-h", "--help", "--version"}:
        root_parser.parse_args(argv)

    if argv and argv[0] == "list-models":
        parser = argparse.ArgumentParser(description="列出指定提供商的可用模型")
        parser.add_argument(
            "provider",
            nargs="?",
            default=saved_config.get("provider", "deepseek"),
            choices=["deepseek", "openai", "claude", "gemini"],
            help="LLM 提供商"
        )
        args = parser.parse_args(argv[1:])
        args.command = "list-models"
        return args

    if argv and argv[0] == "list-scenarios":
        parser = argparse.ArgumentParser(description="列出内置评测场景")
        args = parser.parse_args(argv[1:])
        args.command = "list-scenarios"
        return args

    if argv and argv[0] == "inspect-run":
        parser = argparse.ArgumentParser(description="查看已保存的 run trace")
        parser.add_argument("run_id", nargs="?", help="run_id 或 trace JSON 路径；不传则查看最新 trace")
        parser.add_argument(
            "--output-dir",
            default=saved_config.get("output_dir", "./outputs"),
            help="输出目录（默认：./outputs）",
        )
        parser.add_argument(
            "--timeline",
            action="store_true",
            help="显示 Trace v2 事件时间线",
        )
        parser.add_argument(
            "--event",
            dest="event_id",
            help="只显示指定 event_id 的时间线事件",
        )
        args = parser.parse_args(argv[1:])
        args.command = "inspect-run"
        return args

    if argv and argv[0] == "replay":
        parser = argparse.ArgumentParser(description="使用 trace 中记录的 I/O deterministic replay 一次运行")
        parser.add_argument("run_id", help="run_id 或 trace JSON 路径")
        parser.add_argument(
            "--output-dir",
            default=saved_config.get("output_dir", "./outputs"),
            help="输出目录（默认：./outputs）",
        )
        args = parser.parse_args(argv[1:])
        args.command = "replay"
        return args

    if argv and argv[0] == "diff-runs":
        parser = argparse.ArgumentParser(description="比较两次 run trace")
        parser.add_argument("left", help="基准 run_id 或 trace JSON 路径")
        parser.add_argument("right", help="候选 run_id 或 trace JSON 路径")
        parser.add_argument(
            "--output-dir",
            default=saved_config.get("output_dir", "./outputs"),
            help="输出目录（默认：./outputs）",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="输出 JSON diff",
        )
        args = parser.parse_args(argv[1:])
        args.command = "diff-runs"
        return args

    if argv and argv[0] == "runs":
        parser = argparse.ArgumentParser(description="管理已保存的 run bundle")
        subparsers = parser.add_subparsers(dest="runs_command")
        list_parser = subparsers.add_parser("list", help="列出最近 run")
        list_parser.add_argument(
            "--output-dir",
            default=saved_config.get("output_dir", "./outputs"),
            help="输出目录（默认：./outputs）",
        )
        list_parser.add_argument("--limit", type=int, default=20, help="最多显示多少条")
        args = parser.parse_args(argv[1:])
        args.command = "runs"
        args.runs_command = args.runs_command or "list"
        return args

    if argv and argv[0] == "benchmark" and len(argv) > 1 and argv[1] == "compare":
        parser = argparse.ArgumentParser(description="比较两个 benchmark summary JSON")
        parser.add_argument("baseline", help="基准 eval_summary JSON")
        parser.add_argument("candidate", help="候选 eval_summary JSON")
        parser.add_argument("--json", action="store_true", help="输出 JSON diff")
        args = parser.parse_args(argv[2:])
        args.command = "benchmark-compare"
        return args

    if argv and argv[0] in {"eval", "benchmark"}:
        command_name = argv[0]
        if command_name == "benchmark":
            argv = argv[:1] + (argv[2:] if len(argv) > 1 and argv[1] == "run" else argv[1:])
        parser = argparse.ArgumentParser(description="运行 SDYJ Agent 评测套件")
        parser.add_argument(
            "--live",
            action="store_true",
            help="使用真实 LLM（默认 provider=deepseek）；不加则使用 fake LLM 做离线可复现评测",
        )
        parser.add_argument(
            "--provider",
            default=saved_config.get("provider", "deepseek"),
            choices=["deepseek", "openai", "claude", "gemini"],
            help="真实评测使用的 LLM 提供商",
        )
        parser.add_argument(
            "--model",
            default=saved_config.get("model"),
            help="真实评测使用的模型；不填则使用环境变量或提供商默认",
        )
        parser.add_argument(
            "--scenario",
            action="append",
            help="只运行指定场景 ID；可重复传入",
        )
        parser.add_argument(
            "--max-scenarios",
            type=int,
            default=None,
            help="最多运行多少个场景",
        )
        parser.add_argument(
            "--live-search",
            action="store_true",
            help="同时使用真实搜索工具；默认使用 canned evidence 保证可复现",
        )
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=saved_config.get("max_iterations", 3),
            help="每个场景最大研究迭代次数",
        )
        parser.add_argument(
            "--output-dir",
            default=saved_config.get("output_dir", "./outputs"),
            help="评测报告和 trace 输出目录",
        )
        parser.add_argument(
            "--output-format",
            default=saved_config.get("output_format", "markdown"),
            choices=["markdown", "html", "json"],
            help="评测报告输出格式",
        )
        parser.add_argument(
            "--fail-under",
            type=float,
            default=None,
            help="平均分低于该值时返回非 0，适合 CI gate",
        )
        parser.add_argument(
            "--threshold",
            action="append",
            help="覆盖单个指标阈值，格式 metric=value，可重复传入",
        )
        parser.add_argument(
            "--compare-summary",
            help="和历史 eval_summary JSON 比较，出现明显回退时 gate 失败",
        )
        parser.add_argument(
            "--determinism-repeats",
            type=int,
            default=1,
            help="离线模式重复运行次数，用于检查 benchmark 确定性",
        )
        args = parser.parse_args(argv[1:])
        args.command = "eval"
        return args

    if argv and argv[0] == "config-info":
        parser = argparse.ArgumentParser(description="显示当前配置")
        _add_runtime_options(parser, saved_config)
        args = parser.parse_args(argv[1:])
        args.command = "config-info"
        args.query = None
        args.interactive = False
        return args

    if argv and argv[0] == "research":
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="执行深度研究任务，或不提供 query 进入交互模式"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="研究问题或主题（可选，不提供则进入交互模式）"
    )
    _add_runtime_options(parser, saved_config)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启动交互式菜单模式"
    )
    args = parser.parse_args(argv)
    args.command = "research" if args.query else "interactive"
    return args


def main(argv: Any = None) -> int:
    """主入口函数"""
    load_dotenv()
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "list-models":
        show_models(args.provider)
        return 0

    if args.command == "list-scenarios":
        table = Table(title="Built-in Evaluation Scenarios")
        table.add_column("ID")
        table.add_column("Title")
        for scenario in list_scenarios():
            table.add_row(scenario["id"], scenario["title"])
        console.print(table)
        return 0

    if args.command == "inspect-run":
        return inspect_run(args.run_id, args.output_dir, timeline=args.timeline, event_id=args.event_id)

    if args.command == "replay":
        return replay_run(args.run_id, args.output_dir)

    if args.command == "diff-runs":
        return diff_runs(args.left, args.right, args.output_dir, as_json=args.json)

    if args.command == "runs":
        if args.runs_command == "list":
            return list_runs(args.output_dir, limit=args.limit)
        return 1

    if args.command == "benchmark-compare":
        return compare_benchmark_summaries(args.baseline, args.candidate, as_json=args.json)

    if args.command == "eval":
        if args.live and not get_api_key_for_provider(args.provider):
            expected_envs = " 或 ".join(PROVIDER_API_KEY_ENVS.get(args.provider, ()))
            error_console.print("[red][ERR] live 评测缺少 API 密钥。[/red]")
            error_console.print(f"请在 .env 文件中设置 {expected_envs}")
            return 2
        return execute_evaluation(args)

    config = _create_config_from_args(args)

    if args.command == "config-info":
        print_config_info(config)
        return 0

    # 检查 API 密钥
    api_key = get_api_key_for_provider(config.provider)
    if not api_key:
        expected_envs = " 或 ".join(PROVIDER_API_KEY_ENVS.get(config.provider, ()))
        error_console.print("[red][ERR] 缺少 API 密钥。[/red]")
        error_console.print(f"请在 .env 文件中设置 {expected_envs}")
        return 2

    # 如果提供了任务参数，直接执行任务
    if args.command == "research" and args.query:
        return run_single_task(config, args.query)

    # 如果指定了交互模式或没有提供任务，进入交互式菜单
    if args.interactive or args.command == "interactive":
        return interactive_mode(config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
