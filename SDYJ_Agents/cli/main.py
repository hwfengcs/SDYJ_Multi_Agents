"""
CLI Main Program

This module provides the command-line interface for the SDYJ research system.
Refactored to follow the example.py structure with argparse and config persistence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from ..utils.config import load_config_from_env
from ..utils.logger import setup_logger
from ..llm.factory import LLMFactory
from ..agents.coordinator import Coordinator
from ..agents.planner import Planner
from ..agents.researcher import Researcher
from ..agents.rapporteur import Rapporteur
from ..workflow.graph import ResearchWorkflow

console = Console()


@dataclass
class CLIConfig:
    """CLI运行时配置"""
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    max_iterations: int = 5
    auto_approve: bool = False
    output_dir: str = "./outputs"
    show_steps: bool = False
    output_format: str = "markdown"  # "markdown" or "html"


# 配置文件路径
CONFIG_FILE = Path(__file__).parent.parent.parent / "config.json"


def load_config_from_file() -> Dict[str, Any]:
    """从配置文件加载设置"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[yellow]⚠ 配置文件加载失败：{e}，使用默认设置[/yellow]")
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
        console.print("[green]✓ 配置已保存[/green]")
    except Exception as e:
        console.print(f"[red]✗ 配置保存失败：{e}[/red]")


def get_api_key_for_provider(provider: str) -> str | None:
    """根据提供商获取对应的 API 密钥"""
    provider_env_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "CLAUDE_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = provider_env_map.get(provider.lower())
    return os.getenv(env_var) if env_var else None


def print_separator(char: str = "─", length: int = 70) -> None:
    """打印分隔线"""
    console.print(f"[cyan]{char * length}[/cyan]")


def print_header(text: str) -> None:
    """打印标题"""
    console.print(Panel.fit(
        f"[bold cyan]{text}[/bold cyan]",
        border_style="cyan"
    ))


def print_welcome() -> None:
    """打印欢迎界面"""
    console.print("\n")
    print_header("SDYJ 深度研究系统")
    console.print("[yellow]欢迎使用基于 LangGraph 的多智能体研究系统！[/yellow]")

    # 显示配置文件状态
    if CONFIG_FILE.exists():
        console.print(f"[green]✓ 已加载配置文件: {CONFIG_FILE.name}[/green]")
    else:
        console.print("[cyan]ℹ 使用默认配置 (max_iterations=5, auto_approve=False)[/cyan]")
    console.print()


def print_menu() -> None:
    """打印主菜单"""
    console.print("\n[bold cyan]主菜单：[/bold cyan]\n")
    console.print("  [green]1.[/green] 执行研究任务")
    console.print("  [green]2.[/green] 查看可用模型")
    console.print("  [green]3.[/green] 配置设置")
    console.print("  [green]4.[/green] 查看当前配置")
    console.print("  [green]5.[/green] 退出程序")
    console.print()


def show_models(provider: str) -> None:
    """显示可用模型列表"""
    models = {
        'openai': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        'claude': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229'],
        'gemini': ['gemini-pro', 'gemini-1.5-pro'],
        'deepseek': ['deepseek-chat', 'deepseek-coder']
    }

    print_separator("-")
    console.print(f"\n[bold cyan]{provider.upper()} 的可用模型：[/bold cyan]\n")

    for model in models.get(provider, []):
        console.print(f"  • {model}")
    console.print()
    print_separator("-")


def configure_settings(config: CLIConfig) -> None:
    """配置设置"""
    print_separator("-")
    console.print("[bold cyan]当前配置：[/bold cyan]\n")
    console.print(f"  提供商：[yellow]{config.provider}[/yellow]")
    console.print(f"  模型：[yellow]{config.model}[/yellow]")
    console.print(f"  最大迭代次数：[yellow]{config.max_iterations}[/yellow]")
    console.print(f"  自动批准计划：[yellow]{'是' if config.auto_approve else '否'}[/yellow]")
    console.print(f"  输出目录：[yellow]{config.output_dir}[/yellow]")
    console.print(f"  输出格式：[yellow]{config.output_format.upper()}[/yellow]")
    console.print(f"  显示步骤：[yellow]{'是' if config.show_steps else '否'}[/yellow]")
    console.print()

    console.print("[cyan]选择要修改的设置（直接回车跳过）：[/cyan]\n")

    config_changed = False

    # 修改提供商
    provider_input = input(f"LLM 提供商 (deepseek/openai/claude/gemini) [{config.provider}]: ").strip().lower()
    if provider_input and provider_input in ["deepseek", "openai", "claude", "gemini"]:
        if provider_input != config.provider:
            # 检查 API 密钥
            new_api_key = get_api_key_for_provider(provider_input)
            if not new_api_key:
                console.print(f"[red]✗ 未找到 {provider_input.upper()}_API_KEY 环境变量[/red]")
                console.print(f"[yellow]请在 .env 文件中配置 {provider_input.upper()}_API_KEY[/yellow]")
            else:
                config.provider = provider_input
                # 自动更新默认模型
                model_defaults = {
                    'deepseek': 'deepseek-chat',
                    'openai': 'gpt-4',
                    'claude': 'claude-3-5-sonnet-20241022',
                    'gemini': 'gemini-pro'
                }
                config.model = model_defaults.get(provider_input, config.model)
                config_changed = True
                console.print(f"[green]✓ 已更新提供商为 {provider_input}，模型自动调整为 {config.model}[/green]")
    elif provider_input and provider_input not in ["deepseek", "openai", "claude", "gemini"]:
        console.print("[red]✗ 无效的提供商[/red]")

    # 修改模型
    model_input = input(f"模型名称 [{config.model}]: ").strip()
    if model_input:
        config.model = model_input
        config_changed = True
        console.print(f"[green]✓ 已更新模型为 {model_input}[/green]")

    # 修改最大迭代次数
    try:
        max_iter_input = input(f"最大迭代次数 [{config.max_iterations}]: ").strip()
        if max_iter_input:
            new_max_iter = int(max_iter_input)
            if new_max_iter > 0:
                config.max_iterations = new_max_iter
                config_changed = True
                console.print(f"[green]✓ 已更新最大迭代次数为 {new_max_iter}[/green]")
            else:
                console.print("[red]✗ 最大迭代次数必须大于 0[/red]")
    except ValueError:
        console.print("[red]✗ 无效的数字[/red]")

    # 修改自动批准
    auto_approve_input = input(f"自动批准计划 (y/n) [{'y' if config.auto_approve else 'n'}]: ").strip().lower()
    if auto_approve_input in ['y', 'yes', '是']:
        if not config.auto_approve:
            config.auto_approve = True
            config_changed = True
        console.print("[green]✓ 已启用自动批准[/green]")
    elif auto_approve_input in ['n', 'no', '否']:
        if config.auto_approve:
            config.auto_approve = False
            config_changed = True
        console.print("[green]✓ 已禁用自动批准[/green]")

    # 修改输出目录
    output_dir_input = input(f"输出目录 [{config.output_dir}]: ").strip()
    if output_dir_input:
        config.output_dir = output_dir_input
        config_changed = True
        console.print(f"[green]✓ 已更新输出目录为 {output_dir_input}[/green]")

    # 修改输出格式
    output_format_input = input(f"输出格式 (markdown/html) [{config.output_format}]: ").strip().lower()
    if output_format_input in ['markdown', 'md', 'html']:
        # 规范化格式名称
        normalized_format = 'markdown' if output_format_input in ['markdown', 'md'] else 'html'
        if normalized_format != config.output_format:
            config.output_format = normalized_format
            config_changed = True
            console.print(f"[green]✓ 已更新输出格式为 {normalized_format.upper()}[/green]")
    elif output_format_input:
        console.print("[red]✗ 无效的输出格式，请选择 markdown 或 html[/red]")

    # 修改显示步骤
    show_steps_input = input(f"显示步骤 (y/n) [{'y' if config.show_steps else 'n'}]: ").strip().lower()
    if show_steps_input in ['y', 'yes', '是']:
        if not config.show_steps:
            config.show_steps = True
            config_changed = True
        console.print("[green]✓ 已启用显示步骤[/green]")
    elif show_steps_input in ['n', 'no', '否']:
        if config.show_steps:
            config.show_steps = False
            config_changed = True
        console.print("[green]✓ 已禁用显示步骤[/green]")

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
    console.print("\n")
    print_separator("=")
    console.print("[bold yellow]等待您的决策[/bold yellow]\n")

    console.print("[cyan]您可以选择：[/cyan]")
    console.print("  [green]1.[/green] 批准计划 - 开始执行研究")
    console.print("  [green]2.[/green] 拒绝计划 - 提供反馈重新制定")
    console.print("  [green]3.[/green] 取消任务 - 退出研究")
    console.print()

    choice = input("请选择操作 (1-3): ").strip()

    if choice == "1":
        # 批准计划
        console.print("[green]✓ 计划已批准，开始研究...[/green]\n")
        print_separator("=")
        return True, None

    elif choice == "2":
        # 拒绝并提供反馈
        console.print("\n[yellow]请提供修改意见（描述您希望如何调整研究计划）：[/yellow]")
        console.print("[dim]提示：您可以要求增加/删除某些研究方向，调整优先级等[/dim]\n")

        feedback = input("> ").strip()

        if not feedback:
            console.print("[yellow]未提供反馈，将重新生成计划...[/yellow]")
            feedback = "请重新优化研究计划"

        console.print(f"\n[cyan]已收到反馈，正在重新制定计划...[/cyan]\n")
        print_separator("=")
        return False, feedback

    elif choice == "3":
        # 取消任务
        console.print("\n[yellow]任务已取消[/yellow]")
        raise KeyboardInterrupt("用户取消任务")

    else:
        # 无效选择，默认拒绝
        console.print("[red]无效选择，请重新决策[/red]")
        return human_approval_callback(state)


def execute_research(config: CLIConfig, query: str = None) -> None:
    """执行研究任务"""
    print_separator("-")
    console.print("[bold cyan]执行研究任务[/bold cyan]\n")

    if not query:
        query = input("请输入研究问题：\n> ").strip()

    if not query:
        console.print("[red]✗ 研究问题不能为空[/red]")
        return

    try:
        # Setup logger
        logger = setup_logger()

        # Load config from env
        console.print("\n[dim]正在加载配置...[/dim]")
        env_cfg = load_config_from_env()

        # Override with CLI config
        os.environ['LLM_PROVIDER'] = config.provider
        env_cfg = load_config_from_env()  # Reload
        env_cfg.llm.model = config.model
        env_cfg.workflow.max_iterations = config.max_iterations
        env_cfg.workflow.auto_approve_plan = config.auto_approve

        # Create LLM
        console.print(f"[dim]正在初始化 {config.provider.upper()} LLM...[/dim]")
        llm = LLMFactory.create_llm(
            provider=env_cfg.llm.provider,
            api_key=env_cfg.llm.api_key,
            model=env_cfg.llm.model
        )

        # Create agents
        console.print("[dim]正在初始化智能体...[/dim]")
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
        console.print("[dim]正在设置研究工作流...[/dim]\n")
        workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur)

        # Run workflow
        print_separator("-")
        console.print(f"[bold green]开始研究：[/bold green]{query}\n")

        current_state = None

        # Always use stream_interactive to handle interrupts properly
        stream_iter = workflow.stream_interactive(
            query,
            config.max_iterations,
            auto_approve=config.auto_approve,
            human_approval_callback=human_approval_callback if not config.auto_approve else None,
            output_format=config.output_format
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
                    console.print("[cyan]正在创建研究计划...[/cyan]")
                    if current_state.get('research_plan'):
                        plan_display = planner.format_plan_for_display(current_state['research_plan'])
                        console.print(Panel(plan_display, title="研究计划", border_style="blue"))

                elif step == 'awaiting_approval':
                    if config.auto_approve:
                        console.print("[green]✓ 计划已自动批准[/green]")
                    # Interactive approval is handled by the callback in stream_interactive

                elif step == 'researching':
                    task = current_state.get('current_task', {})
                    iteration = current_state.get('iteration_count', 0)
                    console.print(f"[cyan]正在研究：{task.get('description', '未知任务')}[/cyan]")
                    console.print(f"[dim]迭代 {iteration}/{config.max_iterations}[/dim]")

                elif step == 'generating_report':
                    console.print("[cyan]正在生成最终报告...[/cyan]")

        # Get final report
        # Check completion status
        if current_state and current_state.get('final_report'):
            report = current_state['final_report']

            # Display report
            console.print("\n")
            console.print(Panel(
                Markdown(report),
                title="研究报告",
                border_style="green"
            ))

            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine file extension based on output format
            file_extension = 'html' if current_state.get('output_format') == 'html' else 'md'
            output_path = output_dir / f"research_report_{timestamp}.{file_extension}"

            rapporteur.save_report(report, str(output_path))
            console.print(f"\n[green]✓ 报告已保存至：{output_path}[/green]")

        elif current_state and current_state.get('simple_response'):
            # Simple query was handled, no need to show error
            pass
        else:
            console.print("[red]✗ 研究未成功完成[/red]")

        print_separator("-")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]任务已被用户中断[/yellow]")
        print_separator("-")
    except Exception as e:
        console.print(f"\n[red]✗ 发生错误：{e}[/red]")
        logger.exception("Research error")
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
                    console.print("\n[bold]选择 LLM 提供商：[/bold]\n")
                    console.print("  [cyan]1[/cyan] - DeepSeek")
                    console.print("  [cyan]2[/cyan] - OpenAI")
                    console.print("  [cyan]3[/cyan] - Claude")
                    console.print("  [cyan]4[/cyan] - Gemini")

                    provider_choice = input("\n选择提供商 (1-4): ").strip()
                    provider_map = {'1': 'deepseek', '2': 'openai', '3': 'claude', '4': 'gemini'}
                    provider = provider_map.get(provider_choice)

                    if provider:
                        show_models(provider)
                    else:
                        console.print("[red]✗ 无效的选择[/red]")

                elif choice == "3":
                    # 配置设置
                    configure_settings(config)

                elif choice == "4":
                    # 查看当前配置
                    print_separator("-")
                    console.print("[bold cyan]当前配置：[/bold cyan]\n")
                    console.print(f"  提供商：[yellow]{config.provider}[/yellow]")
                    console.print(f"  模型：[yellow]{config.model}[/yellow]")
                    console.print(f"  最大迭代次数：[yellow]{config.max_iterations}[/yellow]")
                    console.print(f"  自动批准：[yellow]{'是' if config.auto_approve else '否'}[/yellow]")
                    console.print(f"  输出目录：[yellow]{config.output_dir}[/yellow]")
                    console.print(f"  输出格式：[yellow]{config.output_format.upper()}[/yellow]")
                    console.print(f"  显示步骤：[yellow]{'是' if config.show_steps else '否'}[/yellow]")
                    console.print()
                    print_separator("-")

                elif choice == "5":
                    # 退出程序
                    console.print("\n[yellow]感谢使用 SDYJ 深度研究系统！再见！[/yellow]\n")
                    return 0

                else:
                    console.print("[red]✗ 无效的选择，请输入 1-5[/red]")

            except KeyboardInterrupt:
                console.print("\n\n[yellow]感谢使用！再见！[/yellow]\n")
                return 0
            except EOFError:
                console.print("\n\n[yellow]感谢使用！再见！[/yellow]\n")
                return 0
            except Exception as e:
                console.print(f"\n[red]✗ 发生错误：{e}[/red]\n")

    except Exception as e:
        console.print(f"\n[red]✗ 系统错误：{e}[/red]\n")
        return 1


def run_single_task(config: CLIConfig, query: str) -> int:
    """运行单个任务（命令行模式）"""
    try:
        execute_research(config, query)
        return 0
    except Exception as e:
        console.print(f"[red]✗ 错误：{e}[/red]", file=sys.stderr)
        return 1


def parse_args(argv: Any) -> argparse.Namespace:
    """解析命令行参数"""
    # 先加载配置文件中的默认值
    saved_config = load_config_from_file()

    parser = argparse.ArgumentParser(
        description="SDYJ 深度研究系统 - 基于 LangGraph 的多智能体研究系统"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="研究问题或主题（可选，不提供则进入交互模式）"
    )
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
        choices=["markdown", "html"],
        help="报告输出格式（默认：markdown）"
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        default=saved_config.get("show_steps", False),
        help="显示详细执行步骤"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启动交互式菜单模式"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="SDYJ Deep Research System 0.1.0"
    )

    return parser.parse_args(argv)


def main(argv: Any = None) -> int:
    """主入口函数"""
    load_dotenv()
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # 检查 API 密钥
    api_key = get_api_key_for_provider(args.provider)
    if not api_key:
        console.print(f"[red]✗ 缺少 API 密钥。[/red]", file=sys.stderr)
        console.print(f"请在 .env 文件中设置 {args.provider.upper()}_API_KEY", file=sys.stderr)
        return 2

    # 如果没有指定模型，使用默认模型
    if not args.model:
        model_defaults = {
            'deepseek': 'deepseek-chat',
            'openai': 'gpt-4',
            'claude': 'claude-3-5-sonnet-20241022',
            'gemini': 'gemini-pro'
        }
        args.model = model_defaults.get(args.provider, 'deepseek-chat')

    # 创建配置
    config = CLIConfig(
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
        auto_approve=args.auto_approve,
        output_dir=args.output_dir,
        show_steps=args.show_steps,
        output_format=args.output_format,
    )

    # 如果提供了任务参数，直接执行任务
    if args.query:
        return run_single_task(config, args.query)

    # 如果指定了交互模式或没有提供任务，进入交互式菜单
    if args.interactive or not args.query:
        return interactive_mode(config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
