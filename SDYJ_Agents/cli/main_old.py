"""
CLI Main Program

This module provides the command-line interface for the research system.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from datetime import datetime

from ..utils.config import load_config_from_env
from ..utils.logger import setup_logger, print_success, print_error, print_info, print_step
from ..llm.factory import LLMFactory
from ..agents.coordinator import Coordinator
from ..agents.planner import Planner
from ..agents.researcher import Researcher
from ..agents.rapporteur import Rapporteur
from ..workflow.graph import ResearchWorkflow

console = Console()


def interactive_menu():
    """
    显示交互式菜单供用户选择操作。
    """
    while True:
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]SDYJ 深度研究系统[/bold cyan]\n"
            "[dim]基于 LangGraph 的多智能体研究系统[/dim]",
            border_style="cyan"
        ))

        console.print("\n[bold]主菜单：[/bold]\n")
        console.print("  [cyan]1[/cyan] - 开始研究任务")
        console.print("  [cyan]2[/cyan] - 查看配置信息")
        console.print("  [cyan]3[/cyan] - 列出可用模型")
        console.print("  [cyan]0[/cyan] - 退出系统")

        choice = click.prompt("\n请选择操作", type=str, default='1')

        if choice == '1':
            # Start research
            console.print("\n")

            # Ask if user wants to change LLM configuration
            change_llm = click.confirm('是否修改 LLM 配置？', default=False)

            llm_provider = None
            llm_model = None

            if change_llm:
                console.print("\n[bold]选择 LLM 提供商：[/bold]\n")
                console.print("  [cyan]1[/cyan] - DeepSeek (默认)")
                console.print("  [cyan]2[/cyan] - OpenAI")
                console.print("  [cyan]3[/cyan] - Claude")
                console.print("  [cyan]4[/cyan] - Gemini")

                provider_choice = click.prompt('\n选择提供商', type=str, default='1')
                provider_map = {
                    '1': 'deepseek',
                    '2': 'openai',
                    '3': 'claude',
                    '4': 'gemini'
                }
                llm_provider = provider_map.get(provider_choice, 'deepseek')

                # Ask for model
                model_map = {
                    'deepseek': ['deepseek-chat', 'deepseek-coder'],
                    'openai': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                    'claude': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'],
                    'gemini': ['gemini-pro', 'gemini-1.5-pro']
                }

                console.print(f"\n[bold]{llm_provider.upper()} 可用模型：[/bold]\n")
                for i, model in enumerate(model_map[llm_provider], 1):
                    console.print(f"  [cyan]{i}[/cyan] - {model}")

                model_choice = click.prompt('\n选择模型（直接回车使用默认）', type=str, default='1')
                try:
                    model_idx = int(model_choice) - 1
                    if 0 <= model_idx < len(model_map[llm_provider]):
                        llm_model = model_map[llm_provider][model_idx]
                except:
                    llm_model = model_map[llm_provider][0]

            console.print("\n")
            query = click.prompt('请输入您的研究问题')

            # Ask for options
            console.print("\n[dim]可选设置（直接回车使用默认值）：[/dim]")
            max_iterations = click.prompt('最大迭代次数', type=int, default=5, show_default=True)
            auto_approve = click.confirm('是否自动批准研究计划？', default=False)
            output_file = click.prompt('输出文件路径（可选）', type=str, default='', show_default=False)

            # Create Click context and invoke research command
            from click.testing import CliRunner
            runner = CliRunner()

            args = [query]
            if max_iterations != 5:
                args.extend(['--max-iterations', str(max_iterations)])
            if auto_approve:
                args.append('--auto-approve')
            if output_file:
                args.extend(['--output', output_file])

            # Call research function directly
            ctx = click.Context(research)
            ctx.invoke(
                research,
                query=query,
                config=None,
                output=output_file if output_file else None,
                max_iterations=max_iterations if max_iterations != 5 else None,
                auto_approve=auto_approve,
                llm_provider=llm_provider,
                llm_model=llm_model
            )

            # Ask if continue
            if not click.confirm('\n返回主菜单？', default=True):
                break

        elif choice == '2':
            # View configuration
            console.print("\n")
            ctx = click.Context(config_info)
            ctx.invoke(config_info)
            click.prompt('\n按回车键继续', default='', show_default=False)

        elif choice == '3':
            # List models
            console.print("\n[bold]选择 LLM 提供商：[/bold]\n")
            console.print("  [cyan]1[/cyan] - OpenAI")
            console.print("  [cyan]2[/cyan] - Claude")
            console.print("  [cyan]3[/cyan] - Gemini")
            console.print("  [cyan]4[/cyan] - DeepSeek")

            provider_choice = click.prompt('\n选择提供商', type=str, default='1')
            provider_map = {
                '1': 'openai',
                '2': 'claude',
                '3': 'gemini',
                '4': 'deepseek'
            }

            provider = provider_map.get(provider_choice)
            if provider:
                console.print("\n")
                ctx = click.Context(list_models)
                ctx.invoke(list_models, provider=provider)
            else:
                print_error("无效的提供商选择")

            click.prompt('\n按回车键继续', default='', show_default=False)

        elif choice == '0':
            console.print("\n[green]感谢使用 SDYJ 深度研究系统！[/green]\n")
            break

        else:
            print_error("无效选项，请选择 0-3。")


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version="0.1.0")
def cli(ctx):
    """
    SDYJ 深度研究 - 基于 LangGraph 的多智能体研究系统。
    """
    if ctx.invoked_subcommand is None:
        # No subcommand provided, show interactive menu
        interactive_menu()


@cli.command()
@click.argument('query', required=False)
@click.option('--config', '-c', help='配置文件路径')
@click.option('--output', '-o', help='报告输出文件路径')
@click.option('--max-iterations', '-m', type=int, help='最大研究迭代次数')
@click.option('--auto-approve', is_flag=True, help='自动批准研究计划')
@click.option('--llm-provider', type=click.Choice(['openai', 'claude', 'gemini', 'deepseek']), help='LLM 提供商')
@click.option('--llm-model', help='LLM 模型名称')
def research(query, config, output, max_iterations, auto_approve, llm_provider, llm_model):
    """
    开始新的研究任务。

    QUERY: 研究问题或主题（也可以交互式输入）
    """
    # Setup logger
    logger = setup_logger()

    # Display banner
    console.print(Panel.fit(
        "[bold cyan]SDYJ 深度研究系统[/bold cyan]\n"
        "[dim]基于 LangGraph 的多智能体研究系统[/dim]",
        border_style="cyan"
    ))

    # Get query if not provided
    if not query:
        query = click.prompt('\n请输入您的研究问题')

    print_step(f"研究问题：{query}")

    try:
        # Load configuration
        print_step("正在加载配置...")
        cfg = load_config_from_env()

        # Override with CLI parameters
        if llm_provider:
            import os
            os.environ['LLM_PROVIDER'] = llm_provider
            cfg = load_config_from_env()  # Reload with new provider

        if llm_model:
            cfg.llm.model = llm_model

        if max_iterations:
            cfg.workflow.max_iterations = max_iterations

        if auto_approve:
            cfg.workflow.auto_approve_plan = True

        # Create LLM
        print_step(f"正在初始化 {cfg.llm.provider.upper()} LLM...")
        llm = LLMFactory.create_llm(
            provider=cfg.llm.provider,
            api_key=cfg.llm.api_key,
            model=cfg.llm.model
        )

        # Create agents
        print_step("正在初始化智能体...")
        coordinator = Coordinator(llm)
        planner = Planner(llm)
        researcher = Researcher(
            llm=llm,
            tavily_api_key=cfg.search.tavily_api_key,
            mcp_server_url=cfg.search.mcp_server_url,
            mcp_api_key=cfg.search.mcp_api_key
        )
        rapporteur = Rapporteur(llm)

        # Create workflow
        print_step("正在设置研究工作流...")
        workflow = ResearchWorkflow(coordinator, planner, researcher, rapporteur)

        # Run workflow with streaming
        print_step("开始研究...\n")

        current_state = None
        for state_update in workflow.stream(query, cfg.workflow.max_iterations, auto_approve=cfg.workflow.auto_approve_plan):
            for node_name, state in state_update.items():
                current_state = state
                step = state.get('current_step', 'unknown')

                # Display step updates
                if step == 'planning':
                    print_info("正在创建研究计划...")

                    if state.get('research_plan'):
                        plan_display = planner.format_plan_for_display(state['research_plan'])
                        console.print(Panel(plan_display, title="研究计划", border_style="blue"))

                elif step == 'awaiting_approval':
                    if not cfg.workflow.auto_approve_plan:
                        print_info("计划需要您的批准")
                        # Note: Interactive approval not currently supported in streaming mode
                        # For now, user must use --auto-approve flag or set AUTO_APPROVE_PLAN in .env
                        print_error("交互式批准暂不支持，请使用 --auto-approve 或在 .env 中设置 AUTO_APPROVE_PLAN=true")
                        return
                    else:
                        print_success("计划已自动批准")

                elif step == 'researching':
                    task = state.get('current_task', {})
                    print_step(f"正在研究：{task.get('description', '未知任务')}")
                    print_info(f"迭代 {state.get('iteration_count', 0)}/{cfg.workflow.max_iterations}")

                elif step == 'generating_report':
                    print_step("正在生成最终报告...")

        # Get final report
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
            if not output:
                # Create default output path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = Path(cfg.workflow.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output = output_dir / f"research_report_{timestamp}.md"

            rapporteur.save_report(report, str(output))
            print_success(f"报告已保存至：{output}")

        else:
            print_error("研究未成功完成")

    except Exception as e:
        print_error(f"研究过程中出错：{str(e)}")
        logger.exception("Research error")
        raise click.Abort()


@cli.command()
def config_info():
    """
    显示当前配置信息。
    """
    try:
        cfg = load_config_from_env()
        console.print(Panel(
            f"[bold]LLM 提供商：[/bold] {cfg.llm.provider}\n"
            f"[bold]LLM 模型：[/bold] {cfg.llm.model or '默认'}\n"
            f"[bold]最大迭代次数：[/bold] {cfg.workflow.max_iterations}\n"
            f"[bold]输出目录：[/bold] {cfg.workflow.output_dir}\n"
            f"[bold]Tavily API：[/bold] {'已配置' if cfg.search.tavily_api_key else '未配置'}\n"
            f"[bold]MCP 服务器：[/bold] {'已配置' if cfg.search.mcp_server_url else '未配置'}",
            title="配置信息",
            border_style="blue"
        ))
    except Exception as e:
        print_error(f"加载配置时出错：{str(e)}")


@cli.command()
@click.argument('provider', type=click.Choice(['openai', 'claude', 'gemini', 'deepseek']))
def list_models(provider):
    """
    列出指定提供商的可用模型。
    """
    models = {
        'openai': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        'claude': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229'],
        'gemini': ['gemini-pro', 'gemini-pro-vision', 'gemini-ultra'],
        'deepseek': ['deepseek-chat', 'deepseek-coder']
    }

    console.print(f"\n[bold]{provider} 的可用模型：[/bold]\n")
    for model in models.get(provider, []):
        console.print(f"  • {model}")
    console.print()


if __name__ == '__main__':
    cli()
