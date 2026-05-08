"""One-command local release-readiness preflight.

The checks in this module intentionally avoid publishing, pushing images, or
calling live provider/search APIs. External prerequisites are reported as
``BLOCKED`` instead of being treated as completed work.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


Command = tuple[str, ...]
CommandRunner = Callable[[Command, Path, int, dict[str, str]], "CommandExecution"]


@dataclass(frozen=True)
class ReleaseGate:
    """A single release-readiness gate."""

    name: str
    command: Command = ()
    required: bool = True
    kind: str = "command"
    detail: str = ""


@dataclass(frozen=True)
class CommandExecution:
    """Result from a subprocess command."""

    returncode: int
    output: str
    duration_seconds: float


@dataclass(frozen=True)
class GateResult:
    """Serialized status for one release-readiness gate."""

    name: str
    status: str
    required: bool
    command: str
    duration_seconds: float
    detail: str


console = Console()


def _python_module(*args: str) -> Command:
    return (sys.executable, "-m", *args)


def command_to_string(command: Sequence[str]) -> str:
    """Render a command without shell-specific quoting."""
    return " ".join(command)


def classify_secret_state(value: str | None) -> str:
    """Classify an env var value without exposing it."""
    if value is None or not value.strip():
        return "missing"
    cleaned = value.strip().strip('"').strip("'")
    lowered = cleaned.lower()
    placeholder_prefixes = (
        "your_",
        "changeme",
        "change-me",
        "placeholder",
        "todo",
        "none",
        "null",
        "xxx",
        "test",
        "dummy",
        "example",
    )
    if (
        lowered.startswith(placeholder_prefixes)
        or lowered in {"...", "<tavily_api_key>", "<api_key>"}
        or "tavily" in lowered and "key" in lowered
        or lowered.startswith("<") and lowered.endswith(">")
    ):
        return "placeholder"
    return "usable"


def build_release_gates(
    *,
    provider: str = "deepseek",
    output_dir: str = "outputs/release_readiness",
    include_build: bool = True,
    include_benchmark: bool = True,
    include_mcp: bool = True,
    include_external_prereqs: bool = True,
) -> list[ReleaseGate]:
    """Build the ordered local release-readiness gate list."""
    gates = [
        ReleaseGate(
            "doctor",
            _python_module("SDYJ_Agents.cli.main", "doctor", "--provider", provider),
            detail="No-network provider/deployment preflight.",
        ),
        ReleaseGate(
            "pytest",
            _python_module("pytest"),
            detail="Full test suite.",
        ),
        ReleaseGate(
            "ruff",
            _python_module("ruff", "check", "SDYJ_Agents", "tests", "examples"),
            detail="Lint gate.",
        ),
    ]

    if include_benchmark:
        gates.extend(
            [
                ReleaseGate(
                    "external benchmark smoke",
                    _python_module(
                        "SDYJ_Agents.cli.main",
                        "benchmark",
                        "external",
                        "--suite",
                        "gaia",
                        "--source",
                        "local",
                        "--limit",
                        "3",
                        "--output-dir",
                        output_dir,
                        "--fail-under",
                        "1.0",
                    ),
                    detail="Synthetic GAIA-style harness/artifact smoke.",
                ),
                ReleaseGate(
                    "offline benchmark gate",
                    _python_module(
                        "SDYJ_Agents.cli.main",
                        "benchmark",
                        "run",
                        "--max-scenarios",
                        "1",
                        "--max-iterations",
                        "2",
                        "--fail-under",
                        "0.75",
                        "--determinism-repeats",
                        "2",
                        "--output-dir",
                        output_dir,
                    ),
                    detail="Deterministic local benchmark gate.",
                ),
            ]
        )

    if include_mcp:
        gates.extend(
            [
                ReleaseGate(
                    "MCP filesystem check",
                    (
                        sys.executable,
                        "examples/mcp_demos/mcp_filesystem_demo.py",
                        "--root",
                        ".",
                        "--check",
                    ),
                    detail="No-secret local MCP filesystem prerequisite check.",
                ),
                ReleaseGate(
                    "MCP GitHub check",
                    (
                        sys.executable,
                        "examples/mcp_demos/mcp_github_demo.py",
                        "--token-env",
                        "GITHUB_PERSONAL_ACCESS_TOKEN",
                        "--check",
                    ),
                    required=False,
                    kind="github_mcp",
                    detail="Blocked until GITHUB_PERSONAL_ACCESS_TOKEN is configured.",
                ),
            ]
        )

    if include_build:
        gates.extend(
            [
                ReleaseGate(
                    "build",
                    _python_module("build"),
                    detail="Build sdist and wheel.",
                ),
                ReleaseGate(
                    "twine check",
                    _python_module("twine", "check", "dist/*"),
                    kind="twine",
                    detail="Validate built distribution metadata.",
                ),
            ]
        )

    if include_external_prereqs:
        gates.extend(
            [
                ReleaseGate(
                    "Tavily live-search prerequisite",
                    required=False,
                    kind="tavily",
                    detail="Needed for DeepSeek + Tavily + arXiv live smoke.",
                ),
                ReleaseGate(
                    "Docker runtime prerequisite",
                    required=False,
                    kind="docker",
                    detail="Needed for docker build/run/compose smoke.",
                ),
            ]
        )

    return gates


def run_command(
    command: Command,
    cwd: Path,
    timeout_seconds: int,
    env: dict[str, str],
) -> CommandExecution:
    """Run a command and capture combined output."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout or ""
        return CommandExecution(completed.returncode, output, time.monotonic() - started)
    except FileNotFoundError as exc:
        return CommandExecution(127, str(exc), time.monotonic() - started)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return CommandExecution(124, f"Timed out after {timeout_seconds}s\n{output}", timeout_seconds)


def _tail_output(output: str, max_lines: int = 8) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _twine_command(repo_root: Path) -> Command:
    artifacts = sorted(str(path) for path in (repo_root / "dist").glob("*"))
    if not artifacts:
        return ()
    return _python_module("twine", "check", *artifacts)


def _evaluate_special_gate(gate: ReleaseGate, repo_root: Path, env: dict[str, str]) -> GateResult | None:
    if gate.kind == "github_mcp" and not env.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
        return GateResult(
            gate.name,
            "BLOCKED",
            gate.required,
            command_to_string(gate.command),
            0.0,
            "GITHUB_PERSONAL_ACCESS_TOKEN=missing; not running GitHub MCP list-tools.",
        )

    if gate.kind == "tavily":
        state = classify_secret_state(env.get("TAVILY_API_KEY"))
        status = "PASS" if state == "usable" else "BLOCKED"
        return GateResult(
            gate.name,
            status,
            gate.required,
            "",
            0.0,
            f"TAVILY_API_KEY={state}; live smoke is {'available' if state == 'usable' else 'blocked'}.",
        )

    if gate.kind == "docker":
        docker_path = shutil.which("docker")
        status = "PASS" if docker_path else "BLOCKED"
        detail = "docker found on PATH" if docker_path else "docker=missing; runtime smoke needs Docker host."
        return GateResult(gate.name, status, gate.required, "docker --version", 0.0, detail)

    if gate.kind == "twine":
        command = _twine_command(repo_root)
        if not command:
            return GateResult(
                gate.name,
                "FAIL",
                gate.required,
                command_to_string(gate.command),
                0.0,
                "No dist artifacts found; run the build gate first.",
            )
        return None

    return None


def collect_release_readiness(
    *,
    provider: str = "deepseek",
    output_dir: str = "outputs/release_readiness",
    include_build: bool = True,
    include_benchmark: bool = True,
    include_mcp: bool = True,
    include_external_prereqs: bool = True,
    dry_run: bool = False,
    timeout_seconds: int = 900,
    repo_root: Path | None = None,
    command_runner: CommandRunner = run_command,
) -> list[GateResult]:
    """Run release-readiness gates and return structured results."""
    root = (repo_root or Path.cwd()).resolve()
    load_dotenv(root / ".env")
    env = os.environ.copy()
    gates = build_release_gates(
        provider=provider,
        output_dir=output_dir,
        include_build=include_build,
        include_benchmark=include_benchmark,
        include_mcp=include_mcp,
        include_external_prereqs=include_external_prereqs,
    )

    results: list[GateResult] = []
    for gate in gates:
        if dry_run:
            results.append(
                GateResult(
                    gate.name,
                    "SKIP",
                    gate.required,
                    command_to_string(gate.command),
                    0.0,
                    f"dry-run: {gate.detail}",
                )
            )
            continue

        special = _evaluate_special_gate(gate, root, env)
        if special is not None:
            results.append(special)
            continue

        command = _twine_command(root) if gate.kind == "twine" else gate.command
        execution = command_runner(command, root, timeout_seconds, env)
        if execution.returncode == 0:
            detail = gate.detail
            tail = _tail_output(execution.output, max_lines=2)
            if tail:
                detail = f"{detail} Last output: {tail}"
            results.append(
                GateResult(
                    gate.name,
                    "PASS",
                    gate.required,
                    command_to_string(command),
                    execution.duration_seconds,
                    detail,
                )
            )
        else:
            results.append(
                GateResult(
                    gate.name,
                    "FAIL",
                    gate.required,
                    command_to_string(command),
                    execution.duration_seconds,
                    _tail_output(execution.output) or gate.detail,
                )
            )

    return results


def print_release_results(results: Sequence[GateResult]) -> None:
    table = Table(title="SDYJ Release Readiness")
    table.add_column("Gate")
    table.add_column("Status")
    table.add_column("Required")
    table.add_column("Seconds", justify="right")
    table.add_column("Detail")

    for result in results:
        style = {
            "PASS": "green",
            "FAIL": "red",
            "BLOCKED": "yellow",
            "SKIP": "cyan",
        }.get(result.status, "white")
        table.add_row(
            result.name,
            f"[{style}]{result.status}[/{style}]",
            "yes" if result.required else "no",
            f"{result.duration_seconds:.1f}",
            result.detail,
        )

    console.print(table)
    failures = [result for result in results if result.required and result.status == "FAIL"]
    blockers = [result for result in results if result.status == "BLOCKED"]
    if failures:
        console.print(f"[red][FAIL] {len(failures)} required release gate(s) failed.[/red]")
    elif blockers:
        console.print(
            f"[yellow][WARN] Required gates passed, with {len(blockers)} external blocker(s) still open.[/yellow]"
        )
    else:
        console.print("[green][OK] Release-readiness gates passed.[/green]")


def release_readiness_exit_code(results: Sequence[GateResult]) -> int:
    return 1 if any(result.required and result.status == "FAIL" for result in results) else 0


def create_release_check_parser(
    *,
    prog: str = "sdyj release-check",
    default_provider: str = "deepseek",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run local release-readiness gates without publishing or calling live APIs.",
    )
    parser.add_argument("--provider", default=default_provider, choices=["deepseek", "openai", "claude", "gemini"])
    parser.add_argument(
        "--output-dir",
        default="outputs/release_readiness",
        help="Output directory used by benchmark smoke gates.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip build and twine metadata checks.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark smoke gates.")
    parser.add_argument("--skip-mcp", action="store_true", help="Skip MCP prerequisite checks.")
    parser.add_argument(
        "--skip-external-prereqs",
        action="store_true",
        help="Skip Tavily/Docker optional blocker reporting.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned gates without executing commands.")
    parser.add_argument("--json", action="store_true", help="Print JSON results instead of a rich table.")
    parser.add_argument("--timeout", type=int, default=900, help="Per-command timeout in seconds.")
    return parser


def run_release_readiness_from_args(args: argparse.Namespace) -> int:
    results = collect_release_readiness(
        provider=args.provider,
        output_dir=args.output_dir,
        include_build=not args.skip_build,
        include_benchmark=not args.skip_benchmark,
        include_mcp=not args.skip_mcp,
        include_external_prereqs=not args.skip_external_prereqs,
        dry_run=args.dry_run,
        timeout_seconds=args.timeout,
    )
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_release_results(results)
    return release_readiness_exit_code(results)


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_release_check_parser(prog="python scripts/release_readiness.py")
    args = parser.parse_args(argv)
    return run_release_readiness_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
