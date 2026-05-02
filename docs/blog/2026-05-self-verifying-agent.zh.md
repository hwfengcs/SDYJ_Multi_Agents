# 你的多 Agent 框架需要的不是更好的 Prompt，而是 Trace、Replay 和 Verifier

*发布于 2026-05-02 · 标签：ai-agent、langgraph、observability、agent-evaluation*

很多开源 agent 框架都有一个挺让人头疼的模式。README 里展示了一条 happy path：定义 agent、连上工具、跑一个 query、惊叹于生成的 markdown 报告。然后你把它推给真实用户。用户提了一个真实问题。Agent 跑了两分钟，给出三段语气笃定的回答，引用的一个 URL 是 404 —— 而你完全不知道哪一步出错了。

于是你去调 prompt。加一句"务必引用真实来源！"。再跑一次。又错了，错的方式还略有不同。再加一句。Prompt 越来越长。行为越来越不可预测，而不是更可预测。

**这不是 prompt 问题，是评测问题。** 走出去的唯一办法是把每次 agent 运行变得**可控、可观测、可回放、可基准测试** —— 把它当成头等的工程问题，而不是事后补丁。

这就是 [SDYJ Multi Agents](https://github.com/hwfengcs/SDYJ_Multi_Agents) v0.6 的核心思路。SDYJ 是我在做的基于 LangGraph 的多智能体研究框架。这篇文章会过一遍让 agent 运营真正可治理的四根柱子，以及 v0.6.0a1 alpha 已经在前三根柱子上落地的内容。

## 典型 "agent demo" 哪里崩

先把生产环境真实会遇到的失败模式列出来：

1. **编造引用**。模型填一个看起来合理、实际 404 的 URL。没有质量 gate。
2. **工具失败用户感知不到**。Tavily 超时，agent 默默继续，报告还宣称"基于充分的网页检索"。如果有 trace，也就是几行 print。
3. **没法复现**。用户说"3 小时前 agent 给了个错答案"，你手上一无所有。LLM 是非确定的。网页内容也变了。
4. **成本失控**。你写 `temperature=0.7, max_tokens=4000`，配一个 5 次迭代的 loop。20 个查询后 DeepSeek 账单是 $40，但你不知道是哪一步贵。
5. **质量回归**。你改了一个 prompt 推上 main。三天后有人发现引用率掉了 30%。你不知道是你那 7 次 prompt 改动里哪一次造成的。

每一个都能解决。但都不是靠改 prompt 解决的。

## 第一根柱子：把所有事件都记下来

Agent 运行里每个有意义的步骤都该产生一条结构化事件。不是 log line —— 是带有时间戳、parent_id、延迟、状态、输入/输出 hash（用于回放）的强类型事件。

在 SDYJ 里这是 v2 trace schema。每次运行都生成一份 bundle：

```text
outputs/runs/<run-id>/
  trace.json        # 完整结构化 trace
  events.jsonl      # 一行一事件，方便 grep
  state.final.json  # 完成时的 workflow state
  report.md         # 实际输出的报告
```

trace 记录：

- 每个节点的 enter/exit，含延迟和 metadata。
- 每次 LLM 调用，含 prompt hash、response hash、token 数、延迟、成本（成本见第三根柱子）。
- 每次工具调用，含 source、query、结果数、延迟、错误。
- 每次路由决策，含原因和驱动决策的 metadata。

这大致等同于 LangSmith / Langfuse / Weave 提供的能力 —— 但是免费、本地、内置。不依赖外部 SaaS、不需要额外注册。trace 是其他一切的 ground truth。

## 第二根柱子：deterministic replay

光有 trace 还不够。另一半是要能 **不调用真实 API 就把出错的 query 重跑一遍**。

SDYJ 的 replay 从源 trace 读取已记录的 LLM 响应和工具结果，然后用 mock 的 I/O 按调用顺序替换重跑：

```bash
sdyj replay <run-id>
```

replay 会产生一份新的 trace bundle。你可以 `diff-runs` 对比原始 run 和 replay run，确认它们走了同一条路径。如果走不同了 —— 比如你把某个阈值收紧了导致路由变了 —— diff 会让回归显而易见。

这件事的价值有两个：

1. **bug 调查零成本**。用户报问题，你本地 replay，不消耗 API。可以用 `sdyj inspect-run --timeline` 一步步看事件。可以改个 prompt 然后 replay 看看新 prompt 能不能修。
2. **CI 里加 determinism gate**。同一个 scenario `--determinism-repeats 2` 跑两次，如果记录的 trace 不能重现，说明 workflow 里有你没意识到的非确定或状态依赖。

## 第三根柱子：trace 里记录每次调用的成本

成本失控的解法是一条规则：**每次 LLM 调用在它发生时就记下美元成本**。

在 v0.6.0a1 里我们把这件事做到端到端：

- 每个 provider 包装层（OpenAI、Claude、DeepSeek、Gemini）都在每次调用后暴露 `last_usage`。
- 新增的 `SDYJ_Agents/utils/cost.py` 模块提供一个透明的 `PRICING_TABLE`，就是个硬编码的 `{(provider, model): (input_per_million, output_per_million)}` 字典 —— 不藏猫腻、不调用三方服务。价格变了就改这张表，发版本时写进 release note。
- trace 的 `InstrumentedLLM` 把 `prompt_tokens_actual`、`completion_tokens_actual`、`cost_usd` 写进每条 `llm_calls`。
- `finalize_trace` 累加到 `trace.metrics`：总 token、总美元、单独的 priced/unpriced 调用计数。
- CLI `sdyj inspect-run` 展示一张 per-call 成本表。

关键设计：当 (provider, model) 不在 `PRICING_TABLE` 里时，我们返回 `None`，不是 `0.00`。`0.00` 是**谎言**。`—` + "请把这条加到 PRICING_TABLE 里" 才是诚实的。

```text
                LLM Calls (per call cost estimates)
+-----------------------------------------------------------------+
| Call | Model            | Prompt tok | Completion tok | Cost    |
|------+------------------+------------+----------------+---------|
| L1   | deepseek-v4-flash|        450 |            120 | $0.00016|
| L2   | deepseek-v4-flash|       2104 |            812 | $0.00104|
| L3   | gpt-4o           |        980 |            340 | $0.00585|
+-----------------------------------------------------------------+
```

现在你看一次 5 迭代的 research run，能精确看到**哪一次调用是成本大头**。通常是 synthesized analysis 那个调用，不是 planner。你有数据可以做决策，而不是凭感觉。

## 第四根柱子：verifier loop（进行中）

这是我最期待的一块，正在 v0.6 分支上做。

大多数 agent 框架把 report 当成最终输出。SDYJ v0.6 加了第 5 个 agent —— **Verifier** —— 它会重新读 report、对照收集到的证据，问四个问题：

1. **claim_evidence_alignment**：每条 key finding 是否真的能追溯到某个 `E1/E2/...` 证据项？还是模型在"总结"凭空冒出来的东西？
2. **citation_completeness**：所有证据是否都被使用？还是收集了 12 个来源只引用了 3 个？
3. **factual_consistency**：报告各章节之间是否自相矛盾？
4. **plan_coverage**：报告是否真的覆盖了 Planner 承诺的子任务，还是跑偏了？

Verifier 输出结构化 JSON，含 `should_revise` 标志和 revision hints。如果 `should_revise` 为 true，workflow 回到 Rapporteur 带着 hints 重做。修订循环有硬上限（默认 2 次），不会无限循环。

这是经典的 ReAct/Reflexion 风格设计 —— 创新点不在算法，而在于 **verifier 是图里的一个 traced 节点**，跟其他 agent 一样。它的批评在 trace 里。它的修订在 trace 里。可以 replay。可以用 `--no-verify` 做 A/B 测量增益。

这就是怎么把"自验证"从营销词变成你能写在 benchmark 计分板上的东西。

## 接下来

完整的 v0.6 计划在 [docs/release-notes/v0.6.md](https://github.com/hwfengcs/SDYJ_Multi_Agents/blob/main/docs/release-notes/v0.6.md)。未来 6 周的重点：

- Verifier agent + revise loop（上面的第四根柱子）。
- Reflexive Researcher：当一批查询返回空或相关性低时，agent 重写查询重试，而不是直接放弃。
- Plan refinement：执行 N 个 task 后 Planner 看到收集的证据，可以删除/添加剩余 task。
- task 内并行工具调用（asyncio.gather + 并发限制）。
- **GAIA Level 1** 和 **AssistantBench** 上的公开 benchmark 跑分，含 v0.5-vs-v0.6 ablation。这是决定 verifier loop 真假的数据。
- Streamlit Web UI（已经在 v0.6.0a1 出了 MVP）+ Hugging Face Spaces 在线 demo。
- 用官方 `mcp` Python SDK 真正集成 MCP，替换当前的 HTTP 占位实现。

如果你在做 agent，并且经历过本文开头那些失败模式，我特别想听你的反馈 —— 尤其是 **trace schema 在你的场景里还差什么**。仓库在 [github.com/hwfengcs/SDYJ_Multi_Agents](https://github.com/hwfengcs/SDYJ_Multi_Agents)，欢迎提 PR/Issue。

如果想跟进 v0.6 进度，欢迎 star 仓库；v0.6 stable 计划 2026 年 6 月底发布，附带 GAIA 跑分。
