# Internship Story

Use this page as a concise narrative when discussing the project in an AI agent
algorithm engineer internship interview.

## One-Sentence Pitch

I built a LangGraph-based multi-agent deep research assistant that combines
human-in-the-loop planning, multi-source retrieval, provider-agnostic LLM
integration, evidence-grounded reporting, JSON tracing, and reproducible
evaluation.

## Technical Highlights

- Designed a controllable state machine instead of a single monolithic prompt.
- Split responsibilities across Coordinator, Planner, Researcher, and Rapporteur.
- Added a human approval gate before expensive or broad retrieval runs.
- Normalized Tavily, arXiv, and MCP-style results into one retrieval shape.
- Normalized retrieved sources into deduplicated `E1/E2/...` evidence items.
- Built a citation-integrity pipeline: relevance-budgeted prompt evidence, instructed inline `[E#]` citations with heuristic backfill, and post-generation validation that strips fabricated ids.
- Layered fault tolerance: transient-error LLM retry with backoff, per-task and per-section graceful degradation, recorded `degraded_events`, and crash-path partial-state persistence with nonzero exit codes.
- Durable per-run SQLite checkpoints with a `resume` command that continues from the last super-step, including pending human review.
- Persisted run traces with node latency, LLM calls (incl. retry counts), tool calls, errors, and report metrics; replay matches recorded responses by prompt hash with sequential fallback.
- Added hard evaluation scenarios (incl. LLM-failure injection) with offline canned evidence, dual-track LLM-as-judge faithfulness scoring, and live DeepSeek evaluation.
- Added tests and CI so core behavior can be verified without real API keys.

## Interview Talking Points

1. **Workflow design**
   - Why the graph has an explicit planning node and approval node.
   - How `max_iterations` limits agent loops (and `recursion_limit` scales with it).
   - How simple requests avoid unnecessary research flow.

2. **Tool-use reliability**
   - Why tools return normalized result dictionaries and never raise.
   - How empty results and tool errors are represented and traced.
   - How evidence IDs connect final claims back to tool, query, URL, and metadata.

3. **Q: 长上下文导致的幻觉怎么处理？**
   - 分三层：进 prompt 的证据按相关性+字符预算裁剪（不是按插入顺序截断）；
     prompt 强制要求 `[E#]` 行内引用、禁止编造编号、证据不足要明说，启发式
     关键词补引只作兜底并单独计数；交付前 `validate_citations` 剔除一切
     不存在的引用编号。
   - 指标端反作弊：引用覆盖率只统计正文（参考资料列表天然含全部编号，计入
     就是自我实现指标）；编造 `[E99]` 会拉低 `citation_validity_rate` 而不是
     推高 coverage。
   - 一个真实教训：旧的 FakeEvalLLM 靠匹配模板文案分发 canned 响应，模板里
     `**必须**` 的加粗标记让 organize_info 分支从未命中、离线基准一直在测
     fallback 路径。改成每个模板嵌入稳定 `[PROMPT_ID]` 标记 + 契约测试
     （tests/test_prompt_markers.py）后，这类漂移会直接红。

4. **Q: 复杂工作流中状态管理和容错降级怎么做？**
   - 状态：LangGraph 单一共享 state + checkpointer；research 默认落
     per-run SQLite checkpoint，`resume <run-id>` 能从崩溃点或待审批中断
     继续。一个踩过的坑：图用的是 `StateGraph(dict)`，整个 state 是单一
     last-value channel——`update_state` 写部分字段会把整个 state 替换掉、
     丢光其余键，恢复和审批写回必须携带完整 state（测试第一轮就抓住了）。
   - 容错分层：工具层错误字典隔离 + 超时；LLM 层瞬时错误（timeout/429/5xx）
     指数退避重试、鉴权类错误 fail fast，重试次数记录在同一条 llm_call 里
     （调用数不变，replay 才稳定）；节点/章节层失败降级为占位输出并记录
     `degraded_events`；条件边里的 LLM 调用失败默认去写报告而不是炸掉图；
     CLI 崩溃保存 `state.partial.json` 并返回退出码 4。
   - 降级永不静默：state、trace 事件流、报告指标三处都有记录，benchmark
     场景 `llm_failure_recovery_hard` 直接 gate `retries_total>=1` 和
     `degraded_event_count>=1`。

5. **Q: 评估这个 Agent 的准确率，Evals 体系怎么建？**
   - 三层：pytest 单测（fake LLM/工具，无 key）；离线 benchmark（canned
     evidence + marker 分发的 canned 响应，确定性可作 CI gate，含故障注入
     场景）；`--live` 冒烟（真模型 + canned evidence 隔离检索漂移）。
   - 指标设计原则：先堵自我实现的口子（正文引用统计、有效性校验），再加
     语义层——LLM-as-judge 对抽样引用论断做 claim↔evidence 支持度评分
     （supported/partial/unsupported，单次批量调用）。
   - judge 双轨：离线用 canned verdict 保证 CI 确定性，live 用真模型；
     faithfulness 作为独立阈值维度，故意不并进 overall_score——固定权重
     的合成分可以被"刷分"，独立 gate 不行。
   - 工程闭环：阈值 gate（exit 3）、`--fail-under`、`--compare-summary`
     回归对比、`--determinism-repeats` 指纹检查、trace completeness。
   - 为什么 `sdyj eval --live --provider deepseek` 能隔离模型行为：evidence
     固定，唯一变量是模型推理质量。

## Resume Bullet

Built a LangGraph-based multi-agent deep research system with human-in-the-loop
planning, multi-source retrieval, evidence-grounded reports, JSON run tracing,
DeepSeek-backed evaluation scenarios, tests, and CI.
