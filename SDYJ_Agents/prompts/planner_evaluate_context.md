---
CURRENT_TIME: {{ CURRENT_TIME }}
---

Evaluate whether the gathered research context is sufficient to answer the query.

<query>
{{ query }}
</query>

<research_goal>
{{ research_goal }}
</research_goal>

<completion_criteria>
{{ completion_criteria }}
</completion_criteria>

<progress>
- research_batches_gathered: {{ results_count }}
- current_iteration: {{ current_iteration }}
- max_iterations: {{ max_iterations }}
</progress>

<decision_rules>
- Respond YES if the gathered context is enough to produce a useful, evidence-grounded report.
- Respond NO if major planned dimensions are still missing and there is remaining iteration budget.
- If max_iterations is reached or nearly reached, prefer YES unless there are zero results.
</decision_rules>

Respond with only "YES" or "NO".
