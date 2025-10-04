---
CURRENT_TIME: {{ CURRENT_TIME }}
---

Evaluate whether the gathered research context is sufficient to answer the query.

Query: "{{ query }}"

Research Plan Goal: {{ research_goal }}
Completion Criteria: {{ completion_criteria }}

Number of research results gathered: {{ results_count }}
Current iteration: {{ current_iteration }}/{{ max_iterations }}

Based on the above information, is the context sufficient to generate a comprehensive report?

Respond with only "YES" or "NO".
