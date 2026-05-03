---
CURRENT_TIME: {{ CURRENT_TIME }}
---

Analyze the following user input in the context of a research workflow:

<user_input>
{{ user_input }}
</user_input>

<current_workflow_step>
{{ current_step }}
</current_workflow_step>

Determine the user's intent. Is the user:
1. Approving the plan (respond with "APPROVE")
2. Requesting modifications to the plan (respond with "MODIFY")
3. Rejecting the plan and wanting to start over (respond with "REJECT")
4. Asking a question (respond with "QUESTION")

<decision_rules>
- If the user says yes/ok/approve/start, return APPROVE.
- If the user asks to change scope, sources, task order, or wording, return MODIFY.
- If the user wants to abandon the plan and restart, return REJECT.
- If the user asks for clarification without approving or modifying, return QUESTION.
</decision_rules>

Respond with only one word: APPROVE, MODIFY, REJECT, or QUESTION.
