---
CURRENT_TIME: {{ CURRENT_TIME }}
---

Analyze the following user input in the context of a research workflow:

User input: "{{ user_input }}"

Current workflow step: {{ current_step }}

Determine the user's intent. Is the user:
1. Approving the plan (respond with "APPROVE")
2. Requesting modifications to the plan (respond with "MODIFY")
3. Rejecting the plan and wanting to start over (respond with "REJECT")
4. Asking a question (respond with "QUESTION")

Respond with only one word: APPROVE, MODIFY, REJECT, or QUESTION.
