# Explanation of the columns in a *result*.csv file:

Columns in the result file:
*[idx, goal, task_idx, status, plan_len, planning_time, planning_objects, object_reducer, planning_node, agent_state]*

Last row meanings:
- idx: Not filled
- goal: Success rate (Longest streak of continuously solved problems / Length of whole goal sequence)
- task_idx: Success rate (Completed Problems / Length of whole goal sequence)
- status: Success rate (Successes / (Failures + Successes))
- plan_len: Total plan length
- planning_time: Total planning time
- planning_objects: "Effective" planning time
- object_reducer: "Wasted" planning time
- planning_node: Not filled
- agent_state: Not filled