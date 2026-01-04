# How It Works?

With so many LLM calls, it may be helpful to visualize the sequence of operations and LLM interactions that occur when using agents. Below is a detailed breakdown of the key steps, iterations, and LLM calls involved in the ReActAgent's workflow with `max_iterations=3` for illustration. Some of the steps are common for CodeActAgent as well, except for the code execution parts.

```
INITIALIZATION PHASE
├─ Call 1: planner.create_plan()
│  └─ ku.call_llm() → Creates initial plan from task description
│     [AgentPlan response format]
│
└─ (Plan displayed to user, added to message history)


ITERATION 1 (idx=0)
├─ Call 2: _think() → _record_thought(ReActChatMessage)
│  └─ _chat(response_format=ReActChatMessage)
│     └─ ku.call_llm() → Agent thinks about next action
│        [ReActChatMessage response format]
│        [With retry logic: MAX_RESPONSE_PARSING_ATTEMPTS = 3]
│
├─ _act() → Executes tool or returns final answer
│  └─ (No LLM call if action taken; potential LLM call only if parsing fails)
│
├─ Call 3: planner.update_plan()
│  └─ ku.call_llm() → Updates plan based on thought + observation
│     [AgentPlan response format]
│
└─ Call 4: observer.observe()
   └─ ku.call_llm() → Observes agent state for loops/issues
      [ObserverResponse response format]
      [Only if: iteration > 1 AND (iteration - last_correction_iteration) >= threshold]
      [First observation typically skipped since iteration=1 and threshold default=3]


ITERATION 2 (idx=1)
├─ Call 5: _think()
│  └─ _chat(response_format=ReActChatMessage)
│     └─ ku.call_llm()
│
├─ _act() → Executes tool or returns final answer
│
├─ Call 6: planner.update_plan()
│  └─ ku.call_llm()
│
└─ Call 7: observer.observe()
   └─ ku.call_llm()
      [Now eligible since iteration=2, but threshold check: 2 - 0 < 3, so skipped]


ITERATION 3 (idx=2)
├─ Call 8: _think()
│  └─ _chat(response_format=ReActChatMessage)
│     └─ ku.call_llm()
│
├─ _act() → Executes tool or returns final answer
│
├─ Call 9: planner.update_plan()
│  └─ ku.call_llm()
│
└─ Call 10: observer.observe()
   └─ ku.call_llm()
      [Eligible since iteration=3, threshold check: 3 - 0 >= 3, so CALLED]


POST-ITERATION PHASE (if final_answer_found = True)
└─ Call 11: planner.update_plan() [Optional, only if final answer found]
   └─ ku.call_llm()
      [Marks completion steps in plan]


FAILURE RECOVERY PHASE (if max_iterations reached without answer)
└─ Call 12: salvage_response()
   └─ ku.call_llm() → Summarizes progress for incomplete task
      [If summarize_progress_on_failure=True]
```
