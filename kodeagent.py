"""
A minimalistic approach to building AI agents.
Implements ReAct and CodeActAgent. Supports multi-agent via SupervisorAgent.
"""
import ast
import asyncio
import inspect
import json
import logging
import os
import re
import shutil
import subprocess as sp
import sys
import tempfile
import textwrap
import uuid
import warnings
from abc import ABC, abstractmethod
from functools import wraps
from json import JSONDecodeError
from typing import (
    AsyncIterator,
    Literal,
    Optional,
    Callable,
    Any,
    Type,
    TypedDict,
    Union,
)

import json_repair
import litellm
import pydantic as pyd
import rich
from dotenv import load_dotenv

import kutils as ku


load_dotenv()

warnings.simplefilter('once', UserWarning)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Get a logger for the current module
logger = logging.getLogger('KodeAgent')

# litellm._turn_on_debug()
litellm.success_callback = ['langfuse']
litellm.failure_callback = ['langfuse']


REACT_PROMPT = '''
You are an expert assistant, helpful and polite, who can solve any task using tool calls. 
Given a task, you think about how to solve it, suggest a tool to use to solve the current step,
observe the outcome of the current action, and then think again.
You practise self-questioning, self-reasoning, self-reflection, and self-healing
to overcome any obstacles and reach your goal.


## Task

The task description is as follows:
{task}

(Optional) input file paths/URLs associated with this task are as follows:
{task_files}


## Tools

The following tools are available to you:
{tool_names}

You can use one or more of these tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.


## Output Format

Please answer in the same language as the question and use the following format:

Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of aforementioned tool names) if using a tool.
Args: the input arguments to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})

ALWAYS start with a Thought.

NEVER surround your response with markdown code markers.
You may use code markers within your response if you need to.
Please use a valid JSON format for the Args. E.g., do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the tool will respond in the following format:

Observation: tool response


You should keep repeating the above format (Thought-Action-Observation cycle) till you have enough
information to answer the question without using any more tools.
At that point, you MUST respond in one of the following two formats:

Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
Successful: True

Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
Successful: False


The `Successful` flag is set to `False` in the second case since the task was failed to be solved.
This flag should be always False until you reach the final step and decide that the task is complete.
Note: if an action fails, the error message will be captured in `Observation`.
Frame your next `Thought` in a way so that it can mitigate the previous error and take correct action.


## Example Conversations

Below, a few sample conversations using notional tools are provided for your reference.
Please study the patterns carefully.

---
[Sample task: Generate an image of the oldest person in this document.]

Thought: I will begin by identifying the oldest person mentioned in the document. I will use the `document_qa` tool for this purpose.
Action: document_qa
Args: {{"document": "document.pdf", "question": "Who is the oldest person mentioned?"}}
Observation: The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland.

Thought: Based on document search, I have identified John Doe, aged 55, as the oldest person. He lives in Newfoundland, Canada. Now, I'll use the `image_generator` tool to generate his portrait.  
Action: image_generator
Args: {{"prompt": "A portrait of John Doe, a 55-year-old man living in Canada."}}
Observation: image.png

Thought: Based on the given document, I have identified John Doe (55) as the oldest person. I have also generated his portrait and saved it in the `image.png` file.
Answer: image.png
Successful: True

---
[Sample task: What is the result of the following operation: 5 + 3 + 1294.678?]

Thought: This is an arithmetic problem. I will use the `calculator` tool to compute the sum.
Action: calculator
Args: {{"expression": "5 + 3 + 1294.678"}}
Observation: 1302.678

Thought: Using the `calculator` tool, the sum of the given numbers is 1302.678.
Answer: 1302.678
Successful: True

---
[Sample task: Generate a video of the moon.]

Thought: The user has asked to generate a video of the moon. Unfortunately, I do not have any tool that can generate a video. So, I can't solve this task.
Answer: Unfortunately, I lack the ability to solve this task at this moment. May I help you with something else?
Successful: False


## Plan

Here's a general plan that someone who may or may not have your tools might try to solve this task.
You can refer to it but adapt as necessary, e.g., add/edit/combine/skip steps:
(ignore if plan is not available)
{plan}

Based on the current state of the Thought-Action-Observation, you will identify what steps from
the plan have already been achieved and what needs to be done next, thus frame your `Thought`.


## Additional Instructions:
- Call a tool only when needed, e.g., do not call the search agent if you do not need to search information.
- Do not use non-existent tools. Only use a tool listed earlier. 
- Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.
- Never re-do a tool call that you previously did with the exact same parameters.
- Do your best! Don't give up! You're in charge of solving the task, not providing directions to solve it.


## Current Interaction

Below is the history of the interaction so far, showing the interleaving "Thought," "Action," and "Observation" steps. (Initially empty.)
{history}
'''

REACT_CONTEXTUAL_PROMPT = '''
You are an expert assistant, helpful and polite, who can solve any task using tool calls. 
Given a task, you think about how to solve it, suggest a tool to use to solve the current step,
observe the outcome of the current action, and then think again.
You practise self-questioning, self-reasoning, self-reflection, and self-healing
to overcome any obstacles and reach your goal.


## Task

The task description is as follows:
{task}

(Optional) input file paths/URLs associated with this task are as follows:
{task_files}


## Tools

The following tools are available to you:
{tool_names}

You can use one or more of these tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.


## Output Format

Please answer in the same language as the question and use the following format:

Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of aforementioned tool names) if using a tool.
Args: the input arguments to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})

ALWAYS start with a Thought.

NEVER surround your response with markdown code markers.
You may use code markers within your response if you need to.
Please use a valid JSON format for the Args. E.g., do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the tool will respond in the following format:

Observation: tool response


You should keep repeating the above format (Thought-Action-Observation cycle) till you have enough
information to answer the question without using any more tools.
At that point, you MUST respond in one of the following two formats:

Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
Successful: True

Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
Successful: False


The `Successful` flag is set to `False` in the second case since the task was failed to be solved.
This flag should be always False until you reach the final step and decide that the task is complete.
Note: if an action fails, the error message will be captured in `Observation`.
Frame your next `Thought` in a way so that it can mitigate the previous error and take correct action.


## Example Conversations

Below, a few sample conversations using notional tools are provided for your reference.
Please study the patterns carefully.

---
[Sample task: Generate an image of the oldest person in this document.]

Thought: I will begin by identifying the oldest person mentioned in the document. I will use the `document_qa` tool for this purpose.
Action: document_qa
Args: {{"document": "document.pdf", "question": "Who is the oldest person mentioned?"}}
Observation: The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland.

Thought: Based on document search, I have identified John Doe, aged 55, as the oldest person. He lives in Newfoundland, Canada. Now, I'll use the `image_generator` tool to generate his portrait.  
Action: image_generator
Args: {{"prompt": "A portrait of John Doe, a 55-year-old man living in Canada."}}
Observation: image.png

Thought: Based on the given document, I have identified John Doe (55) as the oldest person. I have also generated his portrait and saved it in the `image.png` file.
Answer: image.png
Successful: True

---
[Sample task: What is the result of the following operation: 5 + 3 + 1294.678?]

Thought: This is an arithmetic problem. I will use the `calculator` tool to compute the sum.
Action: calculator
Args: {{"expression": "5 + 3 + 1294.678"}}
Observation: 1302.678

Thought: Using the `calculator` tool, the sum of the given numbers is 1302.678.
Answer: 1302.678
Successful: True

---
[Sample task: Generate a video of the moon.]

Thought: The user has asked to generate a video of the moon. Unfortunately, I do not have any tool that can generate a video. So, I can't solve this task.
Answer: Unfortunately, I lack the ability to solve this task at this moment. May I help you with something else?
Successful: False


## Plan

Here's a general plan that someone who may or may not have your tools might try to solve this task.
You can refer to it but adapt as necessary, e.g., add/edit/combine/skip steps:
(ignore if plan is not available)
{plan}

Based on the current state of the Thought-Action-Observation, you will identify what steps from
the plan have already been achieved and what needs to be done next, thus frame your `Thought`.


## Additional Instructions:
- Call a tool only when needed, e.g., do not call the search agent if you do not need to search information.
- Do not use non-existent tools. Only use a tool listed earlier. 
- Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.
- Never re-do a tool call that you previously did with the exact same parameters.
- Do your best! Don't give up! You're in charge of solving the task, not providing directions to solve it.


## Context of Current Interaction

Below is a summary of what has been done so far for the current task.
It includes what was successful, what failed, and what is pending.
Based on this context, you should plan your next step to solve the task without repeating past actions.

{history_summary}
'''

CODE_ACT_AGENT_PROMPT = '''
<system>
You are an expert, helpful agent designed to solve complex tasks iteratively using available tools and Python code.
You have powerful multimodal capabilities, which allow you to analyze images directly.
</system>

<core_principles>
1. Iterative Problem Solving: Propose a step (`Thought`), execute code/tool use (`Code`),
    observe the outcome (`Observation`), then refine until the task is solved.
2. Self-Correction and Adaptive Reasoning: Analyze each `Observation`. If an error occurs, diagnose and adapt.
    Never repeat failed attempts without modification.
3. State Tracking and Task Progression: Maintain awareness of completed sub-tasks.
    Before writing new `Code`, explicitly refer to prior results and reuse available computations.
4. Loop Avoidance and Optimal Execution: Ensure each iteration meaningfully advances the task.
    Avoid redundant execution of already completed thought/code and move forward.
    Never call the same tool with the same arguments twice.
5. Innate Visual Intelligence: If a task involves an image, use your inherent visual capabilities to analyze it. 
    Do not use a tool for image analysis unless a specific, complex, non-standard operation is required.
</core_principles>

<task>
<task_description>{task}</task_description>
<input_files>{task_files}</input_files>
</task>

<tools>
The following *specialized* tools are available for your use:
<tool_list>
{tool_names}
</tool_list>

You are responsible for writing Python code to use these tools and only the following standard libraries:
<authorized_imports>
{authorized_imports}
</authorized_imports>
Do NOT import the provided tool names.
</tools>

<plan>
Optionally, here's a TODO list of items -- general plan to follow to solve the task (ignore if unavailable).
Align your `Thought` and `Code` with the steps from the plan, marking achieved steps mentally.
Completed tasks are marked with `[x]`; pending tasks as `[ ]`.
<todo_list>
{plan}
</todo_list>
</plan>

<output_format>
Adhere strictly to the following `Thought`-`Code`-`Observation` cycle.

<mental_check>
Before writing any `Thought`, perform a mental check:
1. Have I completed all the sub-tasks in the plan?
2. Do I have all the necessary information to provide the final answer without any further tool calls?
If both are true, skip directly to the `Thought` for the final `Answer`.
</mental_check>

<example_cycle>
Thought: Based on the current task status, what is the next logical step to take?
Code: ```py
# Write your Python code here
print(useful_result_information_found)
```
</example_cycle>

In `Thought`, based on the current task status and the `Observation` from the previous step, 
describe precisely your next action (code) and explaining why this specific `Code` block is necessary now. 
If you see a loop, explain how you will break it.

Use `print()` for any result/useful information you want to observe based on code execution. 
When necessary, you can apply your innate capabilities (e.g., summarization, translation, 
static code analysis, image analysis, and so on) on what you see under `Observation`. 
Avoid printing trivial text.

Every iteration of the cycle must move you closer to completing the task.
Keep track of the steps already completed and pending. 
Your `Thought` for each turn must focus on what needs to be done next to *advance* the task.

Repeat this cycle until you have enough information to provide a final answer. 
For the final answer, use one of these formats:

<final_answer>
Thought: I have enough information to answer. I will use the user's language.
Answer: [Your answer in the user's language]
Successful: True
</final_answer>

<failure_state>
Thought: I cannot answer the question with the provided tools/information.
Answer: [Your explanation in the user's language]
Successful: False
</failure_state>

The Successful:  flag should only be `True` for a completed task.
Craft your final answer by carefully reading the instructions from the task.
</output_format>

<examples>
<example>
What color is the sky in this picture? (Image: sky.jpg)


Thought: The user has provided an image file as input and wants to know the color of the sky. I can analyze the image directly using my innate vision capabilities. The image shows a sky. I can see its color is blue.
Answer: The sky in the picture is blue.
Successful: True
</example>

<example>
Which city has the highest population: Guangzhou or Shanghai?

Thought: The current language of the user is: English. I need to get the populations for both cities and compare them: I will use the tool `search` for this purpose. Since search results are important for this task, I'll print them.
Code: ```py
for city in ['Guangzhou', 'Shanghai']:
print(f'Population {{city}}:', search(f'{{city}} population'))
```
Observation: Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: 26 million (2019)

Thought: Based on the search results in the `Observation` from the previous step, I know that Shanghai has the highest population.
Answer: Based on the search results, Shanghai has the highest population.
Successful: True
</example>

<example>
Generate a video of the moon.]

Thought: The user has asked to generate a video of the moon. Unfortunately, I do not have any tool that can generate a video. So, I can't solve this task.
Answer: Unfortunately, I lack the ability to solve this task at this moment. May I help you with something else?
Successful: False
</example>

<example>
Translate the content of 'article.txt' into Bengali

Thought: The user wants a translation of 'article.txt'. I will first read the contents of the file. Since no specific tool for translation is available, I'll print the contents so that it becomes available to me (the LLM) for translation.
Code: ```py
with open('article.txt', 'r', encoding='utf-8') as file:
    print(file.read())
```
Observation: Hello, how are you?

Thought: In the previous step, I have already read the 'article.txt' file and printed its contents: 'Hello, how are you?'. I can translate this text into Bengali (output language) myself without using any further tools and provide the final answer.
Answer: হ্যালো, কেমন আছো?
Successful: True
</example>
</examples>

<guidelines>
- Always generate a `Thought` and `Code` sequence unless you are providing a final `Answer`.
- Do not name new variables with the same name as a tool.
- Smartly decide when to use specialized tools vs. write simple Python code vs. use your innate capabilities.
- Use tools only when needed and only those listed. Prefer tools over writing complex custom code.
- Always use the correct arguments for tools (e.g., tool(arg='value'), not tool({{'arg': 'value'}})).
- Remember to import allowed Python modules before using them within a `Code` block.
- Do NOT print secrets (API keys, passwords).
- Your `Thought` MUST reason whether to print file contents.
  ONLY print if necessary and explicitly justified in your `Thought`.
</guidelines>

<current_interaction>
{history}
</current_interaction>
</prompt>
'''

CODE_ACT_AGENT_CONTEXTUAL_PROMPT = '''
You are an expert, helpful agent designed to solve complex tasks iteratively using available tools & Python code.

Your core operating principles are:
1. Iterative Problem Solving: Propose a step (`Thought`), execute code/tool use (`Code`), observe the outcome (`Observation`), then refine until the task is solved.
2. Self-Correction & Adaptive Reasoning: Analyze each `Observation`. If an error occurs, diagnose and adapt.
   Never repeat failed attempts without modification.
3. State Tracking & Task Progression: Maintain awareness of completed sub-tasks.
   Before writing new `Code`, explicitly refer to prior results and reuse available computations.
4. Loop Avoidance & Optimal Execution: Ensure each iteration meaningfully advances the task.
   Avoid redundant execution of already completed actions and move forward.


# Task

Read the following task description very carefully:
{task}

(Optional) Input files/URLs for this task:
{task_files}


## Tools

The following *specialized* tools are available for your use:
{tool_names}

You are responsible for writing Python code to use these tools and only the following standard libraries:
{authorized_imports}

Do NOT import the provided tool names.


## Task Plan

Optionally, here's a TODO list of items (general plan to follow to solve the task) -- ignore if unavailable. 
Align your `Thought` with the steps from the plan, marking achieved steps mentally.
<todo_list>
{plan}
</todo_list>

# Output Format

Adhere strictly to the following Thought-Code-Observation cycle:

Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Code: ```py
# Write your Python code here
print(useful_result_information_found)
```
Observation: [Output from code execution or tool use.]

In `Thought`, based on the current task status and the `Observation` from the previous step,
describe precisely your next action and explaining why this specific `Code` block is necessary now.
If you see a loop, explain how you will break it.

Use `print()` for any result/useful information you want to observe based on code execution.
When necessary, you can apply your innate capabilities (e.g., summarization, translation, static
code analysis, image analysis, and so on) on what you see under `Observation`.
Avoid printing trivial text (e.g., printing to say that you will do something).

Every iteration of the Thought-Code-Observation cycle must move you closer to completing the task.
Keep track of the steps already completed and pending.
Your `Thought` for each turn must focus on what needs to be done next to *advance* the task.
Repeat this cycle until you have enough information to provide a final answer.
For the final answer, use one of these formats:

Thought: I have enough information to answer. I will use the user's language.
Answer: [Your answer in the user's language]
Successful: True

Thought: I cannot answer the question with the provided tools/information.
Answer: [Your explanation in the user's language]
Successful: False

The `Successful` flag should only be `True` for a completed task.

Craft your final answer by carefully reading the instructions from the task.
When applicable, follow the input style from the given task.


# Examples & Anti-Patterns (with annotations)

The examples below illustrate the Thought-Code-Observation process and common mistakes to AVOID.
IMPORTANT: Every `Code` block is independent. Variables from one block are NOT available in subsequent blocks.

---
[Task: Generate an image of the oldest person in this document.]

Thought: The current language of the user is: English. I will begin by identifying the oldest person mentioned in the document using the `document_qa tool`. I only need to print the answer, not the entire document.
Code: ```py
answer = document_qa(document=document, question='Who is the oldest person mentioned?')
print(answer)
```
Observation: The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland.

Thought: Based on the latest `Observation`, I have identified John Doe, aged 55, as the oldest person. He lives in Newfoundland, Canada. As my next logical step, I'll use the `image_generator` tool to generate his portrait.
Code: ```py
image_path = image_generator(prompt='A portrait of John Doe, a 55-year-old man living in Canada.')
print(f'The output image file is: {{image_path}}')
```
Observation: The output image file is: image.png

Thought: Based on the given document, John Doe (55) is the oldest person. I have also generated his portrait and saved it in the image.png file.
Answer: An image of the oldest person has been generated and saved as image.png
Successful: True

---
[Task: Which city has the highest population: Guangzhou or Shanghai?]

Thought: The current language of the user is: English. I need to get the populations for both cities and compare them: I will use the tool `search` for this purpose. Since search results are important for this task, I'll print them.
Code: ```py
for city in ['Guangzhou', 'Shanghai']:
print(f'Population {{city}}:', search(f'{{city}} population'))
```
Observation: Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: 26 million (2019)

Thought: Based on the search results in the `Observation` from the previous step, I know that Shanghai has the highest population.
Answer: Based on the search results, Shanghai has the highest population.
Successful: True

---
[Task: Generate a video of the moon.]

Thought: The current language of the user is: English. The user has asked to generate a video of the moon. Unfortunately, I do not have any tool that can generate a video. So, I can't solve this task.
Answer: Unfortunately, I lack the ability to solve this task at this moment. May I help you with something else?
Successful: False

---
[Task: Translate the content of 'article.txt' into Bengali]

Thought: The user wants a translation of 'article.txt'. I will first read the contents of the file. Since no specific tool for translation is available, I'll print the contents so that it becomes available to me (the LLM) for translation.
Code: ```py
with open('article.txt', 'r', encoding='utf-8') as file:
    print(file.read())
```
Observation: Hello, how are you?

Thought: In the previous step, I have already read the 'article.txt' file and printed its contents: 'Hello, how are you?'. I can translate this text into Bengali (output language) myself without using any further tools and provide the final answer.
Answer: হ্যালো, কেমন আছো?
Successful: True

---
[Task: Plot data from the file at http://example.com/data.csv]

Thought: The current language of the user is: English. I need to plot data from the CSV. I will first download the file using `download_file`, then read its contents using `read_csv_file`, and finally plot the first two columns using `line_plot`. I will only print the image path, not the entire data.
Code: ```py
file_path = download_file(url='http://example.com/data.csv')
data = read_csv_file(file_path)
img_path = line_plot(data, cols=[1, 2])
print(f'The image path is: {{img_path}}')
```
Observation: The output image file is: figure.png

Thought: Based on the latest `Observation`, the graph has been plotted and saved as figure.png. I have completed the task.
Answer: The graph is saved as figure.png
Successful: True

---
[Task: Word count in https://example.com/article.txt]

Thought: The current language of the user is: English. I'll start by downloading the file using the `download_file` tool.
Code: ```py
path = download_file(url='https://example.com/article.txt')
print(path)
```
Observation: /tmp/somethingXYz

Thought: The current language of the user is: English. I'll extract the contents of the file.
Code: ```py
with open('/tmp/somethingXYz', 'r', encoding='utf-8') as file:
    print(file.read())
```
Observation: Content of the file ...(truncated for brevity)

Thought: The current language of the user is: English. I'll download the file using the `download_file` tool.  # <--- This is a SUBOPTIMAL Thought since it generates the previous thought again without referring to the current task status, repeating an already accomplished step, and then getting STUCK in a loop!
Code: ```py
print(download_file(url='https://example.com/article.txt'))
```


# General Guidelines

- Always generate a Thought-Code sequence.
- Do not name new variables with the same name as a tool.
- Use tools only when needed and only those listed. Prefer tools over writing complex custom code.
- Always use the correct arguments for tools (e.g., tool(arg='value'), not tool({{'arg': 'value'}})).
- Remember to import allowed Python modules before using them within a Code block.
- Do NOT print secrets (API keys, passwords).
- Your Thought MUST reason whether to print file contents.
  ONLY print if necessary and explicitly justified in your Thought.


# Context of Current Interaction

Below is a summary of what has been done so far for the current task.
It includes what was successful, what failed, and what is pending.
Based on this context, you should plan your next step to solve the task without repeating past actions.
DO NOT repeat steps that have already been completed successfully. If a step failed, analyze the error and try a different approach.

{history_summary}


'''

RELEVANT_TOOLS_PROMPT = '''
You are an expert resource planner for solving tasks. You pick the best and necessary tools to solve any task.

Given the following task:
{task_description}

and optional files or URLs for the task:
{task_files}

And the following available tools:

{tool_descriptions}

Which of the above tools are relevant to solving the task?
Identify carefully based on the task and tool descriptions so that no relevant tool is missed.
Return only a comma-separated list of tool names. For example: tool_name1,tool_name2
If no tools are relevant, return an empty string.
'''

AGENT_PLAN_PROMPT = '''
You are a helpful planning assistant. Given a task, you can create a plan to solve the task.
A given task can be complex -- you may need to split it into smaller sub-tasks so that
collectively completing the sub-tasks would enable to achieve the main task.

You're agent type: {agent_type}


## Task

{task}

(Optional) input file paths/URLs associated with this task are as follows:
{task_files}

Your output should be a numbered list of step-by-step plan.
Each step listing only one sub-task in plain English, without any code.
'''

UPDATE_PLAN_PROMPT = '''
You are an expert plan progress tracker.
Given the current plan, the last thought of an agent, and the observation from the last action, update the plan.
- Mark completed tasks with `[x]`.
- Keep pending tasks as `[ ]`.
- If necessary, add new tasks.
- If necessary, modify existing tasks.
- Return ONLY the updated plan in markdown checklist format.

Current Plan:
{plan}

Last Thought:
{thought}

Last Observation:
{observation}

Updated Plan:
'''

SUPERVISOR_TASK_PROMPT = '''
You are the supervisor of an AI agency having one or more helpful agents. Given a task, as well as
the capabilities of the agents, you decide which agent(s) and tool(s) of the agent to use to solve
the task. A given task can be complex -- you may need to split it into smaller parts and invoke
different agents to solve each part using their respective tools. In other words, your job is to
efficiently delegate tasks or subtasks to the agents, collect the results, and delegate again until
a final, satisfactory task completion result is found. You should do it carefully without getting
stuck in the same loop.

Important: Carefully read the original given task. When delegating tasks to the agents or when you
need to split it into sub-tasks, remember to retain ALL information from the original task.
Even punctuations matter sometimes! Otherwise, you might get stuck in an infinite loop where you
ask an agent something but get a different thing in response. The specifications, expectations,
and responses need to be in sync.

Also, tool usage is efficient, so accept the results obtained by using tools of the agents. Unless,
the results returned indicate some obvious error, in which case you ask the agent again by rephrasing
its task along with your feedback.


## Task

{task}

(Optional) input file paths/URLs associated with this task are as follows:
{task_files}


## Agents

The following agents are available to you, each identified with a unique integer ID starting from 0:
{agents}



CRITICAL Guidelines on task division:
(1) Task division across multiple agents MUST be planned carefully and efficiently,
taking the CONTEXT and need for data sharing into consideration.
E.g., avoid asking Agent X to read a data file and Agent Y to plot using that data
since that can lead to failure as Agent Y does not have the relevant data
available. An option is to pass the data (or file/database contents) to Y along with the sub-task
description. However, this can lead to an overhead when the contents are long. So, a better option
is to ask the SAME agent to read data and plot (or process data, in general), provided the agent
is capable of doing it.
(2) If an agent fails to solve a task and you have to ask it to repeat again, rephrase the task
with appropriate context so that the agent can RESUME where it failed. Do NOT ask any agent to do
the same thing again unless it is absolutely necessary. E.g., if an agent is tasked to identify
the keywords from 10 papers on AI on ArXiv, and it has failed after identifying the papers, do NOT
make the agent do that again -- pass it the list of already identified papers from the search results 
so that it can resume from there. This is important to avoid infinite loops of doing nothing useful.


## Task Completion

In case you find that the main task and sub-tasks have been successfully completed, generate
a final answer for the user by nicely collating the results of all the sub-tasks. Also, set
`task_complete` to True. This is the ONLY way to stop your iteration or prevent repeating the same things!
 
 
## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages (initially empty).
Also, agent's response are depicted as user's response.

{history}


Do your best! Don't give up! You're in charge of solving the task, not providing directions to solve it.
'''

SUPERVISOR_TASK_CHECK_PROMPT = '''
Given this task:
{task}

and this sequence of response by agents tool usage:
{response}

determine whether or not the task has been successfully completed.

In case the given attempts appear to effectively capture the final result but fall short only
in some minor way, e.g., not properly formatted, you can give a finishing touch and capture
the result of the task in the `final_answer` field. In this case, also set `status` to `True`.
Otherwise, leave `final_answer` empty when there are significant aspects missing or major deviations
noted from the desired ask result.

Tool usage is generally efficient, so in most cases the results can be accepted unless there is some
indication of obvious error.
'''

SALVATION_PROMPT = '''
You are a helpful AI agency having one or more agents/assistants. You help users by solving their
tasks. Sometimes, due to unpredictable reasons, you might fail to solve the task entirely or
partially. Also, sometimes, you might have completed the task but failed to communicate the final
answer to the user due to some error.

You are here today to address one such scenario.

Given the following task:
{task}

and optional files associated with the task:
{task_files}

Here's a log of what you have done and achieved:
{history}

In the conversation history above, you will find the original task of the user and optionally
delegated sub-tasks.

Your job is to salvage any useful information/output/action related to user's task from the above
sequence of activities. Here's how to respond:
- If you find that the agent/assistant have completed the task satisfactorily (unless there is any
obvious error message or significant deviation from what the task had asked to achieve), simply
generate a final response based on what is available.
- If you notice that one more steps or sub-tasks remain unachieved or failed, begin by apologising.
Identify the useful information available and prepare a final response. After that, display
a bulleted list of what aspects of the task could not be achieved or failed or encountered error. 
- If no portion of the task could be competed, say so and begin by apologising. Then show
a bulleted list of what went wrong. (Skip this part if the task was successful.)

Aside from minor formatting and presenting to the users in a readable way, avoid adding
to the results/facts already found, only report them.
Avoid telling users terms like "logs", "history", "salvaged", and "accepted by you" when responding.
Also, users need to have all information available to them -- avoid telling them to see
agent's previous attempts or tool's previous outputs. 

'''

CONTEXTUAL_SUMMARY_PROMPT = '''
You are an expert summarizer. Given a task and a history of interactions between an AI agent and tools, create a concise summary.
The summary should highlight:
1. What has been attempted.
2. What succeeded.
3. What failed (including errors).
4. What information has been gathered.
5. What is the next logical step to move forward and solve the task.

The summary should not be a list, but a concise paragraph.

Task: {task}

Interaction History:
{history}
'''


def tool(func: Callable) -> Callable:
    """
    A decorator to convert any Python function into a tool with additional metadata.
    Tooling based on async functions is not supported.

    Args:
        func (Callable): The function to be converted into a tool.

    Returns:
        Callable: The decorated function with additional metadata.
    """
    if asyncio.iscoroutinefunction(func):
        raise ValueError(
            'Tooling based on async functions is not supported. Please remove `async` from'
            f' the signature of the `{func.__name__}` function or remove the `@tool` decorator.'
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Create a schema for the function arguments using Pydantic
    signature = inspect.signature(func)
    fields = {name: (param.annotation, ...) for name, param in signature.parameters.items()}

    # Add metadata to the function
    wrapper.name = func.__name__
    wrapper.description = textwrap.dedent(func.__doc__).strip() if func.__doc__ else ''
    wrapper.args_schema = pyd.create_model(func.__name__, **fields)

    return wrapper


@tool
def calculator(expression: str) -> Union[float, None]:
    """
    A simple calculator tool that can evaluate basic arithmetic expressions.
    The expression must contain only the following allowed mathematical symbols:
    digits, +, -, *, /, ., (, )

    The ^ symbol, for example, is not allowed. For exponent, use **.
    In case the expression has any invalid symbol, the function returns `None`.

    Args:
        expression (str): The arithmetic expression as a string.

    Returns:
        The numerical result or `None` in case an incorrect arithmetic expression is provided
         or any other error occurs.
    """
    import re

    # Tools should be self-contained, including the imports
    # This allows their usage in an isolated environment
    # That is why, we import `re` and compute the regex inside this tool definition, not outside
    expression = expression.replace("'", "").replace('^', '**')

    # Define a regex pattern for valid mathematical expressions
    # It's important to define it inside the tool so that the function is complete by itself
    calculator_regex = re.compile(r'^[\d+\-*/().\s]+$')

    if calculator_regex.match(expression) is not None:
        try:
            # Evaluate the expression safely
            result = eval(expression)
            return result
        except Exception as e:
            print(f'calculator:: Error evaluating expression: {e}')
            return None
    else:
        print(f'calculator:: Invalid expression: {expression}')
        return None


@tool
def web_search(query: str, max_results: int = 10, show_description: bool = False) -> str:
    """
    Search the Web using DuckDuckGo. The input should be a search query.
    Use this tool when you need to answer questions about current events.
    Returns (as Markdown text) the top search results with titles, links, and optional descriptions.
    NOTE: The returned URLs should be visited to retrieve the contents the pages.

    Args:
        query: The query string.
        max_results: Maximum no. of search results (links) to return.
        show_description: If `True`, includes the description of each search result.
         Default is `False`.

    Returns:
         The search results.
    """
    import time
    import random

    try:
        from ddgs import DDGS
    except ImportError as e:
        raise ImportError(
            '`ddgs` was not found! Please run `pip install ddgs`.'
        ) from e

    # Note: In general, `verify` should be `True`
    # In some cases, DDGS may fail because of proxy or something else;
    # can set it to `False` but generally not recommended
    results = DDGS(verify=False).text(query, max_results=max_results)
    # DDGS throws a rate limit error
    time.sleep(random.uniform(1.5, 3.5))
    if len(results) == 0:
        return 'No results found! Try a less restrictive/shorter query.'

    if not show_description:
        # If descriptions are not needed, only return titles and links
        results = [f"[{result['title']}]({result['href']})" for result in results]
    else:
        results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
    return '## Search Results\n\n' + '\n\n'.join(results)


@tool
def file_download(url: str) -> str:
    """
    Download a file from the Web and save it locally on the disk.
    (If the `extract_as_markdown` tool does not work, this can be used an alternative.)

    Args:
        url: The URL pointing to the file (must be a correct URL).

    Return:
        Path to the locally saved file or error messages in case of any exception.
    """
    import os
    import requests
    import tempfile

    response = requests.get(url, timeout=20, stream=True, headers={'user-agent': 'kodeagent/0.0.1'})
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_file_path = tmp_file.name
        if os.name == 'nt':
            tmp_file_path = tmp_file_path.replace('\\', '/')
    return tmp_file_path


@tool
def extract_as_markdown(
        url_or_file_path: str,
        scrub_links: bool = True,
        max_length: int = None
) -> str:
    """
    Extract the contents from HTML files (.html), PDF files (.pdf), Word Documents (.docx),
    and Excel spreadsheets (.xlsx) as Markdown text. No other file type is supported.
    The text can be used for analysis with LLMs. Input can be a URL or a local file path.
    This tool can directly work with URLs, so no need to download the files separately.
    NOTE: The output returned by this function can be long and may involve lots of quote marks.

    Args:
        url_or_file_path: URL or Path to a .html, .pdf, .docx, or .xlsx file.
        scrub_links: Defaults to `True`, which removes all links from the extracted Markdown text.
         Set it to `False` if you want to retain the links in the text.
        max_length: If set, limit the output to the first `max_length` characters. This can be used
         to truncate the output but doing so can also lead to loss of information.

    Returns:
        The content of the file in Markdown format.
    """
    import re
    import mimetypes

    try:
        from markitdown import MarkItDown
    except ImportError as e:
        raise ImportError(
            '`markitdown` was not found! Please run `pip install markitdown[pdf,docx,xlsx]`.'
        ) from e

    md = MarkItDown(enable_plugins=False)
    try:
        result = md.convert(url_or_file_path.strip()).text_content

        if mimetypes.guess_type(url_or_file_path)[0] == 'application/pdf':
            # Handling (cid:NNN) occurrences in PDFs
            cid_pattern = re.compile(r'\(cid:(\d+)\)')
            matches = set(cid_pattern.findall(result))
            for cid_num in matches:
                cid_str = f'(cid:{cid_num})'
                result = result.replace(cid_str, chr(int(cid_num) + 29))

        if scrub_links:
            # Remove Markdown links [text](url)
            result = re.sub(r'\[([^\]]+)\]\((https?:\/\/[^\)]+)\)', r'\1', result)

        if max_length is not None:
            result = result[:max_length]

        return result
    except Exception as e:
        return str(e)


@tool
def search_wikipedia(query: str, max_results: Optional[int] = 3) -> str:
    """
    Search Wikipedia and return the top search results as Markdown text.
    The input should be a search query. The output will contain the title, summary, and link
    to the Wikipedia page.

    Args:
        query: The search query string.
        max_results: The max. no. of search results to consider (default 3).

    Returns:
        The search results in Markdown format.
    """
    try:
        import wikipedia
    except ImportError as e:
        raise ImportError('`wikipedia` was not found! Please run `pip install wikipedia`.') from e

    try:
        results = wikipedia.search(query, results=max_results)
        if not results:
            return 'No results found! Try a less restrictive/shorter query.'

        markdown_results = []
        for title in results:
            page = wikipedia.page(title)
            markdown_results.append(f"### [{page.title}]({page.url})\n{page.summary}")

        return '\n\n'.join(markdown_results)
    except wikipedia.exceptions.DisambiguationError as de:
        return f'DisambiguationError: Please select an option from {", ".join(de.options)}'


@tool
def get_youtube_transcript(video_id: str) -> str:
    """
    Retrieve the transcript/subtitles for a given YouTube video. It also works for automatically
    generated subtitles, supports translating subtitles. The input should be a valid Youtube
    video ID. E.g., the URL https://www.youtube.com/watch?v=aBc4E has the video ID `aBc4E`.

    Args:
        video_id: YouTube video ID from the URL.

    Returns:
        The transcript/subtitle of the video, if available.
    """
    import youtube_transcript_api
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript_text = ''
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        transcript_text = ' '.join([item.text for item in transcript.snippets])
    except youtube_transcript_api._errors.TranscriptsDisabled:
        transcript_text = (
            '*** ERROR: Could not retrieve a transcript for the video -- subtitles appear to be'
            ' disabled for this video, so this tool cannot help, unfortunately.'
        )

    return transcript_text


@tool
def get_audio_transcript(file_path: str) -> Any:
    """
    Convert audio files to text using OpenAI's Whisper model via Fireworks API.
    The input should be a path to an audio file (e.g., .mp3, .wav, .flac).
    The audio file should be in a format that Whisper supports.

    Args:
        file_path: Local file system path to the audio file.

    Returns:
        The transcript of the audio file as text.
    """
    import os

    import requests

    with open(file_path, 'rb') as f:
        response = requests.post(
            'https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {os.getenv("FIREWORKS_API_KEY")}'},
            files={'file': f},
            data={
                'model': 'whisper-v3-turbo',
                'temperature': '0',
                'vad_model': 'silero'
            },
            timeout=15,
        )

    if response.status_code == 200:
        return response.json()

    return f'Audio transcription error: {response.status_code}: {response.text}'


# The different types of senders of messages
MESSAGE_ROLES = Literal['user', 'assistant', 'system', 'tool']
# The different types of updates emitted by an agent
AGENT_RESPONSE_TYPES = Literal['step', 'final', 'log']


class Task(pyd.BaseModel):
    """
    Task to be solved by an agent.
    """
    id: str = pyd.Field(description='Auto-generated task ID', default_factory=uuid.uuid4)
    description: str = pyd.Field(description='Task description')
    files: Optional[list[str]] = pyd.Field(description='A list of file paths or URLs')
    result: Optional[Any] = pyd.Field(description='Task result', default=None)
    is_finished: bool = pyd.Field(
        description='Whether the task has finished running', default=False
    )
    is_error: bool = pyd.Field(
        description='Whether the task execution resulted in any error', default=False
    )


class ChatMessage(pyd.BaseModel):
    """
    Generic chat message. This is primarily intended to internal and tool usage.
    Agents shouldn't ask an LLM to respond in this format. In particular, Gemini would fail
    because of `Any`.
    """
    role: MESSAGE_ROLES = pyd.Field(description='Role of the message sender')
    content: Any = pyd.Field(description='Content of the message')


class ReActChatMessage(ChatMessage):
    """
    Messages for the ReAct agent.
    """
    # The content field will not be used by this message (but the LLM can still assign a value)
    # Higher versions of Pydantic allows to exclude the field altogether
    content: Optional[str] = pyd.Field(description='Unused', exclude=True)
    thought: str = pyd.Field(description='Thoughts behind the tool use')
    action: str = pyd.Field(description='Name of the tool to use')
    # Gemini complains about empty objects if `args` is defined as dict,
    # hence string type for compatibility
    args: str = pyd.Field(description='Tool arguments as JSON string')
    answer: Optional[str] = pyd.Field(
        description='Final answer for the task; set only in the final step', default=None
    )
    successful: bool = pyd.Field(description='Task completed or failed? (initially False)')


class CodeChatMessage(ChatMessage):
    """
    Messages for the CodeActAgent.
    """
    # The content field will not be used by this message (but the LLM can still assign a value)
    # Higher versions of Pydantic allows to exclude the field altogether
    content: Optional[str] = pyd.Field(description='Unused', exclude=True)
    thought: str = pyd.Field(description='Thoughts behind the code')
    code: str = pyd.Field(description='Python code with tool use')
    answer: Optional[str] = pyd.Field(
        description='Final answer for the task; set only in the final step', default=None
    )
    successful: bool = pyd.Field(description='Task completed or failed? (initially False)')


class SupervisorTaskMessage(pyd.BaseModel):
    """
    Messages for the supervisor-agent task delegation.
    """
    agent_id: int = pyd.Field(
        description='Integer agent ID based on the instructions (starting from 0)'
    )
    task: str = pyd.Field(
        description='Task or sub-task description to be delegated to an agent'
    )
    image_files: Optional[list[str]] = pyd.Field(
        description='Optional list of image file paths/URLs associated with the task'
    )
    facts_available: str = pyd.Field(
        description='A list of objective facts collected/observed so far'
    )
    task_complete: bool = pyd.Field(
        description=(
            'Initially False; set to True only when the agent(s) have successfully competed'
            ' all the sub-tasks'
        )
    )
    final_answer: str = pyd.Field(
        description=(
            'The final answer for the user when the main task is done, i.e., `task_complete`'
            ' is True. Set to empty string otherwise.'
        )
    )


class DelegatedTaskStatus(pyd.BaseModel):
    """
    The status of a task delegated by the supervisor.
    """
    status: bool = pyd.Field(description='Either `True` or `False`')
    reason: str = pyd.Field(
        description='Brief explanation for the status, e.g., why the task is incomplete'
    )
    how_to_fix: str = pyd.Field(
        description='Briefly describe how to/what would fix the task result'
    )
    final_answer: str = pyd.Field(
        description='Final solution of the task, if found. Otherwise, empty string.'
    )


class AgentResponse(TypedDict):
    """
    Streaming response sent by an agent in the course of solving a task. The receiver can decide
    what to do with the response based on its type.
    """
    type: AGENT_RESPONSE_TYPES
    channel: Optional[str]
    value: Any
    metadata: Optional[dict[str, Any]]


class Agent(ABC):
    """
    An abstract agent. This should serve as the base class for all types of agents.
    All subclasses must override at least the `run()` method.
    """
    def __init__(
            self,
            name: str,
            model_name: str,
            description: Optional[str] = None,
            vision_model_name: Optional[str] = None,
            tools: Optional[list[Callable]] = None,
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
            filter_tools_for_task: bool = False,
    ):
        """
        Initialize an agent.

        Args:
            name: The name of the agent.
            description: Description of the agent's capabilities or scope. Recommended to have.
            model_name: The name of the LLM to be used.
            vision_model_name: (Optional) vision model to use; None by default.
            tools: A list of tools available to the agent.
            litellm_params: Optional parameters for LiteLLM.
            max_iterations: Maximum number of iterations for task solving.
            filter_tools_for_task: Whether to filter tools based on task relevance.
        """
        self.id = uuid.uuid4()
        self.name: str = name
        self.description = description
        self.model_name: str = model_name
        self.vision_model_name = vision_model_name or model_name

        self.tools = tools or []
        self.filter_tools_for_task = filter_tools_for_task
        self.litellm_params: dict = litellm_params or {}
        self.max_iterations = max_iterations

        self.tool_names = {t.name for t in tools} if tools else set()
        self.tool_name_to_func = {t.name: t for t in tools} if tools else {}

        self.task: Optional[Task] = None
        self.messages: list[ChatMessage] = []
        self.msg_idx_of_new_task: int = 0
        self.final_answer_found = False
        self.plan = ''

    def __str__(self):
        return (
            f'Agent: {self.name} ({self.id}); LLM: {self.model_name}; Tools: {self.tools}'
        )

    def _format_plan_as_todo(self, plan: str) -> str:
        """
        Convert a numbered list plan into a markdown checklist.
        """
        lines = plan.strip().split('\n')
        todo_list = []
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.\s*', line):
                # Remove the number and dot
                task = re.sub(r'^\d+\.\s*', '', line)
                todo_list.append(f'- [ ] {task}')
            elif line:
                todo_list.append(f'- [ ] {line}')
        return '\n'.join(todo_list)

    async def _update_plan_progress(self):
        """
        Update the plan based on the last thought and observation.
        """
        last_thought = ''
        last_observation = ''
        if len(self.messages) > 1:
            if isinstance(self.messages[-2], (ReActChatMessage, CodeChatMessage)):
                last_thought = self.messages[-2].thought
            if self.messages[-1].role == 'tool':
                last_observation = self.messages[-1].content

        prompt = UPDATE_PLAN_PROMPT.format(
            plan=self.plan,
            thought=last_thought,
            observation=last_observation
        )
        updated_plan = await self._call_llm(ku.make_user_message(prompt), trace_id=self.task.id)
        self.plan = updated_plan

    async def get_history_summary(self) -> str:
        """
        Generate a summary of the conversation history.
        """
        history = self.format_messages_for_prompt(start_idx=self.msg_idx_of_new_task)
        if not history.strip():
            return "No activities yet."

        prompt = CONTEXTUAL_SUMMARY_PROMPT.format(
            task=self.task.description,
            history=history
        )
        summary = await self._call_llm(
            ku.make_user_message(prompt),
            trace_id=self.task.id if self.task else None
        )
        return summary

    async def get_relevant_tools(
            self,
            task_description: str,
            task_files: Optional[list[str]] = None,
    ) -> list[Any]:
        """
        Calls an LLM to determine which tools are relevant for the given task.

        Args:
            task_description: The task description.
            task_files: Optional list of files associated with the task.

        Returns:
            A list of relevant tools or all tools, in case of error.
        """
        tool_descriptions = self.get_tools_description()
        prompt = RELEVANT_TOOLS_PROMPT.format(
            task_description=task_description,
            task_files=task_files,
            tool_descriptions=tool_descriptions,
        )

        try:
            response = await self._call_llm(
                ku.make_user_message(prompt),
                trace_id=self.task.id if self.task else None
            )
            relevant_tool_names = response.split(',') if response.strip() else []
            relevant_tool_names = {t.strip() for t in relevant_tool_names if t.strip()}
            logger.debug('Relevant tool names: %s', relevant_tool_names)
            relevant_tools = [
                t for t in self.tools if t.name in relevant_tool_names
            ]
            return relevant_tools
        except Exception as e:
            logger.error('Error determining relevant tools: %s', str(e))
            return list(self.tools)

    def _run_init(
            self,
            description: str,
            files: Optional[list[str]] = None,
            task_id: Optional[str] = None
    ):
        """
        Initialize the running of a task by an agent.
        """
        self.add_to_history(ChatMessage(role='user', content=description))
        self.task = Task(description=description, files=files)
        if task_id:
            self.task.id = task_id
        self.msg_idx_of_new_task = len(self.messages)
        self.final_answer_found = False  # Reset from any previous task

    @abstractmethod
    async def run(
            self,
            task: str,
            files: Optional[list[str]] = None,
            task_id: Optional[str] = None
    ) -> AsyncIterator[AgentResponse]:
        """
        Execute a task using the agent.

        Args:
            task: A description of the task.
            files: An optional list of file paths or URLs.
            task_id: (Optional) An ID for the task, if provided by the caller.

        Yields:
            An update from the agent.
        """

    async def _call_llm(
            self,
            messages: list[dict],
            response_format: Optional[
                Type[ChatMessage | SupervisorTaskMessage | DelegatedTaskStatus]
            ] = None,
            trace_id: Optional[str]=None,
    ) -> str:
        """
        Invoke the LLM to generate a response based on a given list of messages.

        Args:
            messages: A list of messages (and optional images) to be sent to the LLM.
            response_format: Optional type of message the LLM should respond with.
            trace_id: (Optional) Langfuse trace ID.

        Returns:
            The LLM response as string.
        """
        params = {
            'model': self.model_name,
            'messages': messages,
        }
        if response_format:
            params['response_format'] = response_format
        params.update(self.litellm_params)
        response = litellm.completion(**params, metadata={'trace_id': str(trace_id)})

        token_usage = {
            'cost': response._hidden_params['response_cost'],
            'prompt_tokens': response.usage.get('prompt_tokens'),
            'completion_tokens': response.usage.get('completion_tokens'),
            'total_tokens': response.usage.get('total_tokens'),
        }
        logger.info(token_usage)
        return response.choices[0].message['content']

    def response(
            self,
            rtype: AGENT_RESPONSE_TYPES,
            value: Any,
            channel: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Prepare a response to be sent by the agent. The calling method must yield this response.

        Note: `response` is not made a static method so that only the agents can invoke it.

        Args:
            rtype: The type of the response.
            value: The response value (content).
            channel: Optional channel (e.g., the method name that generated this response).
            metadata: Optional metadata.

        Returns:
            The agent's response.
        """
        return {'type': rtype, 'channel': channel, 'value': value, 'metadata': metadata}

    def add_to_history(self, message: ChatMessage):
        """
        Add a chat message, generated by user, AI, or tool, to the agent's message history.

        Args:
            message: The message. Must be a valid `ChatMessage` instance.
        """
        assert isinstance(message, ChatMessage), (
            f'add_to_history() expects a `ChatMessage`; got `{type(message)}`'
        )
        self.messages.append(message)

    def format_messages_for_prompt(self, start_idx: int = 0) -> str:
        """
        Generate a formatted string based on the historical messages that can be injected
        into a prompt. The formatting may differ based on the prompts used by different types
        of agents. Subclasses should override this method accordingly.

        Args:
            start_idx: The start index of messages to consider (default 0).

        Returns:
            A formatted string containing the messages.
        """
        return self.get_history(start_idx)

    def get_tools_description(self, tools: Optional[list[Any]] = None) -> str:
        """
        Generate a description of all the tools available to the agent.

        Args:
            tools: Optional list of tools to describe. If not provided, uses the agent's tools.

        Returns:
            A description of the requested or all available tools.
        """
        description = ''
        filtered_tool_names = {t.name for t in (tools or self.tools)}
        for t in self.tools:
            if t.name in filtered_tool_names:
                description += f'- Tool name: {t.name}'
                # description += f'\n  -
                # Schema: {t.args_schema.model_json_schema()}'
                description += f'\n- Tool description: {t.description}'
                description += '\n---\n'

        return description

    @property
    def purpose(self) -> str:
        """
        Describe the name, purpose of, and tools available to an agent.

        Returns:
             A text description of the agent.
        """
        description = f'Name: {self.name}\nDescription: {self.description or "N/A"}'
        description += f'\nTools available to this agent (`{self.name}`):'
        description += f'\n{self.get_tools_description()}'

        return description

    def get_history(self, start_idx: int = 0) -> str:
        """
        Get a formatted string representation of all the messages.

        Args:
            start_idx: The start index of messages to consider (default 0).

        Returns:
            A sequence of the messages showing their role and content.
        """
        return '\n'.join([f'[{msg.role}]: {msg.content}' for msg in self.messages[start_idx:]])

    def clear_history(self):
        self.messages = []


class ReActAgent(Agent):
    """
    Reasoning and Acting agent with thought-action-observation loop.
    """
    def __init__(
            self,
            name: str,
            model_name: str,
            tools: list,
            description: Optional[str] = None,
            vision_model_name: Optional[str] = None,
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
            filter_tools_for_task: bool = False,
            contextual: bool = False,
            use_planning: bool = False,
    ):
        """
        Instantiate a ReAct agent.

        Args:
            name: The name of the agent.
            description: Description of the agent's capabilities or scope. Recommended to have.
            model_name: The name of the LLM to be used (use names from LiteLLM).
            tools: The tools available to the agent.
            litellm_params: Optional parameters for LiteLLM.
            max_iterations: The maximum number of steps that the agent should try to solve a task.
        """
        super().__init__(
            name=name,
            model_name=model_name,
            tools=tools,
            vision_model_name=vision_model_name,
            litellm_params=litellm_params,
            description=description,
            max_iterations=max_iterations,
            filter_tools_for_task=filter_tools_for_task,
        )

        self.contextual = contextual
        self.use_planning = use_planning
        self.plan: Optional[str] = None
        self.final_answer_found: bool = False
        if tools:
            logger.info('Created agent: %s; tools: %s', name, [t.name for t in tools])

    async def run(
            self,
            task: str,
            files: Optional[list[str]] = None,
            task_id: Optional[str] = None
    ) -> AsyncIterator[AgentResponse]:
        """
        Solve a task using ReAct's TAO loop.

        Args:
            task: A description of the task.
            files: An optional list of file paths or URLs.
            task_id: (Optional) An ID for the task, if provided by the caller.

        Yields:
            An update from the agent.
        """
        self._run_init(task, files, task_id)

        yield self.response(
            rtype='log',
            value=f'Solving task: `{self.task.description}`',
            channel='run'
        )

        if self.use_planning:
            messages = ku.make_user_message(
                text_content=AGENT_PLAN_PROMPT.format(
                    agent_type=self.__class__.__name__,
                    task=self.task.description,
                    task_files='\n'.join(self.task.files) if self.task.files else '[None]',
                    # tool_names=self.get_tools_description(),
                ),
                files=self.task.files,
            )
            plan: str = await self._call_llm(messages=messages, trace_id=self.task.id)
            self.plan = self._format_plan_as_todo(plan)
            yield self.response(rtype='log', value=f'Plan:\n{self.plan}', channel='run')

        for idx in range(self.max_iterations):
            if self.final_answer_found:
                break

            yield self.response(rtype='log', channel='run', value=f'* Executing step {idx + 1}')
            # The thought & observation will get appended to the list of messages
            async for update in self._think(plan=self.plan):
                yield update
            async for update in self._act():
                yield update

            if self.use_planning and self.plan:
                await self._update_plan_progress()
            print('-' * 30)

        if not self.final_answer_found:
            yield self.response(
                rtype='final',
                value=(
                    f'Sorry, I failed to get a complete answer'
                    f' even after {self.max_iterations} steps!'
                ),
                channel='run'
            )

    async def _think(self, plan: Optional[str] = None) -> AsyncIterator[AgentResponse]:
        """
        Think about the next step to be taken to solve the given task.

        The LLM is prompted with the available tools and the TAO sequence so far. Based on them,
        the LLM will suggest the next action. "Think" of ReAct is also "Observe."

        Args:
            plan: A tentative plan to solve the task [Optional].

        Yields:
            Update from the thing step.
        """
        # Note: we're not going to chat with the LLM by sending a sequence of messages
        # Instead, every think step will send a single message containing all historical info
        if self.filter_tools_for_task:
            relevant_tools = await self.get_relevant_tools(
                task_description=self.task.description, task_files=self.task.files
            )
        else:
            relevant_tools = self.tools

        if self.contextual:
            history_summary = await self.get_history_summary()
            prompt_template = REACT_CONTEXTUAL_PROMPT
            history_kwargs = {'history_summary': history_summary}
        else:
            prompt_template = REACT_PROMPT
            history_kwargs = {
                'history': self.format_messages_for_prompt(start_idx=self.msg_idx_of_new_task)
            }

        message = prompt_template.format(
            task=self.task.description,
            task_files='\n'.join(self.task.files) if self.task.files else '[None]',
            tool_names=self.get_tools_description(relevant_tools),
            plan=plan or '<No plan provided; please plan yourself>',
            **history_kwargs,
        )
        msg = await self._record_thought(message, ReActChatMessage)
        yield self.response(rtype='step', value=msg, channel='_think')

    async def _record_thought(
            self,
            message: str,
            response_format_class: Type[ChatMessage]
    ):
        """
        Utility method covering the common aspects of the "think" step of the T*O loop.

        Args:
            message: A single, formatted message with history to be sent to the LLM.
            response_format_class: The type of message used by this agent.

        Returns:
            A message of the `response_format_class` type.
        """
        prompt = ku.make_user_message(text_content=message, files=self.task.files)
        response = await self._call_llm(
            messages=prompt,
            response_format=response_format_class,
            trace_id=self.task.id,
        )
        # Sometimes parsing errors are noticed when the JSON appears to have long text, e.g.,
        # contents of files
        for _ in range(3):
            try:
                json.loads(response)
                break
            except JSONDecodeError:
                response = json_repair.repair_json(response)

        msg: response_format_class = response_format_class.model_validate_json(response)
        msg.role = 'assistant'
        self.add_to_history(msg)
        return msg

    async def _act(self) -> AsyncIterator[AgentResponse]:
        """
        Take action based on the agent's previous thought.

        The LLM has suggested an action. This method will identify the tool suggested and
        execute it.

        Yields:
            Updates from the acting step.
        """
        prev_msg: ReActChatMessage = self.messages[-1]  # type: ignore
        if (
                prev_msg.answer == ''
                and (prev_msg.action == '' or prev_msg.args == '' or prev_msg.thought == '')
        ):
            self.add_to_history(
                ChatMessage(
                    role='tool',
                    content=(
                        '* Error: incorrect response generated. Must have values for the `answer`'
                        ' or the `action`, `args`, and `thought` fields.'
                    )
                )
            )
            return

        if prev_msg.answer:
            # The final answer has been found!
            self.final_answer_found = True
            self.task.is_finished = True
            self.task.is_error = prev_msg.successful
            response_msg = ChatMessage(role='assistant', content=prev_msg.answer)
            self.add_to_history(response_msg)

            yield self.response(
                rtype='final',
                value=response_msg,
                channel='_act',
                metadata={'final_answer_found': prev_msg.successful}
            )
        else:
            # No answer yet, keep tool calling
            try:
                tool_name, tool_args = prev_msg.action, prev_msg.args
                tool_args = tool_args.strip().strip('`').strip()
                if tool_args.startswith('json'):
                    tool_args = tool_args[4:].strip()

                try:
                    tool_args = json.loads(tool_args)
                except JSONDecodeError:
                    tool_args = json_repair.json_repair.loads(tool_args)

                if tool_name in self.tool_names:
                    result = self.tool_name_to_func[tool_name](**tool_args)
                    self.add_to_history(ChatMessage(role='tool', content=result))
                    yield self.response(
                        rtype='step',
                        value=result,
                        channel='_act',
                        metadata={'tool': tool_name, 'args': tool_args}
                    )
                else:
                    result = (
                        f'Incorrect tool name generated: {tool_name}!'
                        ' Please suggest a correct tool name from the provided list.'
                    )
                    yield self.response(
                        rtype='step',
                        value=result,
                        channel='_act',
                        metadata={'is_error': True}
                    )

            except Exception as ex:
                error_msg = f'*** An error occurred while taking the suggested action: {ex}'
                yield self.response(
                    rtype='step',
                    value=error_msg,
                    channel='_act',
                    metadata={'is_error': True}
                )

    def format_messages_for_prompt(self, start_idx: int = 0) -> str:
        """
        Generate a formatted string based on the historical messages for the ReAct agent.

        Args:
            start_idx: The start index of messages to consider (default 0).

        Returns:
            A formatted string containing the messages.
        """
        history = ''

        for msg in self.messages[start_idx:]:
            if msg.role == 'assistant' and isinstance(msg, ReActChatMessage):
                history += f'Thought: {msg.thought}\n'
                history += f'Action: {msg.action}\n'
                history += f'Args: {msg.args}\n'
            elif msg.role == 'tool':
                history += f'Observation: {msg.content}\n\n'

        return history


# The environments where LLM-generated code can be executed
CODE_ENV_NAMES = Literal['host', 'docker', 'e2b']


class CodeRunner:
    """
    Run Python code generated by an LLM in a given environment.
    """
    def __init__(
            self,
            env: CODE_ENV_NAMES,
            allowed_imports: list[str],
            pip_packages: Optional[str] = None,
            timeout: int = 30,
            env_vars_to_set: Optional[dict[str, str]] = None
    ):
        """
        Create an environment to run Python code.

        Args:
            env: The code execution environment. Must be a string from `CODE_ENV_NAMES`.
            allowed_imports: A list of Python modules that are allowed to be imported.
            pip_packages: Optional Python libs to be installed by `pip` [E2B].
            timeout: Code execution timeout (default 30s).
            env_vars_to_set: Optional environment variables to set in the code execution
             environment (E2B only).
        """
        self.allowed_imports: set[str] = set(allowed_imports)
        self.env: CODE_ENV_NAMES = env
        self.pip_packages: list[str] = re.split('[,;]', pip_packages) if pip_packages else []
        self.default_timeout = timeout
        self.local_modules_to_copy = ['kutils.py']
        self.pip_packages_str = ' '.join(self.pip_packages)
        self.env_vars_to_set = env_vars_to_set

    def check_imports(self, code) -> set[Union[str]]:
        """
        Check whether there is any module imported in a given source code outside the allowed
        Python modules.

        Args:
            code: The source code to scan.

        Returns:
            A (possibly empty) set of module names that are disallowed.
        """
        tree = ast.parse(code)
        imported_modules = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module)

        # Find any disallowed imports
        disallowed = imported_modules - self.allowed_imports

        return disallowed

    def run(self, source_code: str) -> tuple[str, str, int]:
        """
        Run Python code in a pre-specified environment.
        Do not return the stdout or stderr as `None`, since that would get converted to the string
        'None'. Instead, set them to empty strings when required.

        Args:
            source_code: The Python code to run.

        Returns:
            The stdout, stderr, and the process return code (0 if no error).
        """
        try:
            ast.parse(source_code)
        except SyntaxError as se:
            return (
                '',
                f'Code parsing failed due to: {type(se).__name__}\n{se.text}\nError: {str(se)}',
                -1
            )

        disallowed_imports: set = self.check_imports(source_code)
        if len(disallowed_imports) > 0:
            return (
                '',
                f'The following imports are disallowed: {disallowed_imports}'
                f'\nPlease only use the allowed modules for importing.',
                -1
            )

        if self.env == 'host':
            warnings.warn(
                'You are running LLM-generated code on your host. This could be potentially'
                ' dangerous! Please consider using a different code runner environment.',
                UserWarning
            )
            with tempfile.NamedTemporaryFile(
                    mode='w+t', suffix='.py', delete=False, encoding='utf-8'
            ) as code_file:
                code_file.write(source_code)
                code_file.close()  # Close the file before execution

                # Copy the local dependency modules
                for a_file in self.local_modules_to_copy:
                    shutil.copy2(
                        os.path.join(os.path.dirname(__file__), a_file),
                        tempfile.gettempdir()
                    )

                result = sp.run(
                    [sys.executable, code_file.name],
                    shell=False, capture_output=True, text=True,
                    timeout=self.default_timeout,
                    check=False,
                    encoding='utf-8'
                )
                os.remove(code_file.name)
                return result.stdout, result.stderr, result.returncode

        elif self.env =='e2b':
            # Run the code on an E2B sandbox
            try:
                import e2b_code_interpreter as e2b
            except ModuleNotFoundError:
                logger.critical(
                    'The module `e2b_code_interpreter` was not found. Please install E2B as:'
                    ' `pip install e2b-code-interpreter`\nExecution will halt now.'
                )
                sys.exit(-1)

            running_sandboxes = e2b.Sandbox.list()
            logger.info('%d E2B sandboxes are running', len(running_sandboxes))
            if running_sandboxes:
                sbx = e2b.Sandbox.connect(running_sandboxes[0].sandbox_id)
            else:
                sbx = e2b.Sandbox(
                    timeout=self.default_timeout + 15,
                    envs=self.env_vars_to_set or {},
                )
                if self.pip_packages_str:
                    sbx.commands.run(f'pip install {self.pip_packages_str}')

            # Copy the local dependency modules
            for a_file in self.local_modules_to_copy:
                with open(
                        os.path.join(os.path.dirname(__file__), a_file),
                        'r',
                        encoding='utf-8'
                ) as py_file:
                    sbx.files.write(f'/home/user/{a_file}', py_file.read())
                    logger.info('Copied file %s...', a_file)

            logger.info('E2B sandbox info: %s', sbx.get_info())
            execution = sbx.run_code(code=source_code, timeout=self.default_timeout)
            std_out: str = '\n'.join(execution.logs.stdout)
            std_err: str = '\n'.join(execution.logs.stderr)
            ret_code: int = -1 if execution.error else 0
            return std_out, std_err, ret_code

        else:
            raise ValueError(f'Unsupported code execution env: {self.env}')


class CodeActAgent(ReActAgent):
    """
    CodeAct is somewhat like ReAct but uses the Thought-Code-Observation loop rather than
    the Thought-Action-Observation loop. In the TCO loop, Python code is written to invoke
    tools, print & capture the results, and observe the results.

    CodeActAgent will retain most of the functionality from ReActAgent. Only the prompt formatting,
    `_think(), and the `_act()` steps will change.
    """
    def __init__(
            self,
            name: str,
            model_name: str,
            run_env: CODE_ENV_NAMES,
            tools: Optional[list[Callable]] = None,
            description: Optional[str] = None,
            vision_model_name: Optional[str] = None,
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
            allowed_imports: Optional[list[str]] = None,
            pip_packages: Optional[str] = None,
            timeout: int = 30,
            env_vars_to_set: Optional[dict[str, str]] = None,
            filter_tools_for_task: bool = False,
            contextual: bool = False,
            use_planning: bool = False,
    ):
        """
        Instantiate a CodeActAgent.

        Args:
            name: The name of the agent.
            description: Description of the agent's capabilities or scope. Recommended to have.
            model_name: The name of the LLM to be used (use names from LiteLLM).
            tools: The tools available to the agent.
            run_env: The code execution environment. `host` means code will be run on the system
             where you create this agent. `e2b` means code will be run on an E2B sandbox. You will
             need an E2B API key.
            litellm_params: Optional parameters for LiteLLM.
            max_iterations: The maximum number of steps that the agent should try to solve a task.
            allowed_imports: A list of Python modules that the agent is allowed to import.
            pip_packages: Optional Python libs to be installed with `pip` [for E2B].
            timeout: Code execution timeout (default 30s).
            env_vars_to_set: Optional environment variables to set in the code execution.
        """
        super().__init__(
            name=name,
            model_name=model_name,
            tools=tools,
            vision_model_name=vision_model_name,
            litellm_params=litellm_params,
            max_iterations=max_iterations,
            description=description,
            filter_tools_for_task=filter_tools_for_task,
            use_planning=use_planning,
        )
        self.contextual = contextual

        # Combine the source code of all tools into one place
        # TODO Somehow dynamically identify and include the modules used by the tools
        self.tools_source_code: str = 'from typing import *\n\nimport kutils as ku\n\n'

        if tools:
            for t in self.tools:
                self.tools_source_code += inspect.getsource(t).replace('@tool\n', '', 1) + '\n'

        self.pip_packages = pip_packages

        if not allowed_imports:
            allowed_imports = []

        # The following imports are allowed by default
        self.allowed_imports = allowed_imports + ['datetime', 'typing', 'mimetypes']
        self.code_runner = CodeRunner(
            env=run_env,
            allowed_imports=self.allowed_imports + ['kutils'],
            pip_packages=pip_packages,
            timeout=timeout,
            env_vars_to_set=env_vars_to_set,
        )

    def format_messages_for_prompt(self, start_idx: int = 0) -> str:
        """
        Generate a formatted string based on the historical messages for the ReAct agent.

        Args:
            start_idx: The start index of messages to consider (default 0).

        Returns:
             A formatted string containing the messages.
        """
        history = ''

        for msg in self.messages[start_idx:]:
            if msg.role == 'assistant' and isinstance(msg, CodeChatMessage):
                history += f'Thought: {msg.thought}\n'
                code = msg.code.strip()
                if not code.startswith('```py'):
                    code = f'```py\n{code}'
                if not code.endswith('```'):
                    code = f'{code}\n```'
                history += f'Code:{code}\n'
            elif msg.role == 'tool':
                history += f'Observation: {msg.content}\n\n'

        return history

    async def _think(self, plan: Optional[str] = None) -> AsyncIterator[AgentResponse]:
        """
        Think about the next step to be taken to solve the given task.

        The LLM is prompted with the available tools and the TCO sequence so far. Based on them,
        the LLM will suggest the next action/code.

        Args:
            plan: A tentative plan to solve the task [Optional].

        Yields:
            Update from the thing step.
        """
        if self.filter_tools_for_task:
            relevant_tools = await self.get_relevant_tools(
                task_description=self.task.description, task_files=self.task.files
            )
        else:
            relevant_tools = self.tools

        if self.contextual:
            history_summary = await self.get_history_summary()
            prompt_template = CODE_ACT_AGENT_CONTEXTUAL_PROMPT
            history_kwargs = {'history_summary': history_summary}
        else:
            prompt_template = CODE_ACT_AGENT_PROMPT
            history_kwargs = {
                'history': self.format_messages_for_prompt(start_idx=self.msg_idx_of_new_task)
            }

        message = prompt_template.format(
            task=self.task.description,
            task_files='\n'.join(self.task.files) if self.task.files else '[None]',
            tool_names=self.get_tools_description(relevant_tools),
            authorized_imports=','.join(self.allowed_imports),
            plan=plan or '[No plan provided; please plan yourself]',
            **history_kwargs,
        )
        msg = await self._record_thought(message, CodeChatMessage)
        yield self.response(rtype='step', value=msg, channel='_think')

    async def _act(self) -> AsyncIterator[AgentResponse]:
        """
        Code action based on CodeActAgent's previous thought.

        The LLM has suggested code. This method will run the code.

        Yields:
            Updates from the acting step.
        """
        prev_msg: CodeChatMessage = self.messages[-1]  # type: ignore

        if (
                not prev_msg.answer and (not prev_msg.code or not prev_msg.thought)
        ):
            self.add_to_history(
                ChatMessage(
                    role='tool',
                    content=(
                        '* Error: incorrect response generated. Must have values for the `answer`'
                        ' or the `action`, `args`, and `thought` fields.'
                    )
                )
            )
            return

        if prev_msg.answer:
            # The final answer has been found!
            self.final_answer_found = True
            self.task.is_finished = True
            self.task.is_error = prev_msg.successful
            response_msg = ChatMessage(role='assistant', content=prev_msg.answer)
            self.add_to_history(response_msg)

            yield self.response(
                rtype='final',
                value=response_msg,
                channel='_act',
                metadata={'final_answer_found': prev_msg.successful}
            )
        else:
            # No answer yet, keep tool calling
            try:
                code = prev_msg.code.strip()
                code = code.replace('```py', '')
                code = code.replace('```', '').strip()
                code = f'{self.tools_source_code}\n\n{code}'

                stdout, stderr, exit_status = self.code_runner.run(code)
                observation = f'{stdout}\n{stderr}'.strip()
                msg = ChatMessage(role='tool', content=observation)
                self.add_to_history(msg)
                yield self.response(
                    rtype='step',
                    value=observation,
                    channel='_act',
                    metadata={'is_error': exit_status != 0}
                )

            except Exception as ex:
                error_msg = f'*** An error occurred while running the code: {ex}'
                yield self.response(
                    rtype='step',
                    value=error_msg,
                    channel='_act',
                    metadata={'is_error': True}
                )


class SupervisorAgent(Agent):
    """
    A supervising agency, consisting of multiple agents, which can solve tasks via delegations.

    WARNING: SupervisorAgent is an experimental feature and may not work as expected.
    """
    def __init__(self, model_name: str, agents: list[Agent], name: str, max_iterations: int = 20):
        """
        Create a supervisor who delegates tasks to a list of agents.

        Args:
            model_name: The name of the LLM to be used.
            agents: A list of agents available to the supervisor.
            name: The name of the supervisor agent.
            max_iterations: The max no. of iterations/attempted delegations made by the supervisor.
        """
        super().__init__(name=name, model_name=model_name, max_iterations=max_iterations)
        self.agents = agents

    async def run(
            self,
            task: str,
            files: Optional[list[str]] = None,
            task_id: Optional[str] = None
    ) -> AsyncIterator[AgentResponse]:
        """
        Solve a task using the supervisor agent/agency.

        Args:
            task: A description of the task.
            files: An optional list of file paths or URLs.
            task_id: (Optional) An ID for the task, if provided by the caller.

        Yields:
            An update from the agency.
        """
        self._run_init(task, files, task_id)

        agents_desc = ''
        for idx, agent in enumerate(self.agents):
            agents_desc += f'Agent# {idx}\n{agent.purpose}\n'

        for _ in range(self.max_iterations):
            prompt = SUPERVISOR_TASK_PROMPT.format(
                task=task,
                task_files=files,
                agents=agents_desc,
                history=self.get_history(start_idx=self.msg_idx_of_new_task),
            ).strip()
            update = await self._call_llm(
                ku.make_user_message(text_content=prompt, files=files),
                SupervisorTaskMessage,
                trace_id=self.task.id,
            )
            task_msg: SupervisorTaskMessage = SupervisorTaskMessage.model_validate_json(update)

            if task_msg.task_complete:
                # Currently, this is the ONLY way to stop the supervisor's loop
                # In some cases, it is possible that `task_complete` is not set, and the supervisor
                # keeps repeating the same task
                yield AgentResponse(
                    type='final',
                    channel='supervisor',
                    value=task_msg.final_answer,
                    metadata=None
                )
                self.add_to_history(
                    message=ChatMessage(role='assistant', content=task_msg.final_answer)
                )
                return

            content = (
                f'Delegating to Agent# {task_msg.agent_id} //'
                f' Sub-task: {task_msg.task} //'
                f' Files: {task_msg.image_files} //'
                f' Facts collected: {task_msg.facts_available}'
            )
            yield AgentResponse(type='log', channel='sup:run', value=content, metadata=None)
            self.add_to_history(message=ChatMessage(role='assistant', content=content))

            updates: list[str] = []
            tools_evidence: list[tuple[str, dict]] = []

            async for update in self.agents[task_msg.agent_id].run(  # type: AgentResponse
                    task=task_msg.task,
                    files=task_msg.image_files
            ):
                yield update

                if update['type'] == 'final':
                    yield AgentResponse(
                        type='step',
                        channel='supervisor',
                        value=update['value'],
                        metadata=None
                    )
                    updates = [update['value']]
                elif update['type'] == 'step':
                    updates.append(update['value'])
                    metadata = update['metadata']
                    if metadata:
                        tools_evidence.append(
                            (metadata.get('tool', None), metadata.get('args', None))
                        )

            updates = [
                f'Agent\'s attempt {idx}: {v.content if isinstance(v, ChatMessage) else v}'
                for idx, v in enumerate(updates, start=1)
            ]
            evidence = '\n'.join(updates)
            tools_evidence: list[tuple[str, dict]] = [(t, a) for (t, a) in tools_evidence if t]
            evidence += '\n\nTool usage evidence by the agent:\n'
            evidence += '\n'.join([f'Tool: {t} // args: {a}' for (t, a) in tools_evidence])
            status = await self._check_if_task_done(task_msg.task, evidence)
            if status.status:
                self.add_to_history(
                    message=ChatMessage(
                        role='user',
                        content=(
                            f'I can accept the result: `{status.final_answer}`'
                            '\nYou can proceed to the next subtask, if any.'
                        )
                    )
                )
            else:
                feedback = (
                    f'\nThe result does not look good: {status.final_answer}'
                    f'## Why the task result is incomplete:\n{status.reason}'
                    f'\n## Here\'s what can be done to fix it:\n{status.how_to_fix}'
                )
                self.add_to_history(
                    message=ChatMessage(role='user', content='\n'.join(updates) + feedback)
                )
        # END of supervisor loop
        # The supervisor has exhausted all attempts, but a final answer was not found/returned
        async for update in self._salvage_response():
            yield update

    async def _check_if_task_done(self, task: str, evidence: str) -> DelegatedTaskStatus:
        """
        Check if a task delegated to an agent has reached completion.

        Args:
            task: Delegated task description.
            evidence: Evidence (the steps performed so far).

        Return:
             The delegated task status.
        """
        prompt = SUPERVISOR_TASK_CHECK_PROMPT.format(task=task, response=evidence)
        response = await self._call_llm(
            ku.make_user_message(prompt),
            DelegatedTaskStatus,
            trace_id=self.task.id,
        )

        return DelegatedTaskStatus.model_validate_json(response)

    async def _salvage_response(self):
        """
        The supervisor has failed to return an answer in stipulated number of steps. This is
        a final result to save face and try salvage what little information could be!
        """
        prompt = SALVATION_PROMPT.format(
            task=self.task,
            task_files=self.task.files,
            history=self.get_history()
        )
        response = await self._call_llm(ku.make_user_message(prompt), trace_id=self.task.id)
        yield AgentResponse(
            type='final',
            channel='supervisor',
            value=response,
            metadata={'salvage': True}
        )


def llm_vision_support(model_names: list[str]) -> list[bool]:
    """
    Utility function to check whether images can be used with given LLMs.

    Args:
        model_names: A list of LLM names.

    Returns:
        A list of booleans, containing `True` or `False` for each model.
    """
    status = [litellm.supports_vision(model=model) for model in model_names]
    for model, value in zip(model_names, status):
        print(f'- Vision supported by {model}: {value}')

    return status


def print_response(response: AgentResponse):
    """
    A utility function to print agent's response in a terminal, optionally with colors.

    Args:
        response: A response obtained from an agent.
    """

    if response['type'] == 'final':
        msg = (
            response['value'].content
            if isinstance(response['value'], ChatMessage) else response['value']
        )
        rich.print(f'[blue][bold]Agent[/bold]: {msg}[/blue]\n')
    elif response['type'] == 'log':
        rich.print(f'[white]{response}[/white]')
    else:
        rich.print(f'{response}')


async def main():
    """
    Demonstrate the use of ReActAgent and CodeActAgent.
    """
    litellm_params = {'temperature': 0}
    model_name = 'gemini/gemini-2.5-flash-lite'
    # model_name = 'vertex_ai/gemini-2.0-flash'
    # model_name = 'azure/gpt-4.1-mini'
    # model_name = 'azure/gpt-4o-mini'

    react_agent = ReActAgent(
        name='Maths agent',
        model_name=model_name,
        tools=[calculator, ],
        max_iterations=3,
        litellm_params=litellm_params,
        filter_tools_for_task=True,
    )
    code_agent = CodeActAgent(
        name='Web agent',
        model_name=model_name,
        tools=[web_search, extract_as_markdown, file_download, get_youtube_transcript],
        run_env='host',
        max_iterations=6,
        litellm_params=litellm_params,
        allowed_imports=[
            'os', 're', 'time', 'random', 'requests', 'tempfile',
            'ddgs', 'markitdown', 'youtube_transcript_api',
        ],
        pip_packages='ddgs~=9.5.2;"markitdown[all]";',
        filter_tools_for_task=False,
        contextual=False,
        use_planning=True
    )

    the_tasks = [
        ('What is ten plus 15, raised to 2, expressed in words?', None),
        ('What is the date today? Express it in words.', None),
        (
            'Which image has a purple background?',
            [
                'https://www.slideteam.net/media/catalog/product/cache/1280x720/p/r/process_of_natural_language_processing_training_ppt_slide01.jpg',
                'https://cdn.prod.website-files.com/61a05ff14c09ecacc06eec05/66e8522cbe3d357b8434826a_ai-agents.jpg',
            ]
        ),
        (
            'What is four plus seven? Also, what are the festivals in Paris?'
            ' How they differ from Kolkata?',
            None
        ),
        (
            'Summarize the notes',
            ['https://web.stanford.edu/class/cs102/lectureslides/ClassificationSlides.pdf',]
        ),
    ]

    print('ContextualAgent demo\n')
    for task, img_urls in the_tasks:
        rich.print(f'[yellow][bold]User[/bold]: {task}[/yellow]')
        # await code_agent.get_relevant_tools(task_description=task, task_files=img_urls)
        async for response in code_agent.run(task, files=img_urls):
            print_response(response)
        print('\n\n')


if __name__ == '__main__':
    # For Windows; in case of Unicode error with PDF extraction
    os.environ['PYTHONUTF8'] = '1'

    asyncio.run(main())
