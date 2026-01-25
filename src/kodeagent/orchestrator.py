"""Planner and Observer for Agent orchestration."""

from typing import Literal

from . import kutils as ku
from . import tracer
from .models import AgentPlan, ObserverResponse, PlanStep, Task
from .usage_tracker import UsageTracker

logger = ku.get_logger()

PLANNER_SYSTEM_PROMPT = ku.read_prompt('system/planner.txt')
PLAN_UPDATER_SYSTEM_PROMPT = ku.read_prompt('system/plan_updater.txt')
AGENT_PLAN_PROMPT = ku.read_prompt('agent_plan.txt')
UPDATE_PLAN_PROMPT = ku.read_prompt('update_plan.txt')
OBSERVER_SYSTEM_PROMPT = ku.read_prompt('system/observer.txt')
OBSERVATION_PROMPT = ku.read_prompt('observation.txt')


class Planner:
    """Given a task, generate and maintain a step-by-step plan to solve it."""

    def __init__(
        self,
        model_name: str,
        litellm_params: dict | None = None,
        max_retries: int = ku.DEFAULT_MAX_LLM_RETRIES,
        usage_tracker: UsageTracker | None = None,
        tracer_manager: tracer.AbstractTracerManager | None = None,
    ):
        """Create a planner using the given model.

        Args:
            model_name: The name of the LLM to use.
            litellm_params: LiteLLM parameters.
            max_retries: Maximum number of retries for LLM calls.
            usage_tracker: Optional UsageTracker instance to record usage.
            tracer_manager: Optional AbstractTracerManager for hierarchical tracing.
        """
        self.model_name = model_name
        self.litellm_params = litellm_params or {}
        self.max_retries = max_retries
        self.usage_tracker = usage_tracker
        self.tracer_manager = tracer_manager or tracer.NoOpTracerManager()
        self.plan: AgentPlan | None = None

    async def create_plan(
        self,
        task: Task,
        agent_type: str,
        parent_trace: tracer.AbstractObservation | None = None,
    ) -> AgentPlan:
        """Create a plan to solve the given task and store it.

        Args:
            task: The task to solve.
            agent_type: Type of the agent that would solve the task.
            parent_trace: Optional parent observation for hierarchical tracing.

        Returns:
            A plan to solve the task.
        """
        # Create tracing span
        parent = parent_trace or tracer.NoOpObservation()
        plan_span = self.tracer_manager.start_span(
            parent=parent,
            name='plan_creation',
            input_data={
                'agent_type': agent_type,
                'task_id': str(task.id),
                'task_description': task.description,
                'file_count': len(task.files) if task.files else 0,
            },
        )

        messages = ku.make_user_message(
            text_content=AGENT_PLAN_PROMPT.format(
                agent_type=agent_type,
                task=task.description,
                task_files='\n'.join(task.files) if task.files else '[None]',
            ),
            files=task.files,
        )
        messages = [{'role': 'system', 'content': PLANNER_SYSTEM_PROMPT}, *messages]
        plan_response = await ku.call_llm(
            model_name=self.model_name,
            litellm_params=self.litellm_params,
            messages=messages,
            response_format=AgentPlan,
            trace_id=task.id,
            max_retries=self.max_retries,
            usage_tracker=self.usage_tracker,
            component_name='Planner.create',
        )
        self.plan = AgentPlan.model_validate_json(plan_response)

        # Update trace with results
        plan_span.end(output={'steps': self.get_formatted_plan()})

        return self.plan

    async def update_plan(
        self,
        thought: str,
        observation: str,
        task_id: str,
        parent_trace: tracer.AbstractObservation | None = None,
    ):
        """Update the plan based on the last thought and observation.

        Args:
            thought: The ReAct/CodeAct agent's thought.
            observation: The agent's observation.
            task_id: ID of the task for which the plan is to be updated.
            parent_trace: Optional parent observation for hierarchical tracing.
        """
        if not self.plan:
            return

        # Create tracing span
        parent = parent_trace or tracer.NoOpObservation()
        update_span = self.tracer_manager.start_span(
            parent=parent,
            name='plan_update',
            input_data={
                'task_id': str(task_id),
                'thought_length': len(thought),
                'observation_length': len(observation),
                'current_steps': len(self.plan.steps) if self.plan else 0,
            },
        )

        prompt = UPDATE_PLAN_PROMPT.format(
            plan=self.plan.model_dump_json(indent=2), thought=thought, observation=observation
        )
        messages = [
            {'role': 'system', 'content': PLAN_UPDATER_SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ]
        plan_response = await ku.call_llm(
            model_name=self.model_name,
            litellm_params=self.litellm_params,
            messages=messages,
            response_format=AgentPlan,
            trace_id=task_id,
            max_retries=self.max_retries,
            usage_tracker=self.usage_tracker,
            component_name='Planner.update',
        )
        self.plan = AgentPlan.model_validate_json(plan_response)

        # Update trace with results
        update_span.end(output={'steps': self.get_formatted_plan()})

    def get_steps_done(self) -> list[PlanStep]:
        """Returns the completed steps from the current plan.

        Returns:
            A list of completed PlanStep objects.
        """
        if not self.plan:
            return []
        return [step for step in self.plan.steps if step.is_done]

    def get_steps_pending(self) -> list[PlanStep]:
        """Returns the pending steps from the current plan.

        Returns:
            A list of pending PlanStep objects.
        """
        if not self.plan:
            return []
        return [step for step in self.plan.steps if not step.is_done]

    def get_formatted_plan(self, scope: Literal['all', 'done', 'pending'] = 'all') -> str:
        """Convert the agent's plan into a Markdown checklist."""
        if not self.plan or not self.plan.steps:
            return ''

        if scope == 'all':
            steps_to_format = self.plan.steps
        elif scope == 'done':
            steps_to_format = self.get_steps_done()
        else:  # pending
            steps_to_format = self.get_steps_pending()

        todo_list = []
        for step in steps_to_format:
            status = 'x' if step.is_done else ' '
            todo_list.append(f'- [{status}] {step.description}')
        return '\n'.join(todo_list)

    def reset(self):
        """Reset the planner state."""
        self.plan = None


class Observer:
    """Monitors an agent's behavior to detect issues like loops or stalled plans."""

    def __init__(
        self,
        model_name: str,
        tool_names: set[str],
        litellm_params: dict | None = None,
        threshold: int | None = 3,
        max_retries: int = ku.DEFAULT_MAX_LLM_RETRIES,
        usage_tracker: UsageTracker | None = None,
        tracer_manager: tracer.AbstractTracerManager | None = None,
    ):
        """Create an Observer for an agent.

        Args:
            model_name: The LLM to use.
            tool_names: The set of tools available to the agent.
            litellm_params: LiteLLM parameters.
            threshold: Observation threshold, i.e., how frequently the observer will analyze
             the chat history.
            max_retries: Maximum number of retries for LLM calls.
            usage_tracker: Optional UsageTracker instance to record usage.
            tracer_manager: Optional AbstractTracerManager for hierarchical tracing.
        """
        self.threshold = threshold
        self.model_name = model_name
        self.tool_names = tool_names
        self.litellm_params = litellm_params or {}
        self.max_retries = max_retries
        self.usage_tracker = usage_tracker
        self.tracer_manager = tracer_manager or tracer.NoOpTracerManager()
        self.last_correction_iteration: int = 0

    async def observe(
        self,
        iteration: int,
        task: Task,
        history: str,
        plan_before: str | AgentPlan | None,
        plan_after: str | AgentPlan | None,
        parent_trace: tracer.AbstractObservation | None = None,
    ) -> str | None:
        """Observe the agent's state and return a corrective message if a problem is detected.

        Args:
            iteration: The current iteration of the agent.
            task: The task being solved by the agent.
            history: Task progress history (LLM chat history).
            plan_before: The agent's plan before this iteration.
            plan_after: The updated plan.
            parent_trace: Optional parent observation for hierarchical tracing.

        Returns:
            Optional correction message for the agent (LLM), e.g., what to do or avoid.
        """
        if self.threshold is None or iteration <= 1:
            return None
        if iteration - self.last_correction_iteration < self.threshold:
            return None

        # Create tracing span
        parent = parent_trace or tracer.NoOpObservation()
        observe_span = self.tracer_manager.start_span(
            parent=parent,
            name='observe',
            input_data={
                'iteration': iteration,
                'task_id': str(task.id),
                'history_length': len(history),
                'tool_count': len(self.tool_names),
            },
        )

        try:
            tool_names = '\n'.join(sorted(list(self.tool_names)))
            prompt = OBSERVATION_PROMPT.format(
                task=task.description,
                plan_before=plan_before,
                plan_after=plan_after,
                history=history,
                tools=tool_names,
            )
            observation_response = await ku.call_llm(
                model_name=self.model_name,
                litellm_params=self.litellm_params,
                messages=[
                    {'role': 'system', 'content': OBSERVER_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                max_retries=self.max_retries,
                response_format=ObserverResponse,
                usage_tracker=self.usage_tracker,
                component_name='Observer',
            )
            observation = ObserverResponse.model_validate_json(observation_response)

            if not observation.is_progressing or observation.is_in_loop:
                self.last_correction_iteration = iteration
                msg = (
                    observation.correction_message
                    or observation.reasoning
                    or 'Adjust your approach based on the plan and history.'
                )
                correction = f'!!!CRITICAL FOR COURSE CORRECTION: {msg}\n'

                if self.tool_names:
                    correction += (
                        f'Here are the exact TOOL names once again for reference:\n{tool_names}'
                    )

                # Update trace with correction findings
                observe_span.end(
                    output={
                        'is_progressing': observation.is_progressing,
                        'is_in_loop': observation.is_in_loop,
                        'correction_issued': True,
                        'observation': msg,
                    }
                )

                return correction

            # No issue detected
            observe_span.end(
                output={
                    'is_progressing': observation.is_progressing,
                    'is_in_loop': observation.is_in_loop,
                    'correction_issued': False,
                }
            )

        except Exception as e:
            logger.exception('LLM Observer failed: %s', str(e))
            observe_span.update(status='error', error=str(e))
            observe_span.end(is_error=True)
            return None

        return None

    def reset(self):
        """Reset the observer state."""
        self.last_correction_iteration = 0
