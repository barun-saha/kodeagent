"""Interfaces for hierarchical tracing and agent observability.

Supports multiple observability backends (e.g., Langfuse and LangSmith) with a unified API for
creating traces, spans, and generations. Provides no-op implementations when tracing is disabled.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

TRACING_TYPES = Literal['langfuse', 'langsmith']


def create_tracer_manager(tracing_type: TRACING_TYPES | None = None) -> 'AbstractTracerManager':
    """Factory function to create a tracer manager based on the specified type.

    Args:
        tracing_type: The type of tracing backend to use. Defaults to None for no-op tracing.

    Returns:
        An instance of LangfuseTracerManager, LangSmithTracerManager, or NoOpTracerManager
        for the specified tracing backend.
    """
    if tracing_type == 'langfuse':
        return LangfuseTracerManager()
    if tracing_type == 'langsmith':
        return LangSmithTracerManager()
    return NoOpTracerManager()


class AbstractObservation(ABC):
    """Abstract interface for trace observations.

    Represents a single node in a hierarchical trace tree. Can be a top-level
    trace, a nested span, or an LLM generation. Implements context manager
    protocol for use with 'with' statements.
    """

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Update observation properties during execution.

        Used to log intermediate states like partial outputs, status, or
        metadata.

        Args:
            **kwargs: Provider-specific properties (output, status, level, etc).
        """

    @abstractmethod
    def end(self, **kwargs: Any) -> None:
        """Explicitly signal the end of the observation.

        Records final state and duration. Called automatically when using
        context manager protocol.

        Args:
            **kwargs: Provider-specific properties (output, result, error, etc).
        """

    def __enter__(self) -> 'AbstractObservation':
        """Context manager entry: return self for 'with' statement."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit: automatically end observation."""
        self.end()


class AbstractTracerManager(ABC):
    """Abstract interface for tracer management.

    Factory for creating hierarchical observations. Handles initialization and
    backend-specific configuration. Implementations should support at least one
    tracing backend (Langfuse, LangSmith, etc.) or be a no-op when tracing is
    disabled.
    """

    @abstractmethod
    def start_trace(self, name: str, input_data: Any) -> AbstractObservation:
        """Start a new top-level trace.

        Args:
            name: Identifier for the trace operation.
            input_data: Input data to log for the trace.

        Returns:
            An observation object for the trace root.
        """

    @abstractmethod
    def start_span(
        self, parent: AbstractObservation, name: str, input_data: Any
    ) -> AbstractObservation:
        """Start a nested span under a parent observation.

        Used for logical sub-operations within a trace or parent span.

        Args:
            parent: Parent observation (trace or span).
            name: Identifier for the span operation.
            input_data: Input data to log for the span.

        Returns:
            An observation object for the span.
        """

    @abstractmethod
    def start_generation(
        self, parent: AbstractObservation, name: str, input_data: Any
    ) -> AbstractObservation:
        """Start a nested LLM generation under a parent observation.

        Used specifically for LLM calls within a trace or span.

        Args:
            parent: Parent observation (trace or span).
            name: Identifier for the generation operation.
            input_data: Input data (e.g., prompt) to log for the generation.

        Returns:
            An observation object for the generation.
        """

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered traces to the backend.

        Ensures that all recorded traces and spans are sent to the observability platform
        before the application exits.
        """


class NoOpObservation(AbstractObservation):
    """No-op observation implementation.

    Used when tracing is disabled. All methods are no-ops and return self
    to support hierarchical nesting without side effects.
    """

    def update(self, **kwargs: Any) -> None:
        """No-op: ignore all property updates."""

    def end(self, **kwargs: Any) -> None:
        """No-op: ignore end signal."""


class NoOpTracerManager(AbstractTracerManager):
    """No-op tracer manager implementation.

    Used when no observability backend is enabled. Provides a complete no-op
    implementation of the TracerManager interface that satisfies the contract
    while performing no actual tracing operations.
    """

    def start_trace(self, name: str, input_data: Any) -> AbstractObservation:
        """No-op: return a no-op observation."""
        return NoOpObservation()

    def start_span(
        self, parent: AbstractObservation, name: str, input_data: Any
    ) -> AbstractObservation:
        """No-op: return a no-op observation."""
        return NoOpObservation()

    def start_generation(
        self, parent: AbstractObservation, name: str, input_data: Any
    ) -> AbstractObservation:
        """No-op: return a no-op observation."""
        return NoOpObservation()

    def flush(self) -> None:
        """No-op: do nothing."""


class LangfuseObservation(AbstractObservation):
    """Langfuse implementation of observation.

    Wraps Langfuse Trace, Span, or Generation objects to provide a consistent
    interface.
    """

    def __init__(self, obj: Any) -> None:
        """Initialize Langfuse observation.

        Args:
            obj: The Langfuse Trace, Span, or Generation object.
        """
        self.obj = obj

    def update(self, **kwargs: Any) -> None:
        """Update observation properties.

        Args:
            **kwargs: Properties to update.
        """
        if hasattr(self.obj, 'update'):
            self.obj.update(**kwargs)

    def end(self, **kwargs: Any) -> None:
        """End the observation.

        Maps 'result' to 'output' for compatibility with Langfuse.
        Calls end() if available (Spans/Generations), else update() (Traces).

        Args:
            **kwargs: Final state data.
        """
        # Map 'result' to 'output' if present
        if 'result' in kwargs and 'output' not in kwargs:
            kwargs['output'] = kwargs.pop('result')

        if hasattr(self.obj, 'end'):
            self.obj.end(**kwargs)
        elif hasattr(self.obj, 'update'):
            self.obj.update(**kwargs)


class LangfuseTracerManager(AbstractTracerManager):
    """Langfuse implementation of TracerManager.

    Integrates with Langfuse observability platform to create and manage hierarchical traces,
    spans, and generations. Tracing is disabled if the Langfuse package is not installed.
    """

    def __init__(self) -> None:
        """Initialize the Langfuse client."""
        try:
            from langfuse.client import Langfuse

            self.client: Any = Langfuse()
        except ImportError:
            logging.error(
                'Langfuse package is not installed. Please install langfuse to use'
                ' LangfuseTracerManager. Tracing will be disabled.'
            )
            self.client = None

    def start_trace(self, name: str, input_data: Any) -> AbstractObservation:
        """Start a new trace with Langfuse.

        Args:
            name: Identifier for the trace operation.
            input_data: Input data to log for the trace.

        Returns:
            A Langfuse trace object wrapped as AbstractObservation.
        """
        if not self.client:
            return NoOpObservation()
        return LangfuseObservation(self.client.trace(name=name, input=input_data))

    def start_span(
        self, parent: AbstractObservation, name: str, input_data: Any
    ) -> AbstractObservation:
        """Start a nested span under a parent observation with Langfuse.

        Args:
            parent: Parent observation (assumes LangfuseObservation).
            name: Identifier for the span operation.
            input_data: Input data to log for the span.

        Returns:
            A Langfuse span object wrapped as AbstractObservation.
        """
        if not self.client:
            return NoOpObservation()

        # Unwrap if it's our LangfuseObservation
        parent_obj = parent.obj if isinstance(parent, LangfuseObservation) else parent

        if hasattr(parent_obj, 'span'):
            return LangfuseObservation(parent_obj.span(name=name, input=input_data))
        return NoOpObservation()

    def start_generation(
        self, parent: AbstractObservation, name: str, input_data: Any
    ) -> AbstractObservation:
        """Start a nested LLM generation under a parent observation with Langfuse.

        Args:
            parent: Parent observation (assumes LangfuseObservation).
            name: Identifier for the generation operation.
            input_data: Input data (e.g., prompt) to log for the generation.

        Returns:
            A Langfuse generation object wrapped as AbstractObservation.
        """
        if not self.client:
            return NoOpObservation()

        # Unwrap if it's our LangfuseObservation
        parent_obj = parent.obj if isinstance(parent, LangfuseObservation) else parent

        if hasattr(parent_obj, 'generation'):
            return LangfuseObservation(parent_obj.generation(name=name, input=input_data))
        return NoOpObservation()

    def flush(self) -> None:
        """Flush Langfuse traces."""
        if self.client:
            self.client.flush()


class LangSmithObservation(AbstractObservation):
    """LangSmith implementation of observation.

    Wraps a LangSmith RunTree object to manage hierarchical runs.
    """

    def __init__(self, run_tree: Any) -> None:
        """Initialize LangSmith observation.

        Args:
            run_tree: The LangSmith RunTree object.
        """
        self.run_tree = run_tree
        self.ended = False

        # Post the run to LangSmith immediately
        try:
            self.run_tree.post()
        except Exception as e:
            logging.error('Error creating run in LangSmith: %s', e)

    def update(self, **kwargs: Any) -> None:
        """Update observation outputs during execution.

        Args:
            **kwargs: Output data to accumulate.
        """
        if not self.ended:
            if 'error' in kwargs:
                self.run_tree.error = kwargs['error']
            # We don't need to manually verify other updates as RunTree handles finalization on end()
            # or we can patch if strictly needed, but post() is usually sufficient for start.

    def end(self, **kwargs: Any) -> None:
        """End the observation and send final data to LangSmith.

        Args:
            **kwargs: Final output/result data.
        """
        if not self.ended:
            self.ended = True

            outputs = kwargs.get('output')
            error = kwargs.get('error')
            metadata = kwargs.get('metadata')

            # Ensure outputs is a dict for RunTree.patch() which calls outputs.copy()
            if outputs is not None and not isinstance(outputs, dict):
                outputs = {'output': outputs}

            # Map 'result' to 'outputs' if provided in result keyword
            if 'result' in kwargs and outputs is None:
                res = kwargs.pop('result')
                outputs = res if isinstance(res, dict) else {'output': res}

            try:
                if metadata:
                    self.run_tree.add_metadata(metadata)

                self.run_tree.end(outputs=outputs, error=error)
                self.run_tree.patch()
            except Exception as e:
                logging.exception('Error updating run in LangSmith: %s', e)


class LangSmithTracerManager(AbstractTracerManager):
    """LangSmith implementation of TracerManager.

    Uses LangSmith RunTree to manage hierarchical runs.
    """

    def __init__(self) -> None:
        """Initialize the LangSmith client."""
        try:
            from langsmith import Client

            self.client: Any = Client()
        except ImportError:
            logging.error(
                'LangSmith package is not installed. Please install langsmith to use'
                ' LangSmithTracerManager. Tracing will be disabled.'
            )
            self.client = None

    def start_trace(self, name: str, input_data: Any) -> AbstractObservation:
        """Start a new trace with LangSmith."""
        if not self.client:
            return NoOpObservation()

        try:
            from langsmith.run_trees import RunTree

            run_tree = RunTree(name=name, run_type='chain', inputs=input_data, client=self.client)
            return LangSmithObservation(run_tree)
        except ImportError:
            return NoOpObservation()

    def start_span(
        self, parent: AbstractObservation, name: str, input_data: Any
    ) -> AbstractObservation:
        """Start a nested span under a parent observation.

        Args:
            parent: Parent observation (assumes LangSmithObservation).
            name: Identifier for the span operation.
            input_data: Input data to log for the span.

        Returns:
            A LangSmith span object wrapped as AbstractObservation.
        """
        if not self.client or not isinstance(parent, LangSmithObservation):
            return NoOpObservation()

        child_run = parent.run_tree.create_child(name=name, run_type='tool', inputs=input_data)
        return LangSmithObservation(child_run)

    def start_generation(
        self, parent: AbstractObservation, name: str, input_data: Any
    ) -> AbstractObservation:
        """Start a nested LLM generation.

        Args:
            parent: Parent observation (assumes LangSmithObservation).
            name: Identifier for the generation operation.
            input_data: Input data (e.g., prompt) to log for the generation.

        Returns:
            A LangSmith generation object wrapped as AbstractObservation.
        """
        if not self.client or not isinstance(parent, LangSmithObservation):
            return NoOpObservation()

        child_run = parent.run_tree.create_child(name=name, run_type='llm', inputs=input_data)
        return LangSmithObservation(child_run)

    def flush(self) -> None:
        """Flush LangSmith runs."""
        if self.client and hasattr(self.client, 'flush'):
            self.client.flush()
