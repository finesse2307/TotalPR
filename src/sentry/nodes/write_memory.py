"""write_memory node: persist each finding as an unlabeled memory.

Runs after critique, before format_review. When a ``MemoryStore`` is
provided, the findings produced by the critique are stored as a batch via
``MemoryStore.store_findings`` — one embedding call per agent run, not per
finding — with ``was_accepted=NULL``. A human (or, in Phase 4, GitHub
thread-resolution data) labels them later via ``scripts/label_memory.py``.

When no ``MemoryStore`` is provided, the node is a no-op — graph topology
stays constant whether memory is configured or not.
"""

from collections.abc import Callable
from typing import Any

from sentry.memory import MemoryStore
from sentry.state import AgentState


def make_write_memory_node(
    memory: MemoryStore | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    """Build a write_memory node, optionally bound to a MemoryStore.

    When ``memory`` is ``None``, the returned function is a no-op returning
    an empty dict so the node can stay in the graph unconditionally.
    """

    def write_memory_node(state: AgentState) -> dict[str, Any]:
        if memory is None or not state.findings:
            return {}

        memory.store_findings(
            repo=state.pr.repo,
            diff_text=state.raw_diff,
            findings=state.findings,
            was_accepted=None,
        )
        return {}

    return write_memory_node