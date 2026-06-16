"""Top-level LangGraph wiring for the Sentry code-review agent.

Constructs the linear state machine that drives a single PR review:
    START → parse_diff → plan → run_tool → critique → format_review
          → post_comment → END

The dependencies (LLM client, tool registry, poster) are injected so the same
graph definition is used in tests (with mocks) and in production (with real
backing services).
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sentry.llm import LLMClient
from sentry.nodes.critique import make_critique_node
from sentry.nodes.format_review import format_review
from sentry.nodes.parse_diff import parse_diff
from sentry.nodes.plan import make_plan_node
from sentry.nodes.post_comment import make_post_comment_node
from sentry.nodes.run_tool import ToolRegistry, make_run_tool_node
from sentry.posting import CommentPoster
from sentry.state import AgentState
from sentry.telemetry import with_span


def build_graph(
    *,
    llm: LLMClient,
    tools: ToolRegistry,
    poster: CommentPoster,
) -> CompiledStateGraph:
    """Build and compile the Sentry code-review agent graph.

    Returns a runnable compiled graph; invoke it with
    ``compiled.invoke(initial_state)`` to drive a full review end to end.
    """
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("parse_diff", with_span("parse_diff", parse_diff))
    graph.add_node("plan", with_span("plan", make_plan_node(llm)))
    graph.add_node("run_tool", with_span("run_tool", make_run_tool_node(tools)))
    graph.add_node("critique", with_span("critique", make_critique_node(llm)))
    graph.add_node("format_review", with_span("format_review", format_review))
    graph.add_node("post_comment", with_span("post_comment", make_post_comment_node(poster)))

    graph.add_edge(START, "parse_diff")
    graph.add_edge("parse_diff", "plan")
    graph.add_edge("plan", "run_tool")
    graph.add_edge("run_tool", "critique")
    graph.add_edge("critique", "format_review")
    graph.add_edge("format_review", "post_comment")
    graph.add_edge("post_comment", END)

    return graph.compile()