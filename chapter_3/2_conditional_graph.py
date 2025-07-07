from operator import add
from typing import Annotated, Literal, Optional, Union

from IPython.display import Image, display
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages
from typing_extensions import TypedDict


def my_reducer(left: list[str], right: Optional[Union[str, list[str]]]) -> list[str]:
	if right:
		return left + [right] if isinstance(right, str) else left + right
	return left


class JobApplicationState(TypedDict):
	job_description: str
	is_suitable: bool
	application: str
	actions: Annotated[list[str], add]  # Fastest way to create a reducer
	# actions: Annotated[list[str], my_reducer]  # Custom reducer
	messages: Annotated[list[AnyMessage], add_messages]


def analyze_job_description(state):
	print("Current state: ", state)
	print("...Analyzing a provided job description ...")
	result = {
		"is_suitable": len(state["job_description"]) < 100,
		"actions": ["action1"],
	}
	return result


def generate_application(state, config: RunnableConfig):
	print("...generating application...")
	model_provider = config["configurable"].get("model_provider", "Google")
	model_name = config["configurable"].get("model_name", "gemini-1.5-flash-002")
	print(f"...generating application with {model_provider} and {model_name} ...")
	return {"application": "some_fake_application", "actions": ["action2", "action3"]}


def is_suitable_condition(
	state: JobApplicationState,
) -> Literal["generate_application", END]:
	if state.get("is_suitable"):
		return "generate_application"
	return END


builder = StateGraph(JobApplicationState)

builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)


builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

res = graph.invoke(
	input={"job_description": "I am a software engineer"},
	config={"configurable": {"model_provider": "OpenAI", "model_name": "gpt-4o"}},
)

# async for chunk in graph.astream(
# 	{"job_description": "I am a software engineer"}, stream_mode="values"
# ):
# 	print(chunk)
# 	print("\n")

display(Image(graph.get_graph().draw_mermaid_png()))
