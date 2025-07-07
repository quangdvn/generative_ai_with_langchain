from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class JobApplicationState(TypedDict):
	job_description: str
	is_suitable: bool
	application: str


# class JobApplicationState(BaseModel):
# 	job_description: str
# 	is_suitable: Optional[bool] = None
# 	application: Optional[str] = None


def analyze_job_description(state):
	print("Current state: ", state)
	print("...Analyzing a provided job description ...")
	return {"is_suitable": len(state["job_description"]) > 100}


def generate_application(state):
	print("...generating application...")
	return {"application": "some_fake_application"}


builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)

builder.add_edge(START, "analyze_job_description")
builder.add_edge("analyze_job_description", "generate_application")
builder.add_edge("generate_application", END)

graph = builder.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))

res = graph.invoke({"job_description": "I am a software engineer"})
