import operator
from typing import Annotated, TypedDict

from IPython.display import Image
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send

# llm = ChatOpenAI(model="gpt-4-vision-preview")
# llm = ChatAnthropic(
# 	model_name="claude-sonnet-4-20250514",
# 	temperature=0,
# 	timeout=None,
# 	max_retries=2,
# 	stop=None,
# )
llm = ChatVertexAI(model_name="gemini-2.0-flash-001")
video_uri = "https://www.youtube.com/watch?v=zcYtSckecD8"


class AgentState(TypedDict):
	video_uri: str
	chunks: int
	interval_secs: int
	summaries: Annotated[list, operator.add]
	final_summary: str


class _ChunkState(TypedDict):
	video_uri: str
	start_offset: int
	interval_secs: int


reduce_prompt = PromptTemplate.from_template(
	"You are given a list of summaries that "
	"of a video splitted into sequential pieces.\n"
	"SUMMARIES:\n{summaries}"
	"Based on that, prepare a summary of a whole video."
)

human_part = {"type": "text", "text": "Provide a summary of the video."}


async def _summarize_video_chunk(state: _ChunkState):
	start_offset = state["start_offset"]
	interval_secs = state["interval_secs"]
	video_part = {
		"type": "media",
		"file_uri": state["video_uri"],
		"mime_type": "video/mp4",
		"video_metadata": {
			"start_offset": {"seconds": start_offset * interval_secs},
			"end_offset": {"seconds": (start_offset + 1) * interval_secs},
		},
	}
	response = await llm.ainvoke(HumanMessage(content=[human_part, video_part]))
	return {"summaries": [response.content]}


def _map_summaries(state: AgentState):
	chunks = state["chunks"]
	payloads = [
		{
			"video_uri": state["video_uri"],
			"interval_secs": state["interval_secs"],
			"start_offset": i,
		}
		for i in range(chunks)
	]
	return [Send("summarize_video_chunk", payload) for payload in payloads]


def _merge_summaries(summaries: list[str], interval_secs: int = 600, **kwargs) -> str:
	sub_summaries = []
	for i, summary in enumerate(summaries):
		sub_summary = (
			f"Summary from sec {i * interval_secs} to sec {(i + 1) * interval_secs}:"
			f"\n{summary}\n"
		)
		sub_summaries.append(sub_summary)
	return "".join(sub_summaries)


async def _generate_final_summary(state: AgentState):
	summary = _merge_summaries(
		summaries=state["summaries"], interval_secs=state["interval_secs"]
	)
	final_summary = await (reduce_prompt | llm | StrOutputParser()).ainvoke(
		{"summaries": summary}
	)
	return {"final_summary": final_summary}


graph = StateGraph(AgentState)

graph.add_node("summarize_video_chunk", _summarize_video_chunk)
graph.add_node("generate_final_summary", _generate_final_summary)

graph.add_conditional_edges(START, _map_summaries, ["summarize_video_chunk"])
graph.add_edge("summarize_video_chunk", "generate_final_summary")
graph.add_edge("generate_final_summary", END)

app = graph.compile()


Image(app.get_graph().draw_mermaid_png())


result = await app.ainvoke(
	{"video_uri": video_uri, "chunks": 5, "interval_secs": 600}, {"max_concurrency": 3}
)
print(result["final_summary"])
