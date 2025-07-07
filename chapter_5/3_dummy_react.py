import math

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

# llm = ChatOpenAI(model="gpt-4o")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


# ========================================================================================
def mocked_google_search(query: str) -> str:
	print(f"CALLED GOOGLE_SEARCH with query={query}")
	return "Donald Trump is a president of USA and he's 78 years old"


def mocked_calculator(expression: str) -> float:
	print(f"CALLED CALCULATOR with expression={expression}")
	if "sqrt" in expression:
		return math.sqrt(78 * 132)
	return 78 * 132


# ========================================================================================
calculator_tool = {
	"title": "calculator",
	"description": "Computes mathematical expressions",
	"type": "object",
	"properties": {
		"expression": {
			"description": "A mathematical expression to be evaluated by a calculator",
			"title": "expression",
			"type": "string",
		},
	},
	"required": ["expression"],
}
search_tool = {
	"description": "Returns about common facts, fresh events and news from Google Search engine based on a query.",
	"title": "google_search",
	"type": "object",
	"properties": {
		"query": {
			"description": "Search query to be sent to the search engine",
			"title": "search_query",
			"type": "string",
		},
	},
	"required": ["query"],
}

system_prompt = (
	"Always use a calculator for mathematical computations, and use Google Search "
	"for information about common facts, fresh events and news. Do not assume anything, keep in "
	"mind that things are changing and always "
	"check yourself with external sources if possible."
)


# ========================================================================================
prompt = ChatPromptTemplate.from_messages(
	[
		("system", system_prompt),
		MessagesPlaceholder(variable_name="messages"),
	]
)
llm_with_tools = llm.bind(tools=[search_tool, calculator_tool]).bind(prompt=prompt)


# ========================================================================================
def invoke_llm(state: MessagesState):
	return {"messages": [llm_with_tools.invoke(state["messages"])]}


def call_tools(state: MessagesState):
	last_message = state["messages"][-1]
	print("Tool message: ", last_message)
	tool_calls = last_message.tool_calls

	new_messages = []

	for tool_call in tool_calls:
		if tool_call["name"] == "google_search":
			print("Tool call args: ", tool_call)
			tool_result = mocked_google_search(**tool_call["args"])
			new_messages.append(
				ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
			)
		elif tool_call["name"] == "calculator":
			tool_result = mocked_calculator(**tool_call["args"])
			new_messages.append(
				ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
			)
		else:
			raise ValueError(f"Tool {tool_call['name']} is not defined!")
	return {"messages": new_messages}


def should_run_tools(state: MessagesState):
	last_message = state["messages"][-1]
	print("Message before routing: ", last_message)
	if last_message.tool_calls:
		return "call_tools"
	return END


# ========================================================================================
builder = StateGraph(MessagesState)
builder.add_node("invoke_llm", invoke_llm)
builder.add_node("call_tools", call_tools)

builder.add_edge(START, "invoke_llm")
builder.add_conditional_edges("invoke_llm", should_run_tools)
builder.add_edge("call_tools", "invoke_llm")

graph = builder.compile()
# ========================================================================================
question = "What is a square root of the current US president's age multiplied by 132?"
# result = graph.invoke({"messages": [HumanMessage(content=question)]})
# print(result["messages"][-1].content)
# ========================================================================================


agent = create_react_agent(
	model="gpt-4o", tools=[search_tool, calculator_tool], prompt=system_prompt
)
agent.invoke({"messages": [HumanMessage(content=question)]})
