from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
sub_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

################## WITHOUT KNOWLEDGE ##################

question = "how old is the US president?"
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
# step1 = llm.invoke(question, tools=[{"type": "web_search_preview"}])
step1 = sub_llm.invoke(question, tools=[search_tool])
print(step1.content)
print(step1.tool_calls)
step1


tool_result = ToolMessage(
	content="Donald Trump › Age 78 years June 14, 1946\n",
	tool_call_id=step1.tool_calls[0]["id"],
)
step2 = sub_llm.invoke(
	[HumanMessage(content=question), step1, tool_result], tools=[search_tool]
)
assert len(step2.tool_calls) == 0

print(step2)
print("*" * 50)
llm_with_tools = sub_llm.bind(tools=[search_tool])
with_tools = llm_with_tools.invoke(question)
print(with_tools.model_dump_json(indent=2))
