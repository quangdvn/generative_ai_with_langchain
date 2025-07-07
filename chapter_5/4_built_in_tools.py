from IPython.display import Image, display
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.ddg_search.tool import DDGInput
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# llm = ChatOpenAI(model="gpt-4o")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

search = DuckDuckGoSearchRun()
print(f"Tool's name = {search.name}")

print(f"Tool's name = {search.description}")
print(f"Tool's arg schema = f{search.args_schema}")

print(DDGInput.model_fields)

# query = "What is the weather in Munich like today and tomorrow?"
query = "Who is the current US president?"
search_input = DDGInput(query=query)
result = search.invoke(search_input.model_dump())
print(result)

result = llm.invoke(
	[
		(
			"system",
			"Always use a duckduckgo_search tool for queries that require a fresh information",
		),
		("user", query),
	],
	tools=[search],
)
print(result.tool_calls[0])

result = llm.invoke(
	[
		(
			"system",
			"Always use a duckduckgo_search tool for queries that require a fresh information",
		),
		("user", "How much is 2+2?"),
	],
	tools=[search],
)
assert not result.tool_calls
print("=" * 50)

########### Search tools With ReAct Agent ###########
agent = create_react_agent(
	model=llm,
	tools=[search],
	prompt="Always use a duckduckgo_search tool for queries that require a fresh information",
)
display(Image(agent.get_graph().draw_mermaid_png()))
print(HumanMessage(content=query).pretty_print())
for event in agent.stream({"messages": [("user", query)]}):
	# print(event)
	update = event.get("agent", event.get("tools", {}))
	for message in update.get("messages", []):
		message.pretty_print()

agent.invoke({"messages": [("user", query)]})

########### Code execution tools With ReAct Agent ###########
python_repl = PythonREPL()
python_repl.run("print(2**4)")

code_interpreter_tool = Tool(
	name="python_repl",
	description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
	func=python_repl.run,
)

query_strawberry = "How many r are in the word strawberry?"
print(llm.invoke(query_strawberry).content)

sb_agent = create_react_agent(model=llm, tools=[code_interpreter_tool])

for event in sb_agent.stream({"messages": [("user", query_strawberry)]}):
	print(event)
	messages = event.get("agent", event.get("tools", {})).get("messages", [])
	for m in messages:
		m.pretty_print()

########### Tool as HTTP Wrapper ##################
toolkit = RequestsToolkit(
	requests_wrapper=TextRequestsWrapper(headers={}),
	allow_dangerous_requests=True,
)

for tool in toolkit.get_tools():
	print(tool.name)

api_spec = """
	openapi: 3.0.0
	info:
		title: Frankfurter Currency Exchange API
		version: v1
		description: API for retrieving currency exchange rates. Pay attention to the base currency and change it if needed.

	servers:
		- url: https://api.frankfurter.dev/v1

	paths:
		/v1/{date}:
			get:
				summary: Get exchange rates for a specific date.
				parameters:
					- in: path
						name: date
						schema:
							type: string
							pattern: '^\d{4}-\d{2}-\d{2}$' # YYYY-MM-DD format
						required: true
						description: The date for which to retrieve exchange rates.  Use YYYY-MM-DD format.  Example: 2009-01-04
					- in: query
						name: symbols
						schema:
							type: string
						description: Comma-separated list of currency symbols to retrieve rates for. Example: GBP,USD,EUR

		/v1/latest:
			get:
				summary: Get the latest exchange rates.
				parameters:
					- in: query
						name: symbols
						schema:
							type: string
						description: Comma-separated list of currency symbols to retrieve rates for. Example: CHF,GBP
					- in: query
						name: base
						schema:
							type: string
						description: The base currency for the exchange rates. If not provided, EUR is used as a base currency. Example: USD
"""

system_message = (
	"You're given the API spec:\n{api_spec}\n"
	"Use the API to answer users' queries if possible. "
)

agent = create_react_agent(
	llm, toolkit.get_tools(), prompt=system_message.format(api_spec=api_spec)
)

query = "What is the swiss franc to US dollar exchange rate today (latest)? Try your best without asking again"

events = agent.stream(
	{"messages": [("user", query)]},
	# stream_mode="updates",
	stream_mode="values",
)
for event in events:
	print("Raw:", event)
	# event["messages"][-1].pretty_print()
	print("\n\n")
