import operator
from operator import ne
from typing import Annotated, Literal, TypedDict

from langchain.agents import load_tools
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import convert_runnable_to_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field

# ===========================
# Data Structure
# ===========================


class Plan(BaseModel):
	"""Plan to follow in future"""

	steps: list[str] = Field(
		description="different steps to follow, should be in sorted order"
	)


class PlanState(TypedDict):
	task: str
	plan: Plan
	past_steps: Annotated[list[str], operator.add]
	final_response: str


class StepState(AgentState):
	plan: str
	step: str
	task: str


# ===========================
# Base Prompt
# ===========================
planner_prompt_template = (
	"For the given task, come up with a step by step plan.\n"
	"This plan should involve individual tasks, that if executed correctly will "
	"yield the correct answer. Do not add any superfluous steps.\n"
	"The result of the final step should be the final answer. Make sure that each "
	"step has all the information needed - do not skip steps."
)

planner_prompt = ChatPromptTemplate.from_messages(
	[
		("system", planner_prompt_template),
		("user", "Prepare a plan how to solve the following task:\n{task}\n"),
	]
)

agent_system_prompt = (
	"You're a smart assistant that carefully helps to solve complex tasks.\n"
	" Given a general plan to solve a task and a specific step, work on this step. "
	" Don't assume anything, keep in minds things might change and always try to "
	"use tools to double-check yourself.\n"
	" Use a calculator for mathematical computations, use Search to gather"
	"for information about common facts, fresh events and news, use Arxiv to get "
	"ideas on recent research and use Wikipedia for common knowledge."
)

agent_step_template = (
	"Given the task and the plan, try to execute on a specific step of the plan.\n"
	"TASK:\n{task}\n\nPLAN:\n{plan}\n\nSTEP TO EXECUTE:\n{step}\n"
)

agent_prompt_template = ChatPromptTemplate.from_messages(
	[
		("system", agent_system_prompt),
		("user", agent_step_template),
	]
)

final_prompt = PromptTemplate.from_template(
	"You're a helpful assistant that has executed on a plan."
	"Given the results of the execution, prepare the final response.\n"
	"Don't assume anything\nTASK:\n{task}\n\nPLAN WITH RESUlTS:\n{plan}\n"
	"FINAL RESPONSE:\n"
)


# ===========================
# Base LLM
# ===========================
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)
llm = ChatOpenAI(model="gpt-4o", temperature=1.0)

planner = planner_prompt | llm.with_structured_output(Plan)


# ===========================
# Base Tools
# ===========================


class CalculatorArgs(BaseModel):
	expression: str = Field(description="Mathematical expression to be evaluated")


def calculator(state: CalculatorArgs, config: RunnableConfig) -> str:
	expression = state["expression"]
	math_constants = config["configurable"].get("math_constants", {})
	result = ne.evaluate(expression.strip(), local_dict=math_constants)
	return str(result)


calculator_with_retry = RunnableLambda(calculator).with_retry(
	wait_exponential_jitter=True,
	stop_after_attempt=3,
)

calculator_tool = convert_runnable_to_tool(
	calculator_with_retry,
	name="calculator",
	description=(
		"Calculates a single mathematical expression, incl. complex numbers."
		"'\nAlways add * to operations, examples:\n73i -> 73*i\n"
		"7pi**2 -> 7*pi**2"
	),
	args_schema=CalculatorArgs,
	arg_types={"expression": "str"},
)

tools = load_tools(tool_names=["ddg-search", "arxiv", "wikipedia"], llm=llm) + [
	calculator_tool
]

# ===========================
# ReAct Agent
# ===========================
execution_agent = create_react_agent(
	model=llm,
	tools=tools + [calculator_tool],
	state_schema=StepState,
	prompt=agent_prompt_template,
)

# ===========================
# Node Functions
# ===========================


def get_current_step(state: PlanState) -> int:
	"""Returns the number of current step to be executed."""
	return len(state.get("past_steps", []))


def get_full_plan(state: PlanState) -> str:
	"""Returns formatted plan with step numbers and past results."""
	full_plan = []
	for i, step in enumerate(state["plan"].steps):
		full_step = f"# {i + 1}. Planned step: {step}\n"
		if i < get_current_step(state):
			full_step += f"Result: {state['past_steps'][i]}\n"
		full_plan.append(full_step)
	return "\n".join(full_plan)


async def _build_initial_plan(state: PlanState) -> PlanState:
	plan = await planner.ainvoke(state["task"])
	print(f"Initial plan: {len(plan.steps)} steps\n")
	for i, step in enumerate(plan.steps):
		print(f"{i + 1}: {step}\n")
	return {"plan": plan}


async def _run_step(state: PlanState) -> PlanState:
	plan = state["plan"]
	current_step = get_current_step(state)
	print("Current step: ", current_step + 1)
	print("Current step task: ", plan.steps[current_step])
	print("Past steps count: ", len(state["past_steps"]))
	step = await execution_agent.ainvoke(
		{
			"plan": get_full_plan(state),
			"step": plan.steps[current_step],
			"task": state["task"],
		}
	)
	print("=" * 30)
	print("Current step to-do list:\n", step["messages"][-1].content)
	print("=" * 30)
	return {"past_steps": [step["messages"][-1].content]}


async def _get_final_response(state: PlanState) -> PlanState:
	final_response = await (final_prompt | llm).ainvoke(
		{"task": state["task"], "plan": get_full_plan(state)}
	)
	return {"final_response": final_response}


def _should_continue(state: PlanState) -> Literal["run", "response"]:
	if get_current_step(state) < len(state["plan"].steps):
		return "run"
	return "response"


# ===========================
# Build Graph
# ===========================

builder = StateGraph(PlanState)

builder.add_node("initial_plan", _build_initial_plan)
builder.add_node("run", _run_step)
builder.add_node("response", _get_final_response)

builder.add_edge(START, "initial_plan")
builder.add_edge("initial_plan", "run")
builder.add_conditional_edges("run", _should_continue)
builder.add_edge("response", END)

graph = builder.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))

# ===========================
# Run Graph
# ===========================

task = "Write a strategic one-pager of building an AI startup"
result = await graph.ainvoke({"task": task})
result
# result
