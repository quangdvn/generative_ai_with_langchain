from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Chat models are LLMs that are fine-tuned for multi-turn interaction between a model and a human.
###########################################################################
# Initialize OpenAI model - ChatModel LLM interface
chat = ChatOpenAI(model="gpt-4o")
messages = [
	SystemMessage(content="You're a helpful programming assistant"),
	HumanMessage(content="Write a Python function to calculate factorial"),
]
response = chat.invoke(messages)
response.model_dump()
print(response)
print("-" * 50)
print(response.content)
print("-" * 50)
print("-" * 50)
###########################################################################
# Claude - Anthropic thinking - Use pre-define prompt template
print("-" * 50)
# Create a template
template = ChatPromptTemplate.from_messages(
	[
		("system", "You are an experienced programmer and mathematical analyst."),
		("user", "{problem}"),
	]
)
# Initialize Claude with extended thinking enabled
chat = ChatAnthropic(
	model_name="claude-3-7-sonnet-20250219",  # Use latest model version
	max_tokens_to_sample=64_000,  # Total response length limit
	thinking={
		"type": "enabled",
		"budget_tokens": 15000,
	},  # Allocate tokens for thinking,
	timeout=60,
	stop=None,
)
# Create and run a chain
chain = template | chat
# Complex algorithmic problem
problem = """
Design an algorithm to find the kth largest element in an unsorted array
with the optimal time complexity. Analyze the time and space complexity
of your solution and explain why it's optimal.
"""
# Get response with thinking included
response = chat.invoke([HumanMessage(content=problem)])
response.model_dump()
print(response.content)
print("-" * 50)
###########################################################################
# OpenAI thinking
print("-" * 50)
template = ChatPromptTemplate.from_messages(
	[("system", "You are a problem-solving assistant."), ("user", "{problem}")]
)
# Initialize with reasoning_effort parameter
chat = ChatOpenAI(
	model="o3-mini",  # Reasoning Open AI model
	reasoning_effort="high",  # Options: "low", "medium", "high"
)
chain = template | chat
response = chain.invoke(
	{
		"problem": "Calculate the optimal strategy for an algorithm to find the kth largest element in an unsorted array with the optimal time complexity"
	}
)
# Test without reasoning_effort parameter
chat = ChatOpenAI(model="gpt-4o")
chain = template | chat
response = chain.invoke(
	{
		"problem": "Calculate the optimal strategy for an algorithm to find the kth largest element in an unsorted array with the optimal time complexity"
	}
)
