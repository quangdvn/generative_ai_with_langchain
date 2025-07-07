from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessageGraph


def test_node(state):
	# ignore the last message since it's an input one
	print(f"History length = {len(state[:-1])}")
	return [AIMessage(content="Hello!")]


# Represents a state with only a list of messages
builder = MessageGraph()

builder.add_node("test_node", test_node)

builder.add_edge(START, "test_node")
builder.add_edge("test_node", END)

# Keep checkpoints in local memory
# and pass it to the graph during compilation
memory = MemorySaver()

graph = builder.compile(checkpointer=memory)

_ = graph.invoke(
	[HumanMessage(content="test")], config={"configurable": {"thread_id": "thread-a"}}
)
_ = graph.invoke(
	[HumanMessage(content="test")], config={"configurable": {"thread_id": "thread-b"}}
)
_ = graph.invoke(
	[HumanMessage(content="test")], config={"configurable": {"thread_id": "thread-a"}}
)

checkpoints = list(memory.list(config={"configurable": {"thread_id": "thread-a"}}))
for check_point in checkpoints:
	print(check_point.config["configurable"]["checkpoint_id"])

checkpoint_id = checkpoints[-1].config["configurable"]["checkpoint_id"]

_ = graph.invoke(
	[HumanMessage(content="test")],
	config={"configurable": {"thread_id": "thread-a", "checkpoint_id": checkpoint_id}},
)

checkpoint_id = checkpoints[-3].config["configurable"]["checkpoint_id"]
_ = graph.invoke(
	[HumanMessage(content="test")],
	config={"configurable": {"thread_id": "thread-a", "checkpoint_id": checkpoint_id}},
)
