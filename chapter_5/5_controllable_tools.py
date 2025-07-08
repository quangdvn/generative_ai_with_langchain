from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o")


class Step(BaseModel):
	"""A step that is a part of the plan to solve the task."""

	step: str = Field(description="Description of the step")


class Plan(BaseModel):
	"""A plan to solve the task."""

	steps: list[Step]


prompt = PromptTemplate.from_template(
	"Prepare a step-by-step plan to solve the given task.\nTASK:\n{task}\n"
)
result = (prompt | llm.with_structured_output(Plan)).invoke(
	"How to write a bestseller on Amazon about generative AI?"
)

assert isinstance(result, Plan)
print(f"Amount of steps: {len(result.steps)}")
for i, step in enumerate(result.steps):
	print(f"{i}", step)
	print("=" * 10)
	print(step.step)
# break

result

