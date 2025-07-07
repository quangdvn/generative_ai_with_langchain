from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain.output_parsers.retry import NAIVE_RETRY_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI()


class SearchAction(BaseModel):
	query: str = Field(description="A query to search for if a search action is taken")


parser = PydanticOutputParser(pydantic_object=SearchAction)

completion_with_error = "{'action': 'what is the weather likein Munich tomorrow}"

try:
	parser.parse(completion_with_error)
except Exception as e:
	print(e)


fix_parser = RetryWithErrorOutputParser.from_llm(llm=llm, parser=parser)


print(NAIVE_RETRY_PROMPT)


retry_template = (
	"Your previous response doesn't follow the required schema and fails parsing. Fix the response so that it follow the expected schema."
	"Do not change the nature of response, only adjust the schema."
	"\n\nEXPECTED SCHEMA:{schema}\n\n"
)
retry_prompt = PromptTemplate.from_template(retry_template)

fixed_output = fix_parser.parse_with_prompt(
	completion=completion_with_error,
	prompt_value=retry_prompt.format_prompt(schema=parser.get_format_instructions()),
)
if isinstance(fixed_output, str):
	print("Fixed output (string):", fixed_output)
	print("LLM response:", llm.invoke(fixed_output))
	print(
		"Structured LLM response:",
		llm.with_structured_output(SearchAction).invoke(fixed_output),
	)
else:
	print("Fixed output is not a string. Type:", type(fixed_output))
