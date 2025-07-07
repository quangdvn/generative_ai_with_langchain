from operator import itemgetter

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
math_cot_prompt = hub.pull("arietem/math_cot")
cot_chain = math_cot_prompt | llm
# print(cot_chain.invoke("Solve equation 2*x+5=15"))


parse_prompt_template = (
	"Given the initial question and a full answer, "
	"extract the concise answer. Do not assume anything and "
	"only use a provided full answer.\n\nQUESTION:\n{question}\n"
	"FULL ANSWER:\n{full_answer}\n\nCONCISE ANSWER:\n"
)

parse_prompt = PromptTemplate.from_template(parse_prompt_template)

full_chain = itemgetter("question") | cot_chain

final_chain = (
	{
		"full_answer": full_chain,
		"question": itemgetter("question"),
	}
	| parse_prompt
	| llm
	| StrOutputParser()
)

res = final_chain.invoke({"question": "Solve equation 2*x+5=15"})
print(res)
full_chain.invoke({"question": "Solve equation 2*x+5=15"}).content
