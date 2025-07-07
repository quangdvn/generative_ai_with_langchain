# from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()


# Initialize OpenAI model - Completion LLM interface
# For factual, consistent responses
factual_llm = OpenAI(temperature=0.1, max_tokens=256)
fact_resp = factual_llm.invoke("Tell me a story")

# For creative brainstorming
creative_llm = OpenAI(temperature=0.8, top_p=0.95, max_tokens=512)
creative_resp = creative_llm.stream("Tell me a true story")
for chunk in creative_resp:
	print(chunk, end="")

# creative_resp = creative_llm.ainvoke("Tell me a true story")
