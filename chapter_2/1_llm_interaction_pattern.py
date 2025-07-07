# from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()


# Initialize OpenAI model - Completion LLM interface
openai_llm = OpenAI()

# Transform a single input into an output.
response = openai_llm.invoke("Tell me a joke about light bulbs!")
print(response)
print("-" * 50)

# Initialize a Gemini model
# gemini_pro = GoogleGenerativeAI(model="gemini-1.5-pro")
# response = gemini_pro.invoke("Tell me a joke about light bulbs!")

# Either one or both can be used with the same interface
