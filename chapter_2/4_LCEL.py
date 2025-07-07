from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

#########? Simple chain workflow #########

# Create components
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI()
output_parser = StrOutputParser()

# Chain them together using LCEL
chain = prompt | llm | output_parser
#  Execute the workflow with a single call
result = chain.invoke({"topic": "programming"})
print(result)

# Without LCEL, the same workflow is equivalent to separate function calls
# with manual data passing:

# formatted_prompt = prompt.invoke({"topic": "programming"})
# llm_output = llm.invoke(formatted_prompt)
# result = output_parser.invoke(llm_output)


#########? Complex chain workflow #########

# First chain generates a story
story_prompt = PromptTemplate.from_template("Write a short story about {topic}")
story_chain = story_prompt | llm | StrOutputParser()

# Second chain analyzes the story
analysis_prompt = PromptTemplate.from_template(
	"Analyze the following story's mood:\n{story}"
)
analysis_chain = analysis_prompt | llm | StrOutputParser()

# We can compose these two chains together.
# Our first simple approach pipes the story directly into the analysis chain:
# Combine chains
story_with_analysis = story_chain | analysis_chain

# Run the combined chain
story_analysis = story_with_analysis.invoke({"topic": "a rainy day"})
print("\nAnalysis:", story_analysis)

#########? Complex workflow with preserved context #########
enhanced_chain = RunnablePassthrough.assign(
	story=story_chain  # Add 'story' key with generated content
).assign(
	analysis=analysis_chain  # Add 'analysis' key with analysis of the story
)


# Execute the chain
result = enhanced_chain.invoke({"topic": "a rainy day"})
print(result.keys())  # Output: dict_keys(['topic', 'story', 'analysis'])

#########? Complex workflow with structured output #########
# Alternative approach using dictionary construction
manual_chain = (
	RunnablePassthrough()  # Pass through input
	| {
		"story": story_chain,  # Add story result
		"topic": itemgetter("topic"),  # Preserve original topic
	}
	| RunnablePassthrough().assign(  # Add analysis based on story
		analysis=analysis_chain
	)
)
result = manual_chain.invoke({"topic": "a rainy day"})
print(result.keys())  # Output: dict_keys(['story', 'topic', 'analysis'])
