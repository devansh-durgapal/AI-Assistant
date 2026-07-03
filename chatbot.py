from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
    RunnableParallel,
    RunnableSequence,
)
from langchain_core.output_parsers import StrOutputParser
from LlmModel import GroqModel, GoogleModel
from searchEngine import DuckDuckGo
from dotenv import load_dotenvrequests

load_dotenv()

google_model = GoogleModel()
groq_model = GroqModel()
search_engine = DuckDuckGo()
parser = StrOutputParser()


class LLM:
    def search_ddg_branch(self):
        return RunnableLambda(lambda x: search_engine.search(x["query"]))

    def doc_search_branch(self):
        chain = groq_model.qa_prompt | groq_model.model | parser
        return chain


llm = LLM()

# Classifier chain -> decide what to do
classifier_chain = groq_model.prompt | groq_model.model | parser

# Input preparation: Keeps both the query and the classification decision
# input_prep = {"query": RunnablePassthrough(), "decision": classifier_chain}
input_prep = RunnableParallel(
    {"query": RunnablePassthrough(), "decision": RunnableSequence(classifier_chain)}
)

# Define branches using RunnableBranch
branch = RunnableBranch(
    (
        lambda x: "search engine" in x["decision"].lower()
        or "1" in x["decision"].lower(),
        llm.search_ddg_branch(),
    ),
    llm.doc_search_branch(),
)

final_chain = input_prep | branch

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Thank you for chatting with Prerana. Have a great day!")
            break

        response = final_chain.invoke(user_input)
        print("Prerana: ", response)
        print("-" * 100)
