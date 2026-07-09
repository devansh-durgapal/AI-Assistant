from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
    RunnableParallel,
    RunnableSequence,
)
from Chroma import ChromaDB
from LlmModel import GroqModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# Stage 1 ==> Personal chat
class Branch1:
    def __init__(self, query, model, ai_message, parser):
        self.query = query
        self.ai_message = ai_message
        self.model = model
        self.chromadb = ChromaDB()
        self.parser = parser

    def is_similar(self, query):
        "If similar info found return it or None"
        data = self.chromadb.get_memory(query)
        if data:
            return data["data"]
        return None

    def add_conversation_to_data(self, response):
        conversation = self.chromadb.save_to_memory(self.query, response)
        return "Successfull added summary"

    def execute_branch(self):
        #  pre_llm_branch --> model chain --> post_llm_branch
        # post_llm_branch gives op and store the conversation
        pre_llm_branch = RunnableParallel(
            {
                "similar": RunnableLambda(self.is_similar),
                "query": RunnablePassthrough(),
            }
        )
        pre_llm_prompt = PromptTemplate(
            template="""
            Similar conversation:
            {similar}

            User query:
            {query}

            Answer the user's query using the similar conversation if it is relevant.
            """,
            input_variables=["similar", "query"],
        )

        post_llm_prompt = PromptTemplate(
            template="Create the summary of the conversion between Ai and the use you are give Ai response {response}",
            input_variables=["response"],
        )
        post_llm_branch = RunnableParallel(
            {
                "response": RunnablePassthrough(),
                "summary": RunnableLambda(self.add_conversation_to_data),
            }
        )

        full_chain = (
            pre_llm_branch
            | pre_llm_prompt
            | self.model.model
            | self.parser
            | post_llm_branch
        )
        return full_chain.invoke(self.query)


if __name__ == "__main__":

    groq_model = GroqModel()
    parser = StrOutputParser()
    b1 = Branch1("Tell me one world what the username", groq_model, "ACha", parser)
    exe_b1 = b1.execute_branch()
    print(exe_b1)
    print(b1.is_similar("Bye"))
    print("End")
