from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

# LlmModel.py
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


class GroqModel:
    def __init__(self):
        self.model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
        )
        self.prompt = PromptTemplate(
            template="""
            You are given user query: "{query}"
            Decide whether the user is asking to:
            1. Use search engine (for current events, real-time info, weather, web searches)
            2. Use user input documents (for querying personal documents/notes)
            3. Do something else (for general chat, greetings, math, basic questions)
            
            Respond only with the option number and name, e.g. "1. Use search engine" or "3. Do something else".
            """,
            input_variables=["query"],
        )
        # Added a general prompt to answer queries in the fallback branch
        self.qa_prompt = PromptTemplate(
            template="Answer the user's question: {query}",
            input_variables=["query"],
        )


class GoogleModel:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def invoke(self, query):
        return self.model.invoke(query)
