from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from abc import ABC, abstractclassmethod
from dotenv import load_dotenv

load_dotenv()

google_embadding_fun = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


class BaseDBModel(ABC):
    def __init__(self, embedding_fun):
        pass

    def save_to_memory(self, user, ai):
        pass

    def get_memory(self, current_question):
        pass


class ChromaDB(BaseDBModel):
    def __init__(self, embedding_fun):
        self.vectorstore = Chroma(
            persist_directory="./database/Chroma_db",
            collection_name="chat_memory",
            embedding_function=embedding_fun,
        )

    def save_to_memory(self, user, ai):
        try:
            user_ids = self.vectorstore.add_texts(texts=[f"user: {user}"])
            ai_ids = self.vectorstore.add_texts(texts=[f"Ai: {ai}"])
            if ai_ids and user_ids:
                return ids
            else:
                return "Can not add to memory"
        except Exception as ex:
            print(ex)

    def get_memory(self, current_question):
        return self.vectorstore.similarity_search(current_question)


if __name__ == "__main__":
    google_embadding_fun = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    cdb = ChromaDB(google_embadding_fun)
    embed = cdb.save_to_memory("Hi How are you", "Hi I am good")
    print(embed)
    print(cdb.get_memory("Hi how"))
