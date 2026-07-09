from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from abc import ABC, abstractclassmethod
from dotenv import load_dotenv
import os

load_dotenv()

google_embedding_fun = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

print(os.getenv("GOOGLE_API_KEY"))


class BaseDBModel(ABC):
    def __init__(self, embedding_fun):
        pass

    def save_to_memory(self, user, ai):
        pass

    def get_memory(self, current_question):
        pass


class ChromaDB(BaseDBModel):
    def __init__(self, embedding_fun=google_embedding_fun):
        self.vectorstore = Chroma(
            persist_directory="./database/Chroma_db",
            collection_name="chat_memory",
            embedding_function=embedding_fun,
        )

    def save_to_memory(self, user, ai, session_id) -> dict:
        user_id = f"{session_id}_user"
        ai_id = f"{session_id}_ai"
        try:
            user_ids = self.vectorstore.add_texts(
                ids=[user_id],
                texts=[user],
                metadatas=[
                    {
                        "session": session_id,
                        "role": "User",
                        "memory_type": "personal_info",
                    }
                ],
            )
            ai_ids = self.vectorstore.add_texts(
                texts=[ai],
                ids=[ai_id],
                metadatas=[{"role": "AI", "session": session_id}],
            )
            return {"user_ids": user_ids, "ai_ids": ai_ids}

        except Exception as ex:
            print(ex)
            raise

    def search_similality(self, current_question, k=2) -> list:
        "It return session id of similar convo"
        search_result = self.vectorstore.similarity_search(current_question, k=k)
        return search_result

    def get_best_memory(self, current_question, session_ids: list[int]) -> dict | None:
        """Return stored memory if the best matching score is below threshold."""

        threshold = 0.7
        best_score = float("inf")
        best_session_id = None

        for session_id in session_ids:
            results = self.vectorstore.similarity_search_with_score(
                current_question, filter={"session": session_id}
            )

            if results:
                score = results[0][1]

                if score < best_score:
                    best_score = score
                    best_session_id = session_id
        print(best_session_id, best_score)
        if best_session_id is not None and best_score <= threshold:
            result = self.vectorstore.similarity_search(
                current_question, filter={"session": best_session_id}
            )
            print(result)

            if result:
                return {"score": best_score, "data": result[0].page_content}

        return {"NOne": None}

    def get_memory(self, current_query, session_ids, k=5, threshold=0.5, filter=None):

        # Number session id as input
        # check each session id score and if less than thershold then add in a list
        # Arrange base on score
        # Return all the list or make summary of it
        convo = []
        for session_id in session_ids:
            session_filter = {"session": session_id}
            if filter:
                session_filter = {"$and": [session_filter, filter]}

            result = self.vectorstore.similarity_search_with_score(
                current_query, k=k, filter=session_filter
            )
            if result[0][1] <= threshold:
                convo.append(result[0][0].page_content)
        return convo


if __name__ == "__main__":

    cdb = ChromaDB()
    embed = cdb.save_to_memory("Devansh loves to love code", "That great", 2)
    # print(embed)
    a = cdb.get_memory("what devansh  do", 2)
    # print(a)

    print("End")
