from abc import ABC, abstractclassmethod
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
import requests
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()


class BaseSearchModel(ABC):
    """This is a abstract class for search engines"""

    @abstractclassmethod
    def search(self, query: str):
        pass


class DuckDuckGo(BaseSearchModel):
    def __init__(self):
        self.engine = DuckDuckGoSearchRun()

    def search(self, query: str):
        if isinstance(query, dict):
            query = query.get("query", "")
        return self.engine.run(query)


class Tavily(BaseSearchModel):
    def __init__(self):
        self.engine = TavilyClient()

    def search(self, query: str):
        res = self.engine.search(query)
        return {
            "content": res["results"][0]["content"],
            "url": res["results"][0]["url"],
        }


# if __name__ == "__main__":
#     query = "Who is the current CM of Bihar,India"
# ddg = DuckDuckGo()
# print(ddg.search(query))
# req = requests.get("https://www.geeksforgeeks.org/")

# tavily = Tavily()
# res = tavily.search(query)
# print(res["content"])
# print(res["url"])
# print("END")
