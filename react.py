from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    :param num: a number to triple
    :return: he triple number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

llm = ChatOllama(
    model="llama3",
    temperature=0,
).bind_tools(tools)

