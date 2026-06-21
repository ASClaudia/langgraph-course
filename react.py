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
    print("it was called")
    return float(num) * 3

# if you want less TavilySearch calls we can change max_results to 5, for example.
tools = [TavilySearch(max_results=3), triple]

llm = ChatOllama(
    model="qwen2.5",
    temperature=0,
).bind_tools(tools)

