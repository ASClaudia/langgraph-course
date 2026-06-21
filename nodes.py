from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

# This is the reasoning node
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning loop using the provided MessagesState.

    Args:
        state: The current state of messages in the conversation.

    Returns:
        Updated MessagesState after processing the agent's reasoning.
    """

    response = llm.invoke({"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)