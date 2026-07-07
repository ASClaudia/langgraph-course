import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser,
                                                        PydanticToolsParser)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOllama(model="llama3.1:latest", temperature=0.5)
parser = JsonOutputToolsParser(return_id=True)
# it's going to take the answer from the llm and create an AnswerQuestion object we can easily work with.
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

#  tool_choice will force the llm to format its response as AnswerQuestion, which will be parsed by the PydanticToolsParser
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# structured_llm = llm.with_structured_output(AnswerQuestion)
# first_responder = first_responder_prompt_template | structured_llm

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_prompt_template = actor_prompt_template.partial(
    first_instruction=revise_instructions
)

revisor = (revisor_prompt_template
           | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
           )

# structured_llm = llm.with_structured_output(ReviseAnswer)
# revisor = revisor_prompt_template | structured_llm


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )
    # chain = (
    #     first_responder_prompt_template
    #     | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    #     | parser_pydantic
    # )
    #
    # res = chain.invoke(input={"messages": [human_message]})

    # This works with ChatOllama/Qwen.
    # The previous implementation worked with OpenAI models using tool/function calling:
    #
    #     llm.bind_tools(...)
    #     | PydanticToolsParser(...)
    #
    # GPT models correctly populated the AnswerQuestion schema as a tool call.
    # Qwen (via Ollama) instead generated an invalid tool call containing only the
    # input question, which caused Pydantic validation to fail.
    #
    # Using with_structured_output() asks the model to directly produce an
    # AnswerQuestion object and is more reliable with ChatOllama.

    structured_llm = llm.with_structured_output(AnswerQuestion)
    chain = first_responder_prompt_template | structured_llm

    res = chain.invoke({"messages": [human_message]})

    print(res)
