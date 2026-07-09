from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(temperature=0, model="qwen3:1.7b")
# prompt = hub.pull("rlm/rag-prompt")


prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
 If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
""")


generation_chain = prompt | llm | StrOutputParser()