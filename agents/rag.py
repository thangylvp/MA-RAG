import os

from agents.step_definer import task_define
from src.utils import RagState, QAAnswerFormat, QAAnswerState
from src.prompt_template import extract_system_messgage, extract_human_message, extract_input_variables
from src.prompt_template import qa_human_message, qa_input_variables, qa_system_message
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState, StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

def build_rag_agent(retriever_tool = None):
    API_KEY = os.getenv("API_KEY")
    def retrieve(state: RagState):
        user_question = state["question"]
        list_docs, list_doc_ids = retriever_tool(query=user_question)
        state["documents"] = list_docs
        state["doc_ids"] = list_doc_ids
        return state

    def extract(state: RagState):
        # print("--------- EXTRACT -----------")
        list_docs = state["documents"]
        messages = [
            SystemMessagePromptTemplate.from_template(extract_system_messgage),
            HumanMessagePromptTemplate.from_template(extract_human_message),
        ]
        prompt = ChatPromptTemplate(input_variables=extract_input_variables, messages=messages)
        # LLM
        llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"), temperature=0.0, api_key=API_KEY, max_retries=5)
        chain = prompt | llm | StrOutputParser()
        user_question = state['question']
        list_notes = []
        for doc in list_docs:        
            note = chain.invoke({"passage": doc, "question": user_question})
            list_notes.append(f"[{note}]")
        state["notes"] = list_notes
        return state

    def generate(state: RagState):
        # print("--------- GENERATE -----------")
        # print("log before gen", state)
        notes = state["notes"]
        doc_ids = state["doc_ids"]
        question = state["question"]
        tmps = []
        for doc_id, note in zip(doc_ids, notes):
            tmps.append(f"doc_{doc_id}: {note}")
        docs = "\n\n".join(tmps)
        messages = [
            SystemMessagePromptTemplate.from_template(qa_system_message),
            HumanMessagePromptTemplate.from_template(qa_human_message),
        ]
        prompt = ChatPromptTemplate(input_variables=qa_input_variables, messages=messages)
        # LLM
        llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"), temperature=0.3, api_key=API_KEY, max_retries=5)
        structured_llm = llm.with_structured_output(QAAnswerFormat)
        full_prompt = prompt.format(
            context=docs,
            question=question,
        )
        # print(full_prompt)
        # Chain
        rag_chain = prompt | structured_llm

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        response = QAAnswerState(**response.model_dump())
        # print(response)
        return {"final_raw_answer": response}
        # return {"messages": [response]}
    # {"answer": response}


    rag_graph_builder = StateGraph(RagState)
    rag_graph_builder.add_node("retrieve", retrieve)
    rag_graph_builder.add_node("extract", extract)
    rag_graph_builder.add_node("generate", generate)

    rag_graph_builder.add_edge(START, "retrieve")
    rag_graph_builder.add_edge("retrieve", "extract")
    rag_graph_builder.add_edge("extract", "generate")
    rag_graph_builder.add_edge("generate", END)
    rag_graph = rag_graph_builder.compile()
    return rag_graph