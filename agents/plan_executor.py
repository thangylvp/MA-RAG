import os

from agents.step_definer import task_define
from agents.rag import build_rag_agent
from src.utils import GraphState, PlanExecState, RagState, StepTaskFormat, StepTaskState, PlanFormat, PlanSummaryFormat, PlanSummaryState, QAAnswerFormat, QAAnswerState
from src.prompt_template import extract_system_messgage, extract_human_message, extract_input_variables
from src.prompt_template import qa_human_message, qa_input_variables, qa_system_message
from src.prompt_template import aggregate_human_message, aggregate_input_variables, aggregate_system_message
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState, StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

def build_plan_executor(retriever_tool = None):
    API_KEY = os.getenv("API_KEY")


    rag_agent = build_rag_agent(retriever_tool=retriever_tool)

    def single_task_execute(state: PlanExecState):
        cur_task = state["step_question"][-1]
        query = cur_task["task"]
        if cur_task["type"] == "aggregate":
            messages = [
                SystemMessagePromptTemplate.from_template(aggregate_system_message),
                HumanMessagePromptTemplate.from_template(aggregate_human_message),
            ]
            prompt = ChatPromptTemplate(input_variables=aggregate_input_variables, messages=messages)
            llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"), temperature=0.0, api_key=API_KEY,  max_retries=5)
            structured_llm = llm.with_structured_output(QAAnswerFormat)
            chain = prompt | structured_llm
            full_prompt = prompt.format(
                question=query,
            )
            response = chain.invoke({"question": query})
            response = QAAnswerState(**response.model_dump())
            step_doc_ids = []
            step_notes = []
        else:
            response = rag_agent.invoke({
                "question": query
            })
            step_doc_ids = [response["doc_ids"]]
            step_notes = [response["notes"]]
            response = response["final_raw_answer"]
    
        return {"step_output": [response], "step_docs_ids": step_doc_ids, "step_notes": step_notes}
    
    def task_definer_out(state: PlanExecState):
        if state["stop"] == True:
            return END
        else:
            return "single_task_execute"
        
    graph_builder = StateGraph(PlanExecState)
    
    graph_builder.add_node("task_definer", task_define)
    graph_builder.add_node("single_task_execute", single_task_execute)
    graph_builder.add_edge(START, "task_definer")
    graph_builder.add_edge("single_task_execute", "task_definer")
    graph_builder.add_conditional_edges("task_definer", task_definer_out)
    graph = graph_builder.compile()
    return graph

