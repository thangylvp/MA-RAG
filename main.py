
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.load import dumpd, dumps, load, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from transformers import AutoTokenizer, get_scheduler, AutoModel
from langgraph.prebuilt import tools_condition, ToolNode
from data.retrieve import Retriever
from src.utils import load_dataset, load_corpus, parse_args
from langchain_core.tools import BaseTool
from prompt_template import qa_human_message, qa_input_variables, qa_system_message, extract_human_message, step_input_variables, step_system_message, step_human_message, planing_system_message_fever
from prompt_template import extract_input_variables, extract_system_messgage, planing_system_message, planing_human_message, planing_input_variables, aggregate_human_message, aggregate_input_variables, aggregate_system_message, summary_system_message, summary_human_message, summary_input_variables
from langgraph.types import Command
from src.utils import GraphState, PlanExecState, RagState, StepTaskFormat, StepTaskState, PlanFormat, PlanSummaryFormat, PlanSummaryState, QAAnswerFormat, QAAnswerState
import os
import random
import json 
import torch
import pprint
import numpy as np 
from src.utils import RetrieveTopChunk
import argparse

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")



def plan_agent(state: GraphState):
    original_question = state["original_question"]
    all_mem = []
    for past_exp in state["past_exp"]:
        memory = ""
        plan = ', '.join(past_exp["plan"])
        memory += f"Plan: [{plan}]\n"
        memory += f"Status: {past_exp["plan_summary"]["output"]} Score: {past_exp["plan_summary"]["score"]}\n"
        all_mem.append(memory)
    memory = ""
    if len(all_mem) == 0:
        memory = "empty"
    else:
        for id in range(len(all_mem)):
            memory += f"Trial {id}:\n{all_mem[id]}\n"
    
    messages = [
        SystemMessagePromptTemplate.from_template(planing_system_message),
        HumanMessagePromptTemplate.from_template(planing_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=planing_input_variables, messages=messages)
    llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"), temperature=0.3, api_key=API_KEY)
    structured_llm = llm.with_structured_output(PlanFormat)
    chain = prompt | structured_llm
    fprompt = prompt.format(
        question = original_question,
        memory = memory
    )
    output = chain.invoke({
        "question": original_question,
        "memory": memory
    })
    return {"plan": output.step}

def build_plan_executor(retriever_tool = None):
    def task_define(state: PlanExecState):
        messages = [
            SystemMessagePromptTemplate.from_template(step_system_message),
            HumanMessagePromptTemplate.from_template(step_human_message),
        ]
        prompt = ChatPromptTemplate(input_variables=step_input_variables, messages=messages)
        llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"), temperature=0.3, api_key=API_KEY, max_retries=5)
        structured_llm = llm.with_structured_output(StepTaskFormat)
        chain = prompt | structured_llm

        # check stop or continue
        if len(state["step_output"]) == len(state["plan"]) or (len(state["step_output"]) > 0 and state["step_output"][-1]["success"].lower() == "no"):
            # summary about this plan and then stop
            messages = [
                SystemMessagePromptTemplate.from_template(summary_system_message),
                HumanMessagePromptTemplate.from_template(summary_human_message),
            ]
            prompt = ChatPromptTemplate(input_variables=summary_input_variables, messages=messages)
            llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME"), temperature=0.0, api_key=API_KEY,  max_retries=5)
            structured_llm = llm.with_structured_output(PlanSummaryFormat)

            chain = prompt | structured_llm
            question = state["original_question"]
            plan = f"[{', '.join(state["plan"])}]"
            memory = ""
            for id, item in enumerate(state["step_output"]):
                memory += f"Task: {state["plan"][id]}\nQuestion: {state["step_question"][id]["task"]}\nAnswer: {item["answer"]}\nConfident score: {item["rating"]}\n\n"
            full_prompt = prompt.format(
                question = question,
                plan = plan,
                memory = memory
            )
            output = chain.invoke({
                "question": question, 
                "plan": plan,
                "memory": memory
            })
            output = PlanSummaryState(**output.model_dump())
            return {"plan_summary": output, "stop": True}
        else:
            plan = f"[{', '.join(state["plan"])}]"
            cur_step = state["plan"][len(state["step_output"])]
            memory = ""
            for id in range(len(state["step_output"])):
                memory += f"Task: {state["plan"][id]}\nAnswer: {state["step_output"][id]["answer"]}\n\n"
            response = chain.invoke({"plan": plan, "cur_step": cur_step, "memory": memory})
            response = StepTaskState(**response.model_dump())
            return {"step_question": [response]}
    
    def build_rag_agent():
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
    
    rag_agent = build_rag_agent()
    
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

def plan_executor_node(state: GraphState):
    input = {
        "original_question": state["original_question"],
        "plan": state["plan"],
        "stop": False
    }
    output = plan_executor_agent.invoke(input)
    return {"past_exp": [output]}

if __name__ == "__main__":
    args = parse_args()
    
    model_name_or_path = 'hf/gte-multilingual-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = model.to(f"cuda:{args.gpus[0]}")

    retrieve = Retriever(gpu_ids=args.gpus)
    retrieve.init_index_and_add(root_dir="save_embs/gte-ml-base", dataset_name="dpr100")

    dpr100_corpus = load_corpus()
    retriever_tool = RetrieveTopChunk(tokenizer=tokenizer, embedding_model=model, retrieval_model=retrieve, corpus=dpr100_corpus)
    
    global plan_executor_agent

    plan_executor_agent = build_plan_executor(retriever_tool=retriever_tool)


    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("planer_node", plan_agent)
    graph_builder.add_node("plan_executor_node", plan_executor_node)
    graph_builder.add_edge(START, "planer_node")
    graph_builder.add_edge("planer_node", "plan_executor_node")

    graph = graph_builder.compile()

    exp_name = args.exp
    model = args.model
    dataset_name = args.dataset
    dataset = load_dataset(name=dataset_name)
    
    c = 0
    save_dir = f"{exp_name}_{model}_{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    for id, item in enumerate(dataset):
        if id < args.start_index or id > args.end_index:
            continue
        question_id = item["id"]
        print(question_id)
        save_file = os.path.join(save_dir, f"{question_id}.json")
        if os.path.exists(save_file):
            continue
        if dataset_name == "fever":
            question = f"Verify this claim, answer SUPPORTS or REFUTES\n{item['input']}"
        else:
            question = item["input"]
        inputs = {
            "original_question": f"{question}?"
        }
        try:
            output = graph.invoke(inputs)
            print(output)
            print()

            with open(save_file, "w") as f:
                json.dump(output, f)
        except Exception as e: 
            print(e)
            pass
