import os
import random
import json 
import numpy as np 
import argparse

from langgraph.graph import MessagesState, StateGraph, START, END
from transformers import AutoTokenizer, AutoModel
from corpus.retrieve import Retriever
from src.utils import load_dataset, load_corpus, parse_args
from src.utils import GraphState
from src.utils import RetrieveTopChunk
from agents.plan_executor import build_plan_executor
from agents.plan import plan_agent

from dotenv import load_dotenv
load_dotenv()

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
