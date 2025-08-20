from src.utils import GraphState
from src.prompt_template import planing_system_message, planing_human_message, planing_input_variables
from src.utils import PlanFormat

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
import os

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

def plan_agent(state: GraphState):
    API_KEY = os.getenv("API_KEY")
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
