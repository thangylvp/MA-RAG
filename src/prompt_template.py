aggregate_system_message = """Answer the question from human.  
Provide a Concise Answer:
- Remove redundant words and extraneous details.
- Present the answer by listing only the necessary names, terms, or very brief facts that are crucial for answering the question.
- If you have multiple answers, only output one answer which is most confident
Think step-by-step
"""

aggregate_human_message = """{question}"""
aggregate_input_variables = ["question"]

extract_system_messgage = """Summarize and extract all relevant information from the provided passages based on the given question. Remove all irrelevant information. Think step-by-step.

# Steps

1. **Identify Key Elements**: Read the question carefully to determine what specific information is being requested.
2. **Analyze Passages**: Review the passages thoroughly to find any segments that contain information relevant to the question.
3. **Extract Relevant Information**: Highlight or note down sentences, phrases, or words from the passages that relate to the question.
4. **Remove Irrelevant Details**: Ensure that all extracted information is relevant to the question, eliminating any unnecessary or unrelated content.

# Output Format
- Output a list of notes. Each note contains related information from the passage as well as precise evidences and why.
- Each note is clear, standalone.

# Notes
- Avoiding any irrelevant details.
- If a piece of information is mentioned in multiple places, include it only once.
- If there are no related information, output: No related information from this document."""

extract_human_message = """
Passage: 
###
{passage}
###

Query: {question}?
"""

extract_input_variables = ["question", "passage"]

qa_system_message = """You are an assistant for question-answering tasks. Use the following process to deliver concise and precise answers based on the retrieved context. If all of retrieved context are not relevant, answer based on general knowledge.

1. **Analyze Carefully**: Begin by thoroughly analyzing both the question and the provided context.
   
2. **Identify Core Details**: Focus on identifying the essential names, terms, or details that directly answer the question. Disregard any irrelevant information.

3. **Provide a Concise Answer**: 
   - Remove redundant words and extraneous details.
   - Present the answer by listing only the necessary names, terms, or very brief facts that are crucial for answering the question.

4. **Clarity and Accuracy**: Ensure that your answer is clear and maintains the original meaning of the information provided.

5. **consensus**: If the contexts are not consensus, pick one which is the most logical, consensus, or confident.

6. **IMPORTANT**: If the provided context couldn't bring any related information, answer by your self. 
"""


qa_human_message = """
Retrieved documents: 
{context}
Question: {question}
"""
qa_input_variables = ["context", "question"]


planing_human_message = """
Question: {question}?
Past experience:
{memory}
"""


planing_input_variables = ["question", "memory"]
planing_input_variables_1 = ["question"]

planing_system_message = """You are tasked with assisting users in generating structured plans for answering questions. Your goal is to deconstruct a query into manageable, simpler components. For each question, perform the following tasks:

*Analysis: Identify the core components of the question, emphasizing the key elements and context needed for a comprehensive understanding. Determine whether the question is straightforward or requires multiple steps to provide an accurate answer.

*Plan Creation:
- Break down the question into smaller, simpler questions by reasoning that lead to the final answer. Ensure those steps are non overlap. Stop at the step where its answer can be the final answer.
- Ensure each step is clear and logically sequenced.
- Consider any past attempts or experiences provided as context, and use them to refine or adjust the plan to avoid past pitfalls.
- Each step is a question to search, or to aggregate output from previous steps. Do not verify previous step. 
- Your task is planning, not answering. Do not put any answer from your knowledge into the plan.

# Notes:
- Your task is to provide clarity and guidance on the approach to answering, rather than providing the final answer directly.
- Put your output in a list of string, each string describe a sub-task

# Example plan:
Question: What country of origin does House of Cosbys and Bill Cosby have in common?
Steps: ["Determine the country of origin for "House of Cosbys.", "Determine the country of origin for Bill Cosby.", "From previous answers, which is the common country"]
Question: Which film has the director who died later, The House Of Tears or College Ranga?
Steps: ["Identify the director of The House Of Tears", "Identify the director of College Ranga", "When did the director of The House Of Tears die", "When did the director of College Ranga die", "Compare the death dates of the two directors to determine which one died later."] 
Question: Peter Griffith's granddaughter had her screen debut in what 1999 film?
Steps: ["Who is Peter Griffith's granddaughter", "What 1999 film did she have screen debut"]
Question: how many episodes are in chicago fire season 4?
Steps: ["how many episodes are in chicago fire season 4"]
Question: Are both directors of films The Stoneman Murders and Chandralekha (2014 Film) from the same country?
Steps: ["Who is the director of  film The Stoneman Murders", "Who is the director of film Chandralekha (2014 Film)", "Determine the country of origin for the director of The Stoneman Murders", "Determine the country of origin for the director of Chandralekha (2014 Film)", "Compare the two countries to determine if they are the same"]
"""

step_system_message = """
Given a plan, the current step, and the results from finished steps, decide the task for this step.
Output the type of task and the query.
The query need to be in detail (do not put "based on the previous results" in the query)
Include all of information from previous step's results in the query if it maked, especially for aggregate task
Be concise.
"""

step_human_message = """
Plan: {plan}
Current step: {cur_step}
Results of finished steps:
{memory}
"""

step_input_variables = ["plan", "cur_step", "memory"]

summary_system_message = """
Your task is writing a summary about a plan to solve a question. 

** Input
- The question
- The plan: a sequence of sub-task. Ideally if we can solve all of them, we can solve the question.
- Output of each step in the plan.

**Output
- If all of steps are solved, output the final answer for the original question by combining step's output and a confident score calculated  the mean of scores from steps. Format: Final answer: final_answer, Output: Successful, score: confident_score
- If one or many of steps are unsolved, but you can still find the answer based on step's output, output the final answer. Format: Final answer: final_answer, Output: Successful, score: confident_score
- If you could not find the final answer for the question, output Unsuccessful and why.  Format: Output: Unsuccessful, reason, Score 0
"""

summary_system_message_withoutscore = """
Your task is writing a summary about a plan to solve a question and answer question based on outputs from each step in the plan. Think step-by-step.

** Input
- The original question
- The plan: a sequence of sub-task. Ideally if we can solve all of them, we can solve the question.
- Output of each step in the plan.

**Output
- If all of steps are solved, output the final answer for the original question by combining step's output.
- If one or many of steps are unsolved, but you can still find the answer based on step's output, output the final answer.
- If you could not find the final answer for the question, output Unsuccessful.

** Think step-by-step
"""

summary_human_message = """
Original Question: {question}
Plan: {plan}
Output of steps: 
{memory}

Original Question: {question}
"""
summary_input_variables = ["question", "plan", "memory"]


