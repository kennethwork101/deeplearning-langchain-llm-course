""" 
/home/kenneth/learning/venv/lib/python3.10/site-packages/langchain/chains/llm.py:316: UserWarning: The predict_and_parse method is deprecated, instead pass an output parser directly to LLMChain.^M
"""

import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.chains import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate

physics_template = """
You are a very smart physics professor.
You are great at answering questions about physics in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.
Here is a question:
{input}
"""

math_template = """
You are a very good mathematician.
You are great at answering math questions.  You are so good because you are able to break down \RouterOutputParser
hard problems into their component parts, answer the component parts, and then put them together to answer the broader question.
Here is a question:
{input}
"""

history_template = """
You are a very good historian.
You have an excellent knowledge of and understanding of people, events and contexts from a range of historical periods.
You have the ability to think, reflect, debate, discuss and evaluate the past. You have a respect for historical evidence
and the ability to make use of it to support your explanations and judgements.
Here is a question:
{input}
"""

computerscience_template = """
You are a successful computer scientist.
You have a passion for creativity, collaboration, forward-thinking, confidence, strong problem-solving capabilities,
understanding of theories and algorithms, and excellent communication skills. You are great at answering coding questions.
You are so good because you know how to solve a problem by describing the solution in imperative steps
that a machine can easily interpret and you know how to choose a solution that has a good balance between
time complexity and space complexity. 
Here is a question:
{input}
"""

prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]

MULTI_PROMPT_ROUTER_TEMPLATE = """
Given a raw text input to a language model select the model prompt best suited for the input. 
You will be given the names of the available prompts and a description of what the prompt is best suited for. 
You may also revise the original input if you think that revising it will ultimately lead to a better 
response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
    ```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR it can be "DEFAULT" 
if the input is not well suited for any of the candidate prompts.  REMEMBER: "next_inputs" can just be 
the original input if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>
"""

@clock
@execute
def main(options):
    chat_llm = get_llm(options)

    destination_chains = {}
    for p_info in prompt_infos:
        prompt = ChatPromptTemplate.from_template(template=p_info["prompt_template"])
        destination_chains[p_info["name"]] = LLMChain(llm=chat_llm, prompt=prompt)
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=chat_llm, prompt=default_prompt)

    multi_prompt_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    multi_prompt = PromptTemplate(
        template=multi_prompt_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(chat_llm, multi_prompt)

    multi_chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True
    )

    printit("1 destinations", destinations)
    printit("2 destinations_str", destinations_str)
    printit("3 default_prompt", default_prompt)
    printit("4 default_prompt type", type(default_prompt))
    printit("5 destination_chains", destination_chains)
    printit("6 multi_prompt_template", multi_prompt_template)
    printit("7 multi_prompt", multi_prompt)
    printit("8 router_chain", router_chain)
    printit("9 multi_chain", multi_chain)

    question = options["question"]
    response = multi_chain.invoke(question)
    printit(f"1 {question}", response)

    response = multi_chain.invoke({"input": question})
    printit(f"2 {question}", response)
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='gpt4all')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='chat')
    parser.add_argument('--question', type=str, help='question', default='What is black body radiation?')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--model', type=str, help='model', default="mistral:instruct")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument('--models', nargs='+', default=[
        "codellama:7b",            
        "llama2:latest",           
        "medllama2:latest",        
        "mistral:instruct",        
        "mistrallite:latest",      
        "openchat:latest",         
        "orca-mini:latest",        
        "vicuna:latest",           
        "wizardcoder:latest",
    ])
    args = parser.parse_args()
    args = vars(args)
    return args


if __name__ == '__main__':
    options = Options()
    main(**options)