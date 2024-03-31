""" 
LLMChain output is more detailed
dict_keys(['Review', 'English_Review', 'summary', 'language', 'followup_message']
"""

import argparse
import sys

import pandas as pd

_path = "../../"


import langchain
from kwwutils import clock, execute, get_llm, printit
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


@clock
@execute
def main(options):
    llm = get_llm(options)

    df = pd.read_csv("data.csv", delimiter=";")
    review = df.Review[3]

    template1 = "Translate the following review to english: \n\n{Review}"
    template2 =  "Can you summarize the following review in 1 sentence: \n\n{English_Review}"
    prompt1 = ChatPromptTemplate.from_template(template=template1)
    prompt2 = ChatPromptTemplate.from_template(template=template2)
    chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="English_Review")
    chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="summary")

    overall_chain = SequentialChain(
        chains=[chain1, chain2],
        input_variables=["Review"],
        output_variables=["English_Review", "summary"],
    )
    """ 
    response = overall_chain.run(review)
    response = overall_chain.invoke(input={"Review": review})
    """
    response = overall_chain.invoke({"Review": review})



def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='gpt4all')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='chat')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--model', type=str, help='model', default="llama2")
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
        "samantha-mistral:latest",        
        "vicuna:latest",           
        "wizardcoder:latest",
    ])
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    options = Options()
    main(**options)