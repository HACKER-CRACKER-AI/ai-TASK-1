from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor

import torch
import time
import transformers
from transformers import pipeline
from langchain.llms import HuggingFacePipeline, OpenAIChat
from langchain.chains import SQLDatabaseSequentialChain
from langchain.chains import SQLDatabaseChain
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from langchain.chat_models import ChatOpenAI

from langchain.chains import SQLDatabaseChain

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory


from dotenv import load_dotenv
from langchain import OpenAI, SQLDatabase
import psycopg2
import os

import speech_recognition as sr

load_dotenv()


OPENAI_API_KEY = ''
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#llm = OpenAI(temperature=0)
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')

'''
toolkit = SQLDatabaseToolkit(db=db,llm=llm)

agent_executor = create_sql_agent(
    	llm=llm,
    	toolkit=toolkit,
    	verbose=True)
'''

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Listening...")
    audio = recognizer.listen(source)


# Usage:
db = SQLDatabase.from_uri('')
print(db)

#sql_chain = MySQLDatabaseChain(db, question, house)


#chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

try:
    question = recognizer.recognize_google(audio)
    print(f"You said: {question}")
    PROMPT = """ 
Given an input question, first create a syntactically correct postgresql query to run,  
then look at the results of the query and return the answer.  
The question: {question}
Use the following format:

Question: "The question here"
postgresqlQuery: "postgresql Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Examples:

Question: "How many houses listed are not furnished?"
postgresqlQuery: "SELECT COUNT(*) 
FROM house 
WHERE furnishing = 'Unfurnished' OR furnishing = 'Semi-Furnished';"
SQLResult: "164472"
Answer: "164472 houses listed are not furnished."

Question: "How many houses listed are unfurnished?"
postgresqlQuery: "SELECT COUNT(*) 
FROM house 
WHERE furnishing = 'Unfurnished';"
SQLResult: "76154"
Answer: "76154 houses listed are unfurnished."

Question: "What is the average price of a house?"
postgresqlQuery: "SELECT AVG(houseprice) 
FROM house;"
SQLResult: "7583.7718848975074470"
Answer: "The average price of a house is 7583.7718848975074470."

Question: "What is the average price of furnished house?"
postgresqlQuery: "SELECT AVG(houseprice) 
FROM house
WHERE furnishing = 'Furnished';"
SQLResult: "7946.1797646271403871"
Answer: "The average price of furnished house is 7946.1797646271403871."
	"""


    chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

    chain.run(PROMPT.format(question=question))
    #agent_executor.run(PROMPT.format(question=question))
except sr.UnknownValueError:
    print("Sorry, I could not understand your audio.")
except sr.RequestError:
    print("Sorry, I'm having trouble connecting to the Google API.")

    