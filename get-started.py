from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from getpass import getpass
OPENAI_API_KEY = getpass()

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

prompt = ChatPromptTemplate.from_template("Create a short poem about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

user_input = input("Give me a complex topic that you would like to turn into a poem: \n")

chain = prompt | model | output_parser
output = chain.invoke({"topic": user_input})
print(output)