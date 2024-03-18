from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from getpass import getpass
OPENAI_API_KEY = getpass()

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

vectorstore = DocArrayInMemorySearch.from_texts(
    ["Avril's favorite color is pink",
     "Morning is the best time for thinking."],
     embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

template = """
Answer the question based only on the following context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser

output = chain.invoke("What is Avril's favorite color?")
print(output)