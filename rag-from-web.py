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

llm = ChatOpenAI()

# load website data
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# get embeddings
embeddings = OpenAIEmbeddings()

# build index with vectorstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# build chain
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context: 
    
    <context>
    {context}
    </context>

    Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)

# we could run this ourselves by directly passing all the docs
# but more efficient to use a retriever to fetch only the relevant docs based on the question

from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "How can langsmith help with building LLM applications?"})
print(response["answer"])



# vectorstore = DocArrayInMemorySearch.from_texts(
#     ["Avril's favorite color is pink",
#      "Morning is the best time for thinking."],
#      embedding=OpenAIEmbeddings(),
# )
# retriever = vectorstore.as_retriever()

# template = """
# Answer the question based only on the following context: {context}

# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)
# model = ChatOpenAI(model="gpt-3.5-turbo")
# output_parser = StrOutputParser()

# setup_and_retrieval = RunnableParallel(
#     {"context": retriever, "question": RunnablePassthrough()}
# )
# chain = setup_and_retrieval | prompt | model | output_parser

# output = chain.invoke("What is Avril's favorite color?")
# print(output)
