from langchain_openai import OpenAI

from getpass import getpass
OPENAI_API_KEY = getpass()

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = OpenAI(model_name="text-embedding-ada-002")
print(llm("Explain me briefly what KDNuggets is."))