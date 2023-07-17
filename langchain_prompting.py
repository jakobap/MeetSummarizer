from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from google.cloud import aiplatform

import secrets_1

import google.auth

credentials, project_id = google.auth.load_credentials_from_file(secrets_1.gcp_credential_file)

aiplatform.init(credentials=credentials, project=project_id)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = VertexAI()

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

response = llm_chain.run(question)

print(response)
print("Hello World!")