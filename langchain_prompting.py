from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from google.cloud import aiplatform
import asyncio

import secrets_1

import google.auth


# Defining Transcript Chunk Summarization Langchain Prompt Template.
chunk_summarization_prompt = PromptTemplate(input_variables=["attendees", "prompt_chunk"],
                                            template="""/
    SYSTEM: You are truthful and never lie. Never make up facts and if you are not 100% sure, reply with why you can not answer in a truthful way. 

    The following is the list of meeting attendees:
    {attendees}

    The following is a meeting transcript with the format
    [attendee]: [contribution]

    Beginning of transcript:
    {prompt_chunk}
    End of Transcript

    First, separate the contributions for each. of the attendees.
    Second, summarise the contribution of each of the attendee as bullet points.
    Third, bring the contribution summary per attendee in the following format:
    [attendee]: [Summarised contribution]
    [attendee]: [Summarised contribution]

    Only provide one summary per attendee.
    Do not provide a summary for attendees that did not contribute.
    """
    )

meta_summarization_prompt = PromptTemplate(
    input_variables=["attendees", "summarized_chunks"],
    template= """/
    SYSTEM: You are truthful and never lie. Never make up facts and if you are not 100% sure, reply with why you can not answer in a truthful way. 

The following is the list of meeting attendees:
{attendees}

The following is a chunked summary of one single meeting with the format: 
##### [Summarization Chunk] #####
 [attendee]:
[contributions]

Beginning of Summarization Chunks:
{summarized_chunks}
End of Summarization Chunks

First, separate the contributions for each of the attendees across summarization chunks.
Second, summarise the contribution of each of the attendee as bullet points.
Third, bring the contribution summary per attendee across summarization chunks in the following format:
[attendee]: [Summarised contribution]
[attendee]: [Summarised contribution]

Only provide one summary per attendee.
Do not provide a summary for attendees that did not contribute.
"""
)

# async def async_generate(chainattendees, ):
#     resp = chain.arun(attendees=attendees, prompt_chunk=prompt_chunk)
#     print(resp)

# async def generate_concurrently(prompt, transcript_object):
#     llm = VertexAI(temperature=0, max_output_tokens=1024, top_p=0.3, top_k=20)
#     chain = LLMChain(llm=llm, prompt=prompt)
#     tasks = [chain.arun(attendees=transcript_object.attendees, prompt_chunk=chunk) for chunk in transcript_object.prompt_chunks]
#     await asyncio.gather(*tasks)


def run_langchain(attendees, prompt_chunk):
    # GCP authentication via Service Account.
    credentials, project_id = google.auth.load_credentials_from_file(
        secrets_1.gcp_credential_file)
    aiplatform.init(credentials=credentials, project=project_id)

    # Define Model to call.
    llm = VertexAI(temperature=0, max_output_tokens=1024, top_p=0.3, top_k=20)

    # Define LLM Chain
    llm_chain = LLMChain(prompt=chunk_summarization_prompt, llm=llm)

    # Run Chain.
    response = llm_chain.predict(
        attendees=attendees, prompt_chunk=prompt_chunk)
    # print(response)

    return response


def run_meta_summarization_chain(attendees, summarized_chunks):
    # GCP authentication via Service Account.
    credentials, project_id = google.auth.load_credentials_from_file(
        secrets_1.gcp_credential_file)
    aiplatform.init(credentials=credentials, project=project_id)

    # Define Model to call.
    llm = VertexAI(temperature=0, max_output_tokens=1024, top_p=0.3, top_k=20)

    # Define LLM Chain
    llm_chain = LLMChain(prompt=meta_summarization_prompt, llm=llm)

    # Run Chain.
    response = llm_chain.predict(
        attendees=attendees, summarized_chunks=summarized_chunks)
    # print(response)

    return response
