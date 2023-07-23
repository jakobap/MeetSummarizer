from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RefineDocumentsChain
from langchain.evaluation import load_evaluator, CriteriaEvalChain

from google.cloud import aiplatform
import google.auth

import secrets_1


# Defining Transcript Chunk Summarization Langchain Prompt Template.
chunk_summarization_prompt = PromptTemplate(input_variables=["attendees", "chunk_to_summarize"],
                                            template="""/
    SYSTEM: You are truthful and never lie. Never make up facts and if you are not 100% sure, reply with why you can not answer in a truthful way. 

    The following is the list of meeting attendees:
    {attendees}

    The following is a meeting transcript with the format
    [attendee]: [contribution]

    Beginning of transcript:
    {chunk_to_summarize}
    End of Transcript

    First, separate the contributions for each. of the attendees.
    Second, summarise the contribution of each of the attendee as bullet points.
    Third, bring the contribution summary per attendee in the following format:
    [attendee]: [Summarised contribution]
    [attendee]: [Summarised contribution]

    Only provide exaclty one summary per attendee.
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

Only provide exactly one summary per attendee.
Do not provide a summary for attendees that did not contribute.
"""
)

refine_summarization_prompt = PromptTemplate(
    input_variables=["attendees", "prelim_summary", "chunk_to_summarize"],
    template= """/
    SYSTEM: You are truthful and never lie. Never make up facts and if you are not 100% sure, reply with why you can not answer in a truthful way. 

    The following is the list of meeting attendees:
    {attendees}

    The following is a preliminary summary of a meeting:
    {prelim_summary}

    The following is a new part of the meeting transcript with the format
    [attendee]: [contribution]

    Beginning of transcript:
    {chunk_to_summarize}
    End of Transcript

    First, separate the contributions for each of the attendees from preliminary summary and new meeting transcript.
    Second, improve the preliminary summary with the information from the new extract.

    Provide exaclty one summary per attendee.
    Do not provide a summary for attendees that did not contribute.
    """
)

def run_simple_summarization_chain(attendees:str, prompt_chunk:str):
    # # GCP authentication via Service Account.
    # credentials, project_id = google.auth.load_credentials_from_file(
    #     secrets_1.gcp_credential_file)
    # aiplatform.init(credentials=credentials, project=project_id)

    # # Define Model to call.
    # llm = VertexAI(temperature=0, max_output_tokens=1024, top_p=0.3, top_k=20)

    llm = authenticate_vertex_llm(temperature=0, max_output_token=1024, top_p=0.3, top_k=20)

    # Define LLM Chain
    llm_chain = LLMChain(prompt=chunk_summarization_prompt, llm=llm)

    # Run Chain.
    response = llm_chain.predict(
        attendees=attendees, chunk_to_summarize=prompt_chunk)
    # print(response)

    return response


def run_meta_summarization_chain(attendees:str, summarized_chunks:str):
    # # GCP authentication via Service Account.
    # credentials, project_id = google.auth.load_credentials_from_file(
    #     secrets_1.gcp_credential_file)
    # aiplatform.init(credentials=credentials, project=project_id)

    # # Define Model to call.
    # llm = VertexAI(temperature=0, max_output_tokens=1024, top_p=0.3, top_k=20)

    llm = authenticate_vertex_llm(temperature=0, max_output_token=1024, top_p=0.3, top_k=20)

    # Define LLM Chain
    llm_chain = LLMChain(prompt=meta_summarization_prompt, llm=llm)

    # Run Chain.
    response = llm_chain.predict(
        attendees=attendees, summarized_chunks=summarized_chunks)
    # print(response)

    return response


def run_refine_documents_chain(attendees: str, prompt_chunks: list): 

    # # GCP authentication via Service Account.
    # credentials, project_id = google.auth.load_credentials_from_file(
    #     secrets_1.gcp_credential_file)
    # aiplatform.init(credentials=credentials, project=project_id)

    # # Define Model to call.
    # llm = VertexAI(temperature=0, max_output_tokens=1024, top_p=0.3, top_k=20)

    llm = authenticate_vertex_llm(temperature=0, max_output_token=1024, top_p=0.3, top_k=20)

    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )

    # Defining Initial and Refining LLM Chain.
    llm_chain_inital = LLMChain(llm=llm, prompt=chunk_summarization_prompt)
    llm_chain_refine = LLMChain(llm=llm, prompt=refine_summarization_prompt)

    chain = RefineDocumentsChain(
        initial_llm_chain=llm_chain_inital,
        refine_llm_chain=llm_chain_refine,
        document_prompt=document_prompt,
        document_variable_name="chunk_to_summarize",
        initial_response_name="prelim_summary")
    response = chain(inputs={"input_documents": prompt_chunks, "attendees": attendees})["output_text"]

    # Define Evaluation LLM & Respective string evaluation criteria.
    eval_llm = authenticate_vertex_llm()

    evaluator = load_evaluator("criteria", criteria="conciseness", llm=eval_llm)
    iterative_summary_eval = evaluator.evaluate_strings(prediction=response, input="test")
    print(iterative_summary_eval)

    return response


def authenticate_vertex_llm(temperature=.2, max_output_token=1024, top_p=.3, top_k=20):
    # GCP authentication via Service Account.
    credentials, project_id = google.auth.load_credentials_from_file(
        secrets_1.gcp_credential_file)
    aiplatform.init(credentials=credentials, project=project_id)

    # Define Model to call.
    llm = VertexAI(temperature=temperature,
                   max_output_tokens=max_output_token,
                   top_p=top_p,
                   top_k=top_k)
    return llm
