from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from google.cloud import aiplatform

import secrets_1

import google.auth


# Defining Transcript Chunk Summarization Langchain Prompt Template.

def run_langchain(attendees, prompt_chunk):
    # GCP authentication via Service Account.
    credentials, project_id = google.auth.load_credentials_from_file(
        secrets_1.gcp_credential_file)
    aiplatform.init(credentials=credentials, project=project_id)

    # Define Model to call.
    llm = VertexAI(temperature=0,max_output_tokens=1024,top_p=0.3,top_k=20)

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

    # Define LLM Chain
    llm_chain = LLMChain(prompt=chunk_summarization_prompt, llm=llm)

    # Run Chain.
    response = llm_chain.predict(attendees=attendees, prompt_chunk=prompt_chunk)
    # print(response)

    return response
