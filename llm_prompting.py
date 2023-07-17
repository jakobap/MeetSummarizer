from google.cloud import aiplatform
import vertexai

import os
import secrets_1

vertexai.init(project=secrets_1.gcp_project_id, location="us-central1")
parameters = {
    "temperature": 0,
    "max_output_tokens": 1024,
    "top_p": 0.3,
    "top_k": 20
}

from vertexai.language_models import TextGenerationModel

def text_model_api_call(transcript):

    model = TextGenerationModel.from_pretrained("text-bison")

    # # Get the path to the txt file
    # file_path = os.path.join(os.path.dirname(__file__), "transcript_part1.txt")

    # # Open the file
    # with open(file_path, "r") as f:
    #     transcript = f.read()

    # # Print the transcript
    # print(transcript)

    # Constructing the prompt
    prompt = f"""SYSTEM: You are truthful and never lie. Never make up facts and if you are not 100% sure, reply with why you can not answer in a truthful way. 

    The following is the list of meeting attendees:
    Borys Papevis, David Ullrich, Eslam Gamal Abdalla Elhosary, Jakob Pörschmann, Jakob Pörschmann\'s Presentation, João Cunha, Johny Jose, Levan Tsinadze, Mario König, Oleksandr Yuzva, Paul Gerlach, Rose Howell, Rupak Chakraborty, Tamer Temizer

    The following is a meeting transcript with the format
    [attendee]: [contribution]

    First, separate the contributions for each. of the attendees.

    Beginning of transcript:
    {transcript}
    End of Transcript

    First, separate the contributions for each. of the attendees.
    Second, summarise the contribution of each of the attendee
    Third, bring the contribution summary per attendee in the following format:
    [attendee]: [Summarised contribution]
    [attendee]: [Summarised contribution]

    Only provide one summary per attendee.
    Don\'t provide a summary for attendees that did not contribute."""

    # Print completed prompt
    # print(prompt)

    # print number of words of prompt 


    # print(f"Prompt Num Words: {count_words(prompt)}")

    # Call LLM & print model response
    response = model.predict(prompt,**parameters)
    # print(f"Response from Model: {response.text}")

    print("Text Model Call Sucessful")
 
    return response.text