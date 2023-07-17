import transcript
import llm_prompting

def main(transcript_path: str):
   # Create Transcript Object from Input
   transcript_object = transcript.Transcript(transcript_path).load()

   for i in transcript_object.prompt_chunks:
      llm_prompting.text_model_api_call(i)

if __name__ == '__main__':

   main("./transcript_full.txt")

   print("Hello World!")
