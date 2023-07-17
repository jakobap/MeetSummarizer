import transcript
import llm_prompting
import langchain_prompting

import datetime

def main(transcript_path: str) -> None:
   # Create Transcript Object from Input
   transcript_object = transcript.Transcript(transcript_path).load()

   output = []
   for count, prompt_chunk in enumerate(transcript_object.prompt_chunks):
      output.append(f"##### Summarization Chunk {count} #####")
      output.append(langchain_prompting.run_langchain(attendees=transcript_object.attendees, prompt_chunk=prompt_chunk))

   write_to_file(output, f"./output_{int(datetime.datetime.now().timestamp() * 1000)}.txt")

   return None

def write_to_file(list_of_strings, file_path) -> None:
  """Writes a list of strings to a txt file.

  Args:
    list_of_strings: A list of strings to write to the file.
    file_path: The path to the file to write to.
  """

  with open(file_path, "w") as f:
    for string in list_of_strings:
      f.write(string + "\n")
    
    return None


if __name__ == '__main__':

   main("./transcript1_raw.txt")

   print("Hello World!")