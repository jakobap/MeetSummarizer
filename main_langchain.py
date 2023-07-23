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
      output.append(langchain_prompting.run_simple_summarization_chain(attendees=transcript_object.attendees, prompt_chunk=prompt_chunk))

   ts = int(datetime.datetime.now().timestamp() * 1000)
   output_file_path = f"./output_{ts}.txt"
   write_to_file(output, output_file_path)
   
   with open(output_file_path, "r") as f:
     summarized_chunks_string = f.read()

   final_summary = langchain_prompting.run_meta_summarization_chain(attendees=transcript_object.attendees,
                                                                    summarized_chunks=summarized_chunks_string)
   
   iterative_summary = langchain_prompting.run_refine_documents_chain(attendees=transcript_object.attendees,
                                                                      prompt_chunks=transcript_object.prompt_chunks)
   
   write_to_file([final_summary], f"./meta_output_{ts}.txt")
   write_to_file([iterative_summary], f"./iterative_meta_output_{ts}.txt")   
   
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