from rsc import transcript
from rsc import langchain_prompting

import secrets_1

# import datetime

import google.auth
from google.cloud import storage

class SummarizationSession:

  def __init__(self, timestamp, file_path=None, file_blob=None):
    """
    Constructor.

    Args:
      file_path: The path to the transcript file.
    """
    self.file_path: str = file_path
    self.file_blob = file_blob
    self.ts = timestamp

  def __call__(self):

    transcript_object = transcript.Transcript(file_path=self.file_path,
                                              file_blob=self.file_blob,
                                              timestamp=self.ts).load()

    output = []
    for count, prompt_chunk in enumerate(transcript_object.prompt_chunks):
        output.append(f"##### Summarization Chunk {count} #####")
        output.append(langchain_prompting.run_simple_summarization_chain(attendees=transcript_object.attendees, prompt_chunk=prompt_chunk))

    prelim_output = self._write_to_gcs(output, txt_name='prelim_output', bucket=secrets_1.prelim_output_bucket)
    summarized_chunks_string = prelim_output.download_as_text()

    sequential_summary = langchain_prompting.run_meta_summarization_chain(attendees=transcript_object.attendees,
                                                                      summarized_chunks=summarized_chunks_string)
    meta_output = self._write_to_gcs(list_of_strings=[sequential_summary], txt_name='meta_output_', bucket=secrets_1.meta_output_bucket)

    # iterative_summary = langchain_prompting.run_refine_documents_chain(attendees=transcript_object.attendees,
    #                                                                     prompt_chunks=transcript_object.prompt_chunks)    
    # iterative_meta_output = self._write_to_gcs(list_of_strings=[iterative_summary], txt_name='iterative_meta_output_', bucket=secrets_1.iterative_meta_output_bucket)
    return sequential_summary

  def _write_to_file(self, list_of_strings, file_path) -> None:
    """Writes a list of strings to a txt file.

    Args:
      list_of_strings: A list of strings to write to the file.
      file_path: The path to the file to write to.
    """

    with open(file_path, "w") as f:
      for string in list_of_strings:
        f.write(string + "\n")
      
      return None
    
  def _write_to_gcs(self, list_of_strings, txt_name, bucket) -> None:
    """Writes a list of strings to a txt file.

    Args:
      list_of_strings: A list of strings to write to the file.
      bucket: The bucket path to write to.
    """
    credentials, project_id = google.auth.load_credentials_from_file(secrets_1.gcp_credential_file)

    bucket = storage.Client(project=project_id, credentials=credentials).bucket(bucket)
    blob = bucket.blob(f'{self.ts}_{txt_name}.txt')
    blob.upload_from_string(', '.join(list_of_strings))
    return blob


if __name__ == '__main__':
   
   session = SummarizationSession(file_path="./rsc/transcript1_raw.txt")

   session()

   print("Hello World!")