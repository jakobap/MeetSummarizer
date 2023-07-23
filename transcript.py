from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Transcript:
  """
    The Transcript class represents a transcript of a conversation. It can be used to load a transcript from a file, split it into chunks, and count the number of tokens in it.

    Attributes:
        file_path (str): The path to the transcript file.
        transcript_string (str): The transcript string.
        approx_total_tokens (int): The approximate total number of tokens in the transcript.
        prompt_chunks (list): A list of chunks of the transcript.

    Methods:
        load(self): Load the transcript from the file.
        _approx_tokens(self, prompt: str) -> int: Count the number of tokens in a palm 2 bison prompt.
        _split_transcript_into_chunks(self, transcript:str, chunk_size:int = 28000) -> list: Splits the input transcript into chunks of the specified size.
  """
  def __init__(self, file_path):
    """
    Constructor.

    Args:
      file_path: The path to the transcript file.
    """

    self.file_path: str = file_path
    self.clean_file_path: str = None
    self.transcript_string: str = None
    self.attendees: str = None
    self.approx_total_tokens: int = None
    self.prompt_chunks: list = None

  def load(self):
    """
    Load the transcript from the file.
    """

    # Extract attendees from raw transcript
    self._get_attendees()

    # Clean and load transcript into workable str format
    self._clean()

    with open(self.clean_file_path, "r") as f:
      self.transcript_string = f.read()

    self.approx_total_tokens = self._approx_tokens(self.transcript_string)
    self.prompt_chunks = self._split_transcript_into_chunks(self.transcript_string)
    # self.prompt_chunks_dict = {i: chunk for i, chunk in enumerate(self.prompt_chunks)}

    return self
  
  def _approx_tokens(self, prompt: str) -> int:
    """
    Counts the number of tokens in a palm 2 bison prompt.

    Args:
        prompt: The palm 2 bison prompt.

    Returns:
        The number of tokens in the prompt.
    """

    # Split the prompt into words.
    words = prompt.split()

    # Count the number of words.
    num_words = len(words)

    # print(f"Num Words: {num_words}")

    # For the PaLM 2 model, token is equivalent to about 4 characters. 100 tokens are about 60-80 English words.
    # Palm input token limit = 8196
    # Input word limit approx. (8196/100) * 60 = 4917.59
    # Input character limit approx. (8196 * 4 ) = 32768
    approx_num_tokens = int(num_words * (100/60))
    
    return approx_num_tokens
  
  def _split_transcript_into_chunks(self, transcript:str, chunk_size:int = 28000) -> list:
    """
    Splits the input transcript into chunks of the specified size.

    Args:
        transcript: The input transcript.
        chunk_size: The size of the chunks.

    Returns:
        A list of chunks.
    """

    # Generic chunking based on character count
    # documents = []
    # for chunk_count, i in enumerate(range(0, len(transcript), chunk_size)):
    #    documents.append(Document(page_content=transcript[i:i + chunk_size], metadata={"chunk": chunk_count}))


    # Langchain Document Splitter.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                                   chunk_overlap  = 50,
                                                   length_function = len
                                                   )
    
    documents = text_splitter.create_documents([self.transcript_string])

    return documents
  
  def _get_attendees(self):
    """
    Gets the participants from the transcript.

    Returns:
        A string of all meeting participants.
    """
    with open(self.file_path, "r") as f:
        lines = f.readlines()
        
        index = lines.index("Attendees\n")
        
        self.attendees = lines[index + 1]
        
        # print(self.attendees)

    return self
  
  def _clean(self):
    """
    Cleans the raw transcript down to pure transcripted meeting contributions.

    Returns:
        A clean transcript string.
    """
    with open(self.file_path, "r") as f:
        lines = f.readlines()        
        clean_lines = lines[5:-1]
        # print(clean_lines)

    self.clean_file_path = f"{self.file_path.split('.txt')[0]}_cleaned.txt"

    with open(self.clean_file_path, "w") as f:
        f.writelines(clean_lines)

    return self
  

if __name__ == '__main__':
   
#    transcript = Transcript("./transcript_full.txt").load()
   transcript = Transcript("./transcript1_raw.txt").load()

   print("Hello World!")
