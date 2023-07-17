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

    self.file_path = file_path
    self.transcript_string = None
    self.approx_total_tokens = None
    self.prompt_chunks = None

  def load(self):
    """
    Load the transcript from the file.
    """

    with open(self.file_path, "r") as f:
      self.transcript_string = f.read()

    # self.attendees = self._get_attendees()
    # self.contributions = self._get_contributions()

    self.approx_total_tokens = self._approx_tokens(self.transcript_string)
    self.prompt_chunks = self._split_transcript_into_chunks(self.transcript_string)

    return None
  
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

    print(f"Num Words: {num_words}")

    #  For the PaLM 2 model, token is equivalent to about 4 characters. 100 tokens are about 60-80 English words.
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

    chunks = []
    for i in range(0, len(transcript), chunk_size):
        chunks.append(transcript[i:i + chunk_size])

        print(f"Chunk {i}")
        tk = self._approx_tokens(chunks[-1])
        print(f"Num Tokens: {tk}")
        print("####")

    return chunks


if __name__ == '__main__':
   
   transcript = Transcript("./transcript_full.txt").load()

   print("Hello World!")
