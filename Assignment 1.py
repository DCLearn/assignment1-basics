#unicode =  byte sequence  via the UTF-8 encoding
#unicode can be converted to integer with ord()
#then integer can be converted to character with chr()
print(chr(0))
print(repr(chr(0)))
print("this is" +chr(0)+"string")

# string can be converted to unicode with .encode()
# utf-8 unicode can be converted to string with .decode()
test_string = "Hello World!"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
print(list(utf8_encoded))
print(len(list(utf8_encoded)))
print(utf8_encoded.decode("utf-8"))

#problem 2.3
def decode_utf_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

decode_utf_bytes_to_str_wrong("Hello World1234".encode("utf-8"))
# .encode("utf-8") cannot convert numbers to unicode even as strings

# letter for letter tokenization in unicode is too large, use subword (prefix, suffix, articles, roots) instead
#byte-pair encoding (BPE) tokenizers or subword tokenizers have an initial vocab size of 256
#pre-tokenize the corpus test by counting occurences of pairs of characters

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import regex as re

print(re.findall(PAT, "some text that i'll pre-tokenize. Hello World but pre-tokenized"))
#use re.finditer() is better
#most common pairs are merged into tokens then, repeated using tokens
#When computing merges, deterministically break ties in pair frequency using max(). 
# treat some strings as “special tokens” that should never be split 
#re.split with "|" ⌋.join(special_tokens) as the delimiter



import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

import os
from typing import BinaryIO

from multiprocessing import Pool

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
with open(..., "rb") as f:
    boundaries = find_chunk_boundaries(
        f, num_processes, "<|endoftext|>".encode("utf-8"))
        
    # The following is a serial implementation, but you can parallelize this 
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token


# before using pretokemization, remove special tokens from chunk/corpus (ex. <|endoftext|> ) using re.split with "|" ⌋.join(special_tokens)
# test with test_train_bpe_special_tokens
#