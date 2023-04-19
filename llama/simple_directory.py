from gpt_index.indices import GPTListIndex
import os
from llama_index.readers import SimpleDirectoryReader



documents = SimpleDirectoryReader('data').load_data()

print(len(documents))

# create index and query it
index = GPTListIndex.from_documents(documents)

# Save the index and load from disk 

# index.save_to_disk('index.json')
# index = GPTListIndex.load_from_disk('index.json', )

response = index.query(
    "Summarize the retrieved content in 3 points",
    response_mode="compact"
) 

print(response)
