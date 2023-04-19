from llama_index.readers import ChatGPTRetrievalPluginReader
from llama_index.indices.vector_store import ChatGPTRetrievalPluginIndex
import os
from llama_index import GPTSimpleVectorIndex


# load documents
bearer_token = os.getenv("BEARER_TOKEN")

reader = ChatGPTRetrievalPluginReader(
    endpoint_url="http://localhost:8000",
    bearer_token=bearer_token
)
documents = reader.load_data("Adam and Evil,", top_k=1, separate_documents=True)



# print(documents)

# index = ChatGPTRetrievalPluginIndex.from_documents(
#     documents,
#     endpoint_url="http://localhost:8000",
#     bearer_token=bearer_token,
# )

index = GPTSimpleVectorIndex.from_documents(documents)
# index.save_to_disk('index.json')

# index = GPTSimpleVectorIndex.load_from_disk('index.json')


#query index
response = index.query("Summarize the content of the song.",  response_mode='compact')

print(response)
