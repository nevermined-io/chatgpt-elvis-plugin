from llama_index.readers import ChatGPTRetrievalPluginReader
from llama_index.indices.vector_store import ChatGPTRetrievalPluginIndex
import os


# load documents
bearer_token = os.getenv("BEARER_TOKEN")

reader = ChatGPTRetrievalPluginReader(
    endpoint_url="http://localhost:8000",
    bearer_token=bearer_token
)
documents = reader.load_data("*")


print(len(documents))

index = ChatGPTRetrievalPluginIndex.from_documents(
    documents,
    endpoint_url="http://localhost:8000",
    bearer_token=bearer_token,
)

#query index
response = index.query("Summarize the content of the song.",  response_mode='compact')

print(response)
