from llama_index.readers import ChatGPTRetrievalPluginReader
from llama_index.indices.vector_store import ChatGPTRetrievalPluginIndex
from gpt_index.indices import GPTListIndex
import os
from IPython.display import Markdown, display

# load documents
bearer_token = os.getenv("BEARER_TOKEN")
reader = ChatGPTRetrievalPluginReader(
    endpoint_url="http://localhost:8000",
    bearer_token=bearer_token
)

documents = reader.load_data("What song is about a Big hunk of love?")
print(documents)
print(len(documents))

# convert Document objects to Node objects

# create index and query it
index = GPTListIndex.from_documents(documents)
response = index.query(
    "Summarize the retrieved content",
    response_mode="compact"
) 


# index = ChatGPTRetrievalPluginIndex.from_documents(
#     documents, 
#     endpoint_url="http://localhost:8000",
#     bearer_token=bearer_token,
# )

#query index
# response = index.query("Summarize the content of the song.", similarity_top_k=3, response_mode='compact')

# display(Markdown(f"<b>{response}</b>"))
print(response)
