import os
from llama_index import GPTSimpleVectorIndex, download_loader


# load documents
bearer_token = os.getenv("BEARER_TOKEN")

ElasticsearchReader = download_loader("ElasticsearchReader")
elastic_username=os.getenv("ELASTIC_USERNAME")
elastic_password = os.getenv("ELASTIC_PASSWORD")
reader = ElasticsearchReader(
    f"https://{elastic_username}:{elastic_password}@nvm-elastic.es.europe-west3.gcp.cloud.es.io/",
    "hairdao"
)

query_dict = {}
documents = reader.load_data(
    "content", query=query_dict
)
print(len(documents))
# convert Document objects to Node objects

# create index and query it
index = GPTSimpleVectorIndex.from_documents(documents)

#query index
response = index.query("Is Camellia Seed Cake good for hair growth?", response_mode='compact')

print(response)
