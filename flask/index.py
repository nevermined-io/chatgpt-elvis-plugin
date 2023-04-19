from flask import Flask
from llama_index.readers import ChatGPTRetrievalPluginReader
from flask import request
import os
from gpt_index.indices import GPTListIndex

app = Flask(__name__)

bearer_token = os.getenv("BEARER_TOKEN")
index = None



@app.route("/")
def home():
    return "Hello World!"

@app.route("/index", methods=["GET"])
def index():
   global index
   reader = ChatGPTRetrievalPluginReader(
                endpoint_url="http://localhost:8000",
                bearer_token=bearer_token
            )
   document = reader.load_data("*")
   index = GPTListIndex.from_documents(document)
   return f"""Loaded { len(document) } documents"""


@app.route("/query", methods=["GET"])
def query_index():
  global index
  query_text = request.args.get("text", None)
  if query_text is None:
    return "No text found, please include a ?text=blah parameter in the URL", 400
  response = index.query(query_text)
  return str(response), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)