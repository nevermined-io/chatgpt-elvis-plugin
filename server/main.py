import os
import random
from typing import List, Optional
from llama_index import Document, VectorStoreIndex
import uvicorn
from llama_index.readers import ChatGPTRetrievalPluginReader
from starlette.responses import Response
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from loguru import logger
import random

from models.api import (
    DeleteRequest,
    DeleteResponse,
    QueryRequest,
    QueryResponse,
    UpsertRequest,
    UpsertResponse,
)
from datastore.factory import get_datastore
from services.file import get_document_from_file

from models.models import DocumentMetadata, Source

NVM_CREDITS_RESP_HEADER = "NVMCreditsConsumed"

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials


app = FastAPI(dependencies=[Depends(validate_token)])
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")

# Create a sub-application, in order to access just the query endpoint in an OpenAPI schema, found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="Nevermined Retrieval Plugin API",
    description="A retrieval API for querying and filtering Nevermined documents based on natural language queries and metadata",
    version="0.1.0",
    servers=[{"url": "https://zparolnv3lqwtjk26xfkgustpbbk9gz9so4x30zfe5v8x8wqs.proxy.goerli.nevermined.app/"}],
    dependencies=[Depends(validate_token)],
)
app.mount("/sub", sub_app)


@app.post(
    "/upsert-file",
    response_model=UpsertResponse,
)
async def upsert_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    try:
        metadata_obj = (
            DocumentMetadata.parse_raw(metadata)
            if metadata
            else DocumentMetadata(source=Source.file)
        )
    except:
        metadata_obj = DocumentMetadata(source=Source.file)

    document = await get_document_from_file(file, metadata_obj)

    try:
        ids = await datastore.upsert([document])
        return UpsertResponse(ids=ids)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=f"str({e})")


@app.post(
    "/upsert",
    response_model=UpsertResponse,
)
async def upsert(
    request: UpsertRequest = Body(...),
):
    try:
        ids = await datastore.upsert(request.documents)
        return UpsertResponse(ids=ids)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query_main(
    request: QueryRequest = Body(...),
):
    try:
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results, headers={NVM_CREDITS_RESP_HEADER: "1"})
    
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@sub_app.post(
    "/query",
    response_model=QueryResponse,
    # NOTE: We are describing the shape of the API endpoint input due to a current limitation in parsing arrays of objects from OpenAPI schemas. This will not be necessary in the future.
    description="Accepts the title of a Elvis song as a search query objects array each with query and optional filter. Break down complex questions into sub-questions. Refine results by criteria, e.g. time / source, don't do this often. Split queries if ResponseTooLargeError occurs.",
)
async def query(
    request: QueryRequest = Body(...),
):
    try:
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.delete(
    "/delete",
    response_model=DeleteResponse,
)
async def delete(
    request: DeleteRequest = Body(...),
):
    if not (request.ids or request.filter or request.delete_all):
        raise HTTPException(
            status_code=400,
            detail="One of ids, filter, or delete_all is required",
        )
    try:
        success = await datastore.delete(
            ids=request.ids,
            filter=request.filter,
            delete_all=request.delete_all,
        )
        return DeleteResponse(success=success)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/ask",
    # response_model=QueryResponse,
    # NOTE: We are describing the shape of the API endpoint input due to a current limitation in parsing arrays of objects from OpenAPI schemas. This will not be necessary in the future.
    description="Given an Elvis song title it summarizes the lyrics of that song. Break down complex questions into sub-questions. Refine results by criteria, e.g. time / source, don't do this often. Split queries if ResponseTooLargeError occurs.",
)
async def ask_song(
    request: QueryRequest = Body(...)
):
    try:
        documents: List[Document] = []
        results = await datastore.query(
            request.queries,
        )
        print(results)
        for query_result in results:
            for result in query_result.results:
                result_id = result.id
                result_txt = result.text
                result_embedding = result.embedding
                document = Document(
                    text=result_txt,
                    doc_id=result_id,
                    embedding=result_embedding,
                )
                documents.append(document)

            # NOTE: there should only be one query
            break
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        content = query_engine.query("Summarize the content of the song.")        
        print("Content: " + content.__str__())
        # return response
        headers = {NVM_CREDITS_RESP_HEADER: random.randint(1, 5)}
        print("Response Headers: " + headers.__str__())

        return Response(content=content.__str__(), headers=headers)
            
        # return QueryResponse(content=content, headers=headers)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")

@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore()    


def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
