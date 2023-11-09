from fastapi import FastAPI, UploadFile, File, Form, APIRouter, Depends,  HTTPException
from fastapi.responses import JSONResponse
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.text import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import FakeEmbeddings
import os
import csv
import io
from bson import ObjectId 
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel
import shutil
from typing import List
from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from langchain.vectorstores import Vectara
from langchain.vectorstores.vectara import VectaraRetriever
from fastapi.middleware.cors import CORSMiddleware
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import logging
import openai
from langchain.chains import LLMChain

print(openai.Completion)

load_dotenv()
router = APIRouter()
logger = logging.getLogger("uvicorn.error")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY set for Flask application")


# Set up environment variables directly (comment these out if you want to use getpass instead)
os.environ["VECTARA_CUSTOMER_ID"] = "154638175"  # Replace with your Vectara Customer ID
os.environ["VECTARA_CORPUS_ID"] = "1"  # Replace with your Vectara Corpus ID
os.environ["VECTARA_API_KEY"] = "zwt_CTeXX0UrDKzSp3v9FG-Vo0L6V8b5bmM5dMn9uQ"  # Replace with your Vectara API Key

# Initialize Vectara client
vectara_client = Vectara(
    vectara_customer_id=os.getenv("VECTARA_CUSTOMER_ID"),
    vectara_corpus_id=os.getenv("VECTARA_CORPUS_ID"),
    vectara_api_key=os.getenv("VECTARA_API_KEY")
)
VECTARA_CUSTOMER_ID = os.getenv("154638175")
VECTARA_CORPUS_ID = os.getenv("1")
VECTARA_API_KEY = os.getenv("zwt_CTeXX0UrDKzSp3v9FG-Vo0L6V8b5bmM5dMn9uQ")


uri = "mongodb+srv://nkugwamarkwilliam:UJbXVgwpyPZ7I3Iu@cluster0.vndslyl.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["MosiAi"]  # use your actual database name here
collection = db["Chatbots"] 
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


class QueryModel(BaseModel):
    user_question: str



app = FastAPI()

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



def process_chat_with_langchain(question, context):
    # Initialize the LLM (OpenAI GPT model) with your API key
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    # Use Langchain's conversation chain to generate a response
    response = llm.generate(question, context=context)
    return response

def process_chat_with_langchain(question, context):
    # Initialize the LLM (OpenAI GPT model) with your API key
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    # Use Langchain's conversation chain to generate a response
    response = llm.generate(question, context=context)
    return response

def get_database():
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client["MosiAi"]  # The name of the database
    logger.info(f"Database connected: {db.name}")
    return db

def store_chat_history(db, chat_history_entry):
    db.chat_history_collection.insert_one(chat_history_entry)    
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    business_name: str = Form(...),
    description: str = Form(...),
    agent_name: str = Form(...),
):


    uploaded_docs = []
    for file in files:
    	

        # Save file to disk temporarily

        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            # Determine file type and use appropriate loader
            if file.filename.endswith(".pdf"):
                loader = PyPDFLoader(temp_file_path)
            elif file.filename.endswith(".txt"):
                loader = TextLoader(temp_file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            documents = loader.load()
            unique_id = str(uuid4())

            # Add additional metadata
            doc_metadata = {
                "business_name": business_name.strip(),
                "agent_name": agent_name.strip(),
                "filename": file.filename.strip(),
                "description": description.strip(),
                "unique_id": unique_id.strip(),
            }

            
           

            # Log metadata after converting ObjectId to string to verify its structure
            logger.info(f"Document metadata after insert: {doc_metadata}")
          
          

            # Try to index documents in Vectara
            try:
            	 # Initialize Vectara client
                vectara_client = Vectara(vectara_customer_id=os.getenv("VECTARA_CUSTOMER_ID"),vectara_corpus_id=os.getenv("VECTARA_CORPUS_ID"),vectara_api_key=os.getenv("VECTARA_API_KEY"))
                vectara_client.from_documents(documents, embedding=FakeEmbeddings(size=768), doc_metadata={"business_name": business_name, "filename": file.filename, "unique_id": unique_id})
                # Insert document metadata into MongoDB and immediately convert ObjectId to string
                insert_result = collection.insert_one(doc_metadata)
                uploaded_docs.append(file.filename)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Vectara indexing failed: {e}")

        except HTTPException as e:
            # Pass the HTTPException as is
            raise e
        except Exception as e:
            # For other exceptions, return a generic error message
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)
    
    if uploaded_docs:
        return JSONResponse(status_code=200, content={"message": "Documents uploaded successfully", "documents": uploaded_docs, "unique_id": unique_id })
    else:
        raise HTTPException(status_code=400, detail="No documents uploaded")



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=422)        



@app.post("/chat/{unique_doc_id}")
async def get_document(unique_doc_id: str, Mquery: QueryModel, db: MongoClient = Depends(get_database)):
    collection = db["Chatbots"]  # Replace with your collection name
    logger.info(f"Looking for document with unique_id: {unique_doc_id}")
    document_metadata = collection.find_one({"unique_id": unique_doc_id})  # Use the correct key for your MongoDB document
    if not document_metadata:
        logger.warning(f"Document with unique_id {unique_doc_id} not found")
        raise HTTPException(status_code=404, detail="Document not found")

    # Create a new response dictionary excluding the '_id' field
    response_data = {
        "business_name": document_metadata.get("business_name", ""),
        "agent_name": document_metadata.get("agent_name", ""),
        "collective_name": document_metadata.get("collective_name", ""),
        "filename": document_metadata.get("filename", ""),
        "unique_id": document_metadata.get("unique_id", ""),  # Assuming 'unique_id' is the correct field in your document
        # Include any other fields you need
    }

    logger.info(f"Performing similarity search with query: {Mquery.user_question}")

    logger.info(f"Document found: {response_data}")    
    # Initialize ConversationBufferMemory to track chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    logger.info("Initializing the LLM with your API key and desired settings") 
    # Initialize the LLM with your API key and desired settings
    llm = OpenAI(openai_api_key=OPENAI_API_KEY , temperature=0)

    query = Mquery.user_question

    logger.info(f"Performing similarity search with query: {query}")
     
    logger.info(f"doc.business_name = '{response_data['business_name']}'") 

    try:
    	found_docs = vectara_client.similarity_search(
        query,
        n_sentence_context=0,
        filter=f"doc.business_name = '{response_data['business_name']}'"
    )
    
    except Exception as e:
    	logger.error(f"Error during similarity search: {e}")
    	raise HTTPException(status_code=500, detail=str(e))

    

    # Check if we found any documents
    if not found_docs:
        raise HTTPException(status_code=404, detail="No similar documents found")

    # Assuming the first document is the most relevant one
    relevant_doc = found_docs[0]

    logger.info("Initializing the vectorstore and retriever using Vectara and the document metadata") 
    # Initialize the vectorstore and retriever using Vectara and the document metadata
    vectorstore = Vectara.from_documents(
        documents=[relevant_doc],
        embedding=FakeEmbeddings(size=768)
    )


    retriever = vectorstore.as_retriever(lambda_val=0.025, k=5, filter=f"doc.business_name = '{document_metadata['business_name']}'")

    # Initialize ConversationalRetrievalChain with the retriever and memory
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    response = qa_chain({"question": query})

    # Return the response
    return {"response": response['answer']}

 
@app.get("/business/{unique_doc_id}")
async def get_business_name(unique_doc_id: str, db: MongoClient = Depends(get_database)):
    collection = db["Chatbots"]  # Replace with your collection name
    document = collection.find_one({"unique_id": unique_doc_id}, {"_id": 0, "business_name": 1})
    
    if document:
        return document
    else:
        raise HTTPException(status_code=404, detail="Business name not found")



@app.post("/customeranalysis")
async def upload_and_analyze(file: UploadFile = File(...), pdf_file: UploadFile = None, prompt: str = Form(...)):
    # Check CSV file type
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid CSV file type.")

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    # Process PDF file if present
    if pdf_file:
        if pdf_file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="Invalid PDF file type.")

        temp_pdf_path = f"temp_{pdf_file.filename}"
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)

        loader = PyPDFLoader(temp_pdf_path)
        document = loader.load()

        vectorstore = vectara_client.from_documents(document, embedding=FakeEmbeddings(size=768), doc_metadata={"purpose": "analysis"})
        retriever = vectorstore.as_retriever(lambda_val=0.025, k=5, filter="doc.purpose = 'analysis'")
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory={})
        result = qa_chain({"question": "what metrics and strategies for customer data and analysis can you get from this"})

        os.remove(temp_pdf_path)

    # Read and process the CSV file
    content = await file.read()
    data_stream = io.StringIO(content.decode('utf-8'))
    csv_reader = csv.DictReader(data_stream)
    csv_data = [row for row in csv_reader]

    # Build context for analysis
    analysis_context = f"Analyzing customer data:\n{csv_data}"
    if pdf_file:
        analysis_context += f"\nBased on this data and the PDF analysis, {prompt}.And you can also use this information if {result}"
    else:
        analysis_context += f"\nBased on this CSV data, {prompt}."

    # Set up the prompt template
    prompt_template = PromptTemplate(
        input_variables=["user_data"],
        template="You are a customer analysis expert who analyzes CSVs of customer data and writes reports and suggests improvements. Here is the data: {user_data}.",
    )

    # Process the response from OpenAI API
    chain = LLMChain(llm=llm, prompt=prompt_template)
    analysis_result = chain.run(analysis_context)

    return {"analysisResult": analysis_result}
