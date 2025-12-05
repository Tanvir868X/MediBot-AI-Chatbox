from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.prompt import prompt_template
import os

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Set API keys as environment variables for LangChain/Google libraries
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# --- Initialization of Vector Store and Retriever ---
# Ensure these run globally when the app starts
embeddings = download_hugging_face_embeddings()
index_name = "medibot"

# Initialize Pinecone Vector Store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Define the retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- Initialize LLM and RAG Chain using LCEL (Modern Approach) ---

# 1. Define the System/Human prompt template for LCEL
# This uses the string imported from src.prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{input}"),
    ]
)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# 2. Create the chain that stuffs documents into the prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 3. Create the final retrieval chain (The modern replacement for RetrievalQA)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)

    # In the new LCEL approach, the input key is 'input', not 'query'
    response = rag_chain.invoke({"input": msg})

    # The result key in the output dictionary is 'answer', not 'result'
    print("Response:", response.get("answer", "No answer"))
    return str(response.get("answer", "Sorry, no answer was generated."))

if __name__ == '__main__':
    # Make sure you are running this in your new environment (medibot_new)
    app.run(host="0.0.0.0", port=8080, debug=True)
