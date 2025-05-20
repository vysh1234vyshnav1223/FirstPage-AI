from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import sys
import traceback

# Load environment variables
load_dotenv()

# Check if the API key is set
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY is not set in your environment variables.")
    print("Please set it and try again.")
    sys.exit()

# 1. Debugging the Document Loading
print("Debugging: Loading the document...")
try:
    loader = TextLoader("LandingPageGuide.md", encoding="utf-8")
    documents = loader.load()
    print("✅ Document loaded successfully.")
except Exception as e:
    print(f"Error loading the document: {e}")
    print("Check the file path and encoding of 'LandingPageGuide.md'.")
    sys.exit()

# 2. Debugging the Splitting
print("Debugging: Splitting the document...")
try:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"✅ Document split into {len(chunks)} chunks.")
except Exception as e:
    print(f"Error splitting the document: {e}")
    sys.exit()

# 3. Debugging the Embeddings
print("Debugging: Creating embeddings...")
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("✅ Embeddings model loaded.")
except Exception as e:
    print(f"Error creating embeddings: {e}")
    print("Check your model name and API key.")
    sys.exit()

# 4. Debugging the Vectorstore
print("Debugging: Creating vectorstore...")
try:
    print("Before FAISS.from_documents:")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Type of chunks: {type(chunks)}")
    print(f"  Type of embeddings: {type(embeddings)}")

    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    print("✅ Vectorstore created and used.")
except Exception as e:
    print(f"Error creating vectorstore: {e}")
    traceback.print_exc()
    print("Check your FAISS installation and dependencies, and the data being passed to FAISS.")
    sys.exit()

# 5. Debugging the QA Chain and LLM
print("Debugging: Setting up QA chain and testing LLM...")
try:
    retriever = vectordb.as_retriever()
    print("✅ Retriever created.")
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)
    print("✅ LLM model loaded.")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever) # Fix is here
    # Test query
    query = "What are the best CTA design principles for landing pages?"
    response = qa_chain.run(query)
    print("\n✅ Sample response:\n", response)

except Exception as e:
    print(f"Error during QA chain or LLM interaction: {e}")
    print("Check your API key, model name, and network connection.  Also, check the structure of your data and prompt.")
    sys.exit()
