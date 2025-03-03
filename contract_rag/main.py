# Following the code from
# https://medium.com/@danushidk507/rag-with-llama-using-ollama-a-deep-dive-into-retrieval-augmented-generation-c58b9a1cfcd3

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

parser = argparse.ArgumentParser(description="RAG with LLAMA")
parser.add_argument("--pdf-document", type=Path, help="Path to the PDF document", default=Path("resources/archeology-laws-protect-sites.pdf"))
args = parser.parse_args()

assert args.pdf_document.exists(), "PDF document does not exist"

# Load the document
loader = PyPDFLoader(args.pdf_document)
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
docs = text_splitter.split_documents(documents=documents)



# Load embedding model
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs
)

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Save and reload the vector store
vectorstore.save_local("faiss_index_")
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

# Create a retriever
retriever = persisted_vectorstore.as_retriever()



## Initialize the LLaMA model
llm = OllamaLLM(model="llama3.1")

# #Test with a sample prompt

#response = llm.invoke("Tell me a joke")
#print(response)


# Create RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Interactive query loop
while True:
    query = input("Type your query (or type 'Exit' to quit): \n")
    if query.lower() == "exit":
        break
    result = qa.run(query)
    print(result)