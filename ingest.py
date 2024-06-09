from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceHubEmbeddings,HuggingFaceEmbeddings
from langchain.llms import CTransformers,LlamaCpp
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

DB_FAISS_PATH = "vectorestores/db_faiss"
#load the data 
loader = DirectoryLoader("data/",glob="*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()

#split the text into chunks 

text_Splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks = text_Splitter.split_documents(documents=documents)


#create the embeddings 

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device':'cpu'})
#create a vectore store 

vectore_store = FAISS.from_documents(text_chunks,embeddings)
vectore_store.save_local(DB_FAISS_PATH)