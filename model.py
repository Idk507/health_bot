from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceHubEmbeddings,HuggingFaceEmbeddings
from langchain.llms import CTransformers,LlamaCpp
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate

DB_FAISS_PATH = "vectorestores/db_faiss"

#prompt template 
custom_prompt_template = """ 
    You are AI medical assisstant and you have the following piece of information to answer the user's questopm and with the informations answer the user question ,don't hallucinate with the answers.
    Context :{}
    Question :{question}

    Only return the helpful answers 
"""

prompt = PromptTemplate(template=custom_prompt_template,input_variables=['context','question'])
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH,embeddings)


llm  = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",model_type="llama",
                     config = {'max_new_tokens':128,'temperature':0.0
                                 ,'top_k':0,'top_p':0.9,'max_length':256,'max_tokens':256})


qa_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=db.as_retriever(search_kwargs={'k': 2}),
                                        return_source_documents=True,chain_type_kwargs={'prompt':prompt})

result = qa_chain({'query':query})

print(result['result'])