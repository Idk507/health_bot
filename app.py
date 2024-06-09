import streamlit as st
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers,LlamaCpp
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_chat import message
import speech_recognition as sr


DB_FAISS_PATH = "vectorestores/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH,embeddings)

st.set_page_config(page_title="HealthCare Chatbot",layout="wide")

st.title("HealthCare Chatbot")
#create llm
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",
                    config={'max_new_tokens':128,'temperature':0.01})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=db.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)


def conversation_chat(query):
    result = chain({'question':query,"chat_history":st.session_state['history']})
    st.session_state['history'].append(query,result['answer'])
    return result['answer']

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I am here to solve queries!!"]
    
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey"]


def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form",clear_on_submit=True):
            user_input = st.text_input("Question:",placeholder="Ask about you mental health",key="input")
            audio_input = st.audio_recorder(label="Or record your question")

            submit_button = st.form_submit_button(label="Submit")

            if submit_button and (user_input or audio_input):
                if audio_input:
                    r = sr.Recognizer()
                    with sr.AudioFile(audio_input.to_wav().getvalue()) as source:
                        audio_data = r.record(source)
                    try:
                        text = r.recognize_google(audio_data)
                        output = conversation_chat(text)
                    except sr.UnknownValueError:
                        st.warning("Sorry, I could not understand your speech.")
                        return
                    except sr.RequestError as e:
                        st.error("Could not request results from Google Speech Recognition service; {0}".format(e))
                        return
                else:
                    output = conversation_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
                st.session_state['history'] = st.session_state['past'] + st.session_state['generated']
            
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
                
initialize_session_state()
#chat history
display_chat_history()

