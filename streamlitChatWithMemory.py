from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


import streamlit as st

import os
from dotenv import load_dotenv

# Carga las variables desde el archivo .env
load_dotenv()

# Ahora puedes acceder a las variables de entorno
#LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
#LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#GROQ_API_KEY = os.getenv('GROQ_API_KEY')

LANGCHAIN_TRACING_V2 = st.secrets.LANGCHAIN_TRACING_V2
LANGCHAIN_API_KEY = st.secrets.LANGCHAIN_API_KEY

# Get an Groq API Key before continuing
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets.GROQ_API_KEY
else:
    st.info("NO se ha encontrado una Groq API Key")
    

st.set_page_config(page_title="Entrevistador tecnico", page_icon="📖")
st.title("📖 Entrevistador tecnico")

"""
Chatbot de Entrevistas Tecnicas
"""

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("Hola soy tu entrevistador hoy, cuentame un poco sobre tu experiencia con la tecnologia")

#view_messages = st.expander("View the message contents in session state")

# Set up the LangChain, passing in Message History

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un experto en tecnologia, y entrevistas tecnicas para ingenieros y programadores, tu mision es ir haciendo preguntas teoricas relevantes con la experiencia del entrevistado, despues de que te respondan ofrece feedback sobre su respuesta y vuelves a hacer otra pregunta."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

#chain = prompt | ChatGroq(api_key=groq_api_key)
#model = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
#model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=GROQ_API_KEY)
model = ChatGroq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY)

chain = prompt | model
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response.content)

export_as_pdf = st.button("Export Report")

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

if export_as_pdf:
    
    # Create a PDF instance
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    full_interview = ""

    for msg in msgs.messages:
        pdf.multi_cell(0, 10, msg.type+":")
        pdf.multi_cell(0, 10, msg.content, border=1)

        #full_interview =+ msg.type+":\n"+msg.content+"\n"
        
    
    # Save the PDF to the specified path
    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "tech-enterview")

    st.markdown(html, unsafe_allow_html=True)



