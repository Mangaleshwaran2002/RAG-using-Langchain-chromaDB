import os
import streamlit as st
from tempfile import TemporaryDirectory
from langchain_core.messages import HumanMessage, AIMessage
from rag_pipeline import RagPipeline

st.title("Chat with PDF 👻")
st.set_page_config(page_title="Simple RAG using langchain", page_icon="🤖")

temp_db = TemporaryDirectory(dir="./")

MODEL_NAME = "llama3.2:3b"


@st.cache_resource()
def get_pipeline():
    return RagPipeline(llm_model=MODEL_NAME, persist_directory=temp_db.name,Search_top_K=4)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("human"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("ai"):
            st.markdown(msg.content)

rag_pipeline = get_pipeline()
if prompt := st.chat_input(
    placeholder="ask any question",
    accept_file="multiple",
    file_type=[".pdf", ".csv", ".docx", ".docs", ".xlsx"],
):
    if len(prompt.files) > 0:
        with TemporaryDirectory(dir="./") as temp_pdf_folder:
            with st.spinner("Processing Files"):
                for file in prompt.files:
                    pdf_loc = os.path.join(temp_pdf_folder, file.name)
                    with open(pdf_loc, "wb") as f:
                        f.write(file.getbuffer())
                    rag_pipeline.store_docs(pdf_loc)
            st.success("files uploaded successfully")
    if prompt.text:
        with st.chat_message("human"):
            st.markdown(prompt.text)
        retriever = rag_pipeline.get_retriever()
        context = retriever.invoke(prompt.text)
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                answer = st.empty()
                full_response = ""
                rag_chain = rag_pipeline.create_rag_chain()
                for chunk in rag_chain.stream(
                    input={
                        "question": prompt.text,
                        "context": context,
                        "chat_history": st.session_state.messages,
                    }
                ):
                    if hasattr(chunk, "content"):
                        full_response += chunk.content
                    answer.markdown(full_response)
        st.session_state.messages.append(HumanMessage(prompt.text))
        st.session_state.messages.append(AIMessage(full_response))
        answer.markdown(full_response)
