from rag_pipeline import RagPipeline
from tempfile import TemporaryDirectory
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

data_folder = "./data"

messages = []
with TemporaryDirectory(dir="./") as temp_db:
    rag_pipeline = RagPipeline(
        llm_model="llama3.2:3b", persist_directory=temp_db, Search_top_K=4
    )
    data_path = Path(data_folder)
    for file_loc in data_path.rglob("*"):
        if file_loc.is_file():
            print("file: ", file_loc)
            rag_pipeline.store_docs(file_loc)
    if __name__ == "__main__":
        rag_chain = rag_pipeline.create_rag_chain()
        while True:
            user_q = input("\nEnter your query (press q to exit): ").strip()
            if user_q.lower() == "q":
                break

            # Retrieve relevant documents
            retriever = rag_pipeline.get_retriever()
            context = retriever.invoke(user_q)

            # Stream the answer
            print(f"\nQ: {user_q}\nA: ", end="", flush=True)
            full_response = ""
            for chunk in rag_chain.stream(
                input={"question": user_q, "context": context, "chat_history": messages}
            ):
                # Some backends return a dict‑like object; adjust if needed
                content = getattr(chunk, "content", None)
                if content:
                    full_response += content
                    print(content, end="", flush=True)  # print only the new piece

            # Finish the line nicely
            print()  # newline after the streamed answer

            # Update chat history for the next turn
            messages.append(HumanMessage(user_q))
            messages.append(AIMessage(full_response))
