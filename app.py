import streamlit as st
import os
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI

# Sidebar contents
# with st.sidebar:
#     st.title('LLM Chat')
#     st.markdown('''
#     # About
#     This is an LLM
#     ''')
#     add_vertical_space(5)
#     st.write('Made with Love')

def main():
    st.header("Chat with PDF 💭")

    # Load environment variables from .env file (if available)
    load_dotenv()

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Display the PDF content when uploaded
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chucks = text_splitter.split_text(text=text)

        # Embeddings
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)

        else:
            embeddings = OpenAIEmbeddings()

            VectorStore = FAISS.from_texts(chucks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Ask user a question
        query = st.text_input("Ask a question from the PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            # This callback gives you the cost
            with get_openai_callback() as cb:
                # Run the question
                response = chain.run(input_documents=docs, question=query)
                st.write(cb)
            st.write(response)

if __name__ == '__main__':
    main()

    