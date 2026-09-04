import os
import streamlit as st

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)



# 1. LOAD ENVIRONMENT VARIABLES


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env")
    st.stop()


# 2. PAGE CONFIGURATION


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)



# 3. TITLE


st.title("📚 PDF RAG Chatbot")

st.write(
    "Upload multiple PDF documents and ask questions "
    "about their contents."
)



# 4. PDF UPLOADER


uploaded_files = st.file_uploader(
    "📤 Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)



# 5. PROCESS DOCUMENTS


if uploaded_files:

    st.write(f"📄 {len(uploaded_files)} PDF(s) uploaded.")

    if st.button("⚙️ Process Documents"):

        documents = []

        with st.spinner("📖 Reading PDF documents..."):

            for uploaded_file in uploaded_files:

                reader = PdfReader(uploaded_file)

                for page_number, page in enumerate(
                    reader.pages,
                    start=1
                ):

                    text = page.extract_text()

                    if text:

                        documents.append({
                            "text": text,
                            "source": uploaded_file.name,
                            "page": page_number
                        })


        st.success(
            f"✅ Extracted text from {len(uploaded_files)} PDF(s)"
        )


        # 6. SPLIT DOCUMENTS
       

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        all_chunks = []
        metadatas = []


        for document in documents:

            chunks = text_splitter.split_text(
                document["text"]
            )

            for chunk in chunks:

                all_chunks.append(chunk)

                metadatas.append({
                    "source": document["source"],
                    "page": document["page"]
                })


        st.success(
            f"✅ Created {len(all_chunks)} text chunks"
        )


      
        # 7. CREATE EMBEDDINGS
    

        with st.spinner("🔢 Creating embeddings..."):

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            )


     
        # 8. CREATE FAISS DATABASE
      

        with st.spinner(
            "🗄️ Creating FAISS vector database..."
        ):

            vectorstore = FAISS.from_texts(
                all_chunks,
                embedding=embeddings,
                metadatas=metadatas
            )


        st.session_state.vectorstore = vectorstore

        st.success(
            "✅ Documents processed successfully!"
        )



# 9. QUESTION ANSWERING


if "vectorstore" in st.session_state:

    st.divider()

    st.subheader("💬 Ask a question")

    question = st.text_input(
        "Enter your question:"
    )


    if question:


        # 10. RETRIEVE RELEVANT DOCUMENTS
   

        with st.spinner("🔍 Searching documents..."):

            results = (
                st.session_state.vectorstore
                .similarity_search(question, k=3)
            )


    
        # 11. BUILD CONTEXT
   

        context_parts = []

        for result in results:

            source = result.metadata.get(
                "source",
                "Unknown"
            )

            page = result.metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{result.page_content}"
            )


        context = "\n\n".join(context_parts)


   
        # 12. CREATE GEMINI


        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key
        )


  
        # 13. CREATE PROMPT
  

        prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using ONLY the
information provided in the context.

Do not make up information.

If the answer is not available in the context,
say:

"I couldn't find the answer in the uploaded documents."

Context:
================================

{context}

================================

Question:
{question}

Answer:
"""


      
        # 14. GENERATE ANSWER
    

        with st.spinner("🤖 Generating answer..."):

            response = llm.invoke(prompt)


        answer = response.content

        # Handle Gemini structured response
        if isinstance(answer, list):

            answer = "".join(
                item.get("text", "")
                for item in answer
                if isinstance(item, dict)
                and item.get("type") == "text"
            )


     
        # 15. DISPLAY ANSWER
    

        st.subheader("💬 Answer")

        st.write(answer)


        # 16. DISPLAY SOURCES


        st.subheader("📚 Sources")

        shown_sources = set()

        for result in results:

            source = result.metadata.get(
                "source",
                "Unknown"
            )

            page = result.metadata.get(
                "page",
                "Unknown"
            )

            source_key = (source, page)

            if source_key not in shown_sources:

                st.write(
                    f"📄 **{source}** — Page {page}"
                )

                shown_sources.add(source_key)