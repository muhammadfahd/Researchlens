import os
import re
import pandas as pd
import fitz  # PyMuPDF
import gradio as gr

from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# Configuration
# ============================================================


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it as an environment variable."
    )

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# Embedding Model
# ============================================================

# Load once when the application starts.
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# ============================================================
# File Loaders
# ============================================================

def load_pdf(file_path):
    documents = []
    file_name = os.path.basename(file_path)

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf):
            text = page.get_text()

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "file_type": "pdf",
                            "page": page_number + 1
                        }
                    )
                )

    return documents


def load_txt(file_path):
    file_name = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_name,
                "file_type": "txt"
            }
        )
    ]


def load_csv(file_path):
    file_name = os.path.basename(file_path)

    df = pd.read_csv(file_path)

    documents = []

    for row_number, row in df.iterrows():

        row_text = "\n".join(
            f"{column}: {value}"
            for column, value in row.items()
            if pd.notna(value)
        )

        if row_text.strip():
            documents.append(
                Document(
                    page_content=row_text,
                    metadata={
                        "source": file_name,
                        "file_type": "csv",
                        "row": row_number + 1
                    }
                )
            )

    return documents


def load_files(file_paths):
    """
    Load multiple uploaded PDF, TXT, and CSV files.
    """

    documents = []

    if not file_paths:
        return documents

    for file_path in file_paths:

        extension = os.path.splitext(file_path)[1].lower()

        try:

            if extension == ".pdf":
                documents.extend(load_pdf(file_path))

            elif extension == ".txt":
                documents.extend(load_txt(file_path))

            elif extension == ".csv":
                documents.extend(load_csv(file_path))

        except Exception as error:
            print(f"Error processing {file_path}: {error}")

    return documents


# ============================================================
# Text Cleaning
# ============================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    # Normalize spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# Chunking
# ============================================================

def chunk_documents(documents):

    cleaned_documents = []

    for document in documents:

        cleaned = clean_text(document.page_content)

        if cleaned:
            cleaned_documents.append(
                Document(
                    page_content=cleaned,
                    metadata=document.metadata.copy()
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(cleaned_documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


# ============================================================
# Knowledge Base
# ============================================================

def create_knowledge_base(files):
    """
    Process uploaded files and create the FAISS vector database.
    """

    if not files:
        return None, "Please upload at least one file."

    documents = load_files(files)

    if not documents:
        return None, "No readable content was found in the uploaded files."

    chunks = chunk_documents(documents)

    if not chunks:
        return None, "No usable text chunks could be created."

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    file_names = [
        os.path.basename(file)
        for file in files
    ]

    status = (
        f"Knowledge base ready.\n\n"
        f"Files: {len(files)}\n"
        f"Document sections: {len(documents)}\n"
        f"Chunks: {len(chunks)}\n\n"
        f"Sources:\n"
        + "\n".join(f"• {name}" for name in file_names)
    )

    return vector_store, status


# ============================================================
# Citation Helpers
# ============================================================

def get_source_label(document):

    source = document.metadata.get(
        "source",
        "Unknown source"
    )

    file_type = document.metadata.get(
        "file_type"
    )

    if file_type == "pdf":

        page = document.metadata.get(
            "page",
            "?"
        )

        return f"{source}, p. {page}"

    if file_type == "csv":

        row = document.metadata.get(
            "row",
            "?"
        )

        return f"{source}, row {row}"

    return source


# ============================================================
# RAG
# ============================================================

def generate_grounded_answer(
    query,
    vector_store,
    k=4
):

    if vector_store is None:
        return (
            "Please upload and process your sources first.",
            []
        )

    if not query or not query.strip():
        return "Please enter a question.", []

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    if not results:
        return (
            "I couldn't find relevant information "
            "in the uploaded sources.",
            []
        )

    # --------------------------------------------------------
    # Build grounded context
    # --------------------------------------------------------

    context_parts = []

    for index, (document, score) in enumerate(
        results,
        start=1
    ):

        source_label = get_source_label(document)

        context_parts.append(
            f"""
[SOURCE {index}]
Citation: {source_label}

Content:
{document.page_content}
"""
        )

    context = "\n".join(context_parts)

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    system_prompt = """
You are ResearchLens, an AI research assistant.

Your task is to answer questions using ONLY the evidence
retrieved from the user's uploaded sources.

Rules:

1. Use only the supplied evidence.
2. Do not use outside knowledge to fill information gaps.
3. If the evidence is insufficient, say:
   "I couldn't find enough information in the uploaded sources
   to answer this."
4. Never invent facts, authors, datasets, results, page numbers,
   citations, or conclusions.
5. Answer the user's actual question instead of merely
   summarizing the retrieved text.
6. Support factual claims using citations such as [Source 1].
7. Use multiple sources when appropriate.
8. Keep the response clear, structured, and research-oriented.
"""

    user_prompt = f"""
QUESTION:

{query}


RETRIEVED EVIDENCE:

{context}


Answer the question using only the evidence above.
"""

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    return answer, results


# ============================================================
# Format Sources
# ============================================================

def format_sources(results):

    if not results:
        return "No sources retrieved."

    source_lines = []

    seen = set()

    for index, (document, score) in enumerate(
        results,
        start=1
    ):

        label = get_source_label(document)

        # Prevent duplicate source labels
        key = (
            document.metadata.get("source"),
            document.metadata.get("page"),
            document.metadata.get("row")
        )

        if key in seen:
            continue

        seen.add(key)

        source_lines.append(
            f"**[Source {index}]** {label}"
        )

    return "\n\n".join(source_lines)


# ============================================================
# Chat Function
# ============================================================

def ask_question(
    message,
    history,
    vector_store
):

    if vector_store is None:

        return (
            history,
            "",
            "Upload your sources and click **Process Sources** first."
        )

    if not message.strip():
        return history, "", ""

    answer, results = generate_grounded_answer(
        query=message,
        vector_store=vector_store,
        k=4
    )

    # Gradio messages format
    history = history or []

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    sources = format_sources(results)

    return history, "", sources


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="ResearchLens"
) as demo:

    # Holds FAISS database during session
    vector_store_state = gr.State(None)

    gr.Markdown(
        """
        # ResearchLens

        ### Chat with your research sources

        Upload your documents, build a temporary knowledge base,
        and ask grounded questions about their contents.

        **Supported:** PDF · TXT · CSV
        """
    )

    with gr.Row():

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        with gr.Column(scale=1):

            gr.Markdown("## Sources")

            files = gr.File(
                label="Upload Sources",
                file_types=[
                    ".pdf",
                    ".txt",
                    ".csv"
                ],
                file_count="multiple",
                type="filepath"
            )

            process_button = gr.Button(
                "Process Sources",
                variant="primary"
            )

            processing_status = gr.Markdown(
                "Upload files to begin."
            )

        # ----------------------------------------------------
        # Research Chat
        # ----------------------------------------------------

        with gr.Column(scale=2):

            gr.Markdown("## Research Chat")

            chatbot = gr.Chatbot(
                height=450,
                placeholder=(
                    "Your research conversation "
                    "will appear here."
                )
            )

            question = gr.Textbox(
                placeholder=(
                    "Ask a question about your sources..."
                ),
                label="Question"
            )

            ask_button = gr.Button(
                "Ask ResearchLens",
                variant="primary"
            )

            gr.Markdown("### Retrieved Sources")

            sources_output = gr.Markdown(
                "Sources used for the latest answer will appear here."
            )

    # ========================================================
    # Events
    # ========================================================

    process_button.click(
        fn=create_knowledge_base,
        inputs=[files],
        outputs=[
            vector_store_state,
            processing_status
        ]
    )

    ask_button.click(
        fn=ask_question,
        inputs=[
            question,
            chatbot,
            vector_store_state
        ],
        outputs=[
            chatbot,
            question,
            sources_output
        ]
    )

    # Enter key also submits question
    question.submit(
        fn=ask_question,
        inputs=[
            question,
            chatbot,
            vector_store_state
        ],
        outputs=[
            chatbot,
            question,
            sources_output
        ]
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    demo.launch()