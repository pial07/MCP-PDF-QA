import os
import pickle
import pdfplumber
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from decouple import config

import pytesseract  # add this

# Explicitly tell pytesseract where tesseract.exe is
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pdf_folder = r"H:\py4e\Doc Analysis\input"
save_path = r"H:\py4e\Doc Analysis\output\all_chunks_structured.pkl"
persist_directory = r"H:\py4e\Doc Analysis\output\vectorstore"
OPENAI_API_KEY = config("OPENAI_API_KEY", default=None, cast=str)

# --------- Table Extraction Helper ----------
def extract_tables_as_markdown(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table or len(table) < 2:
                    continue
                # Convert to Markdown table
                header = table[0]
                divider = ["---"] * len(header)
                rows = table[1:]
                formatted = [header, divider] + rows
                table_md = "\n".join(["| " + " | ".join(cell or "" for cell in row) + " |" for row in formatted])
                tables.append(table_md)
    return "\n\n".join(tables)

# --------- Load and combine structured content ----------
def load_pdf_with_structure(pdf_path):
    elements = partition_pdf(filename=pdf_path, strategy="hi_res")
    extracted_text = "\n\n".join([el.text for el in elements if el.text])
    tables_text = extract_tables_as_markdown(pdf_path)
    full_text = extracted_text + "\n\n📊 Extracted Tables:\n\n" + tables_text
    return full_text, elements

# --------- Chunking Process ----------
def chunk_pdfs_with_structure():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = []

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing: {filename}")

            full_text, elements = load_pdf_with_structure(pdf_path)
            chunks = text_splitter.split_text(full_text)

            for chunk in chunks:
                all_chunks.append({
                    "source_file": filename,
                    "num_elements": len(elements),
                    "chunk_text": chunk
                })

    print(f"Total chunks from all PDFs: {len(all_chunks)}")

    with open(save_path, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Chunks saved to {save_path}")

    return all_chunks

# --------- Load existing or create new chunks ----------
def load_chunks():
    with open(save_path, "rb") as f:
        all_chunks = pickle.load(f)
    print(f"Loaded {len(all_chunks)} chunks from pickle.")
    return all_chunks

if os.path.exists(save_path):
    chunks = load_chunks()
else:
    chunks = chunk_pdfs_with_structure()

print("Example chunk info:")
print("Source file:", chunks[0]["source_file"])
print("Number of structured elements in PDF:", chunks[0]["num_elements"])
print("Chunk text snippet:", chunks[0]["chunk_text"][:500])

# --------- Vectorstore builder ----------
def create_or_update_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY) # type: ignore

    texts = [chunk["chunk_text"] for chunk in chunks]
    metadatas = [{"source_file": chunk["source_file"]} for chunk in chunks]

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vectorstore from {persist_directory}")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("Adding new documents to existing vectorstore...")
        vectordb.add_texts(texts=texts, metadatas=metadatas)
        vectordb.persist()
    else:
        print("Creating new vectorstore...")
        vectordb = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=persist_directory)
        vectordb.persist()
        print(f"Vectorstore saved to {persist_directory}")

    return vectordb

vectordb = create_or_update_vectorstore(chunks)
