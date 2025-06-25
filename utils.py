from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling.document_converter import DocumentConverter
from decouple import config
from openai import OpenAI
from tokenizer import OpenAITokenizerWrapper
import pickle
import os
from pathlib import Path
from tqdm import tqdm  # Progress bar

FILE_PATH = r"H:\py4e\Doc Analysis\input\Module-3.pdf"
CACHE_DIR = "cache"
CACHE_PATH = f"{CACHE_DIR}/{Path(FILE_PATH).stem}_chunks_first5pages.pkl"  # Changed cache file name for clarity

# Load OpenAI API key from environment
OPENAI_API_KEY = config("OPENAI_API_KEY", default=None)
print(f"Using OPENAI_API_KEY: {OPENAI_API_KEY}")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize tokenizer and chunker settings
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 8191

# Load chunks from cache if available, else process fresh
if os.path.exists(CACHE_PATH):
    print("✅ Loaded chunks from cache.")
    with open(CACHE_PATH, "rb") as f:
        chunks = pickle.load(f)
else:
    print("⏳ Chunking PDF (first 5 pages)...")

    # Step 1: Convert document
    converter = DocumentConverter()
    result = converter.convert(FILE_PATH)

    # Step 2: Restrict document to first 5 pages only
    result.document.pages = result.document.pages[:5]

    # Step 3: Setup chunker
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )

    # Step 4: Chunk with progress bar
    chunk_iter = chunker.chunk(dl_doc=result.document)
    chunks = []
    for chunk in tqdm(chunk_iter, desc="📚 Chunking in progress", unit="chunk"):
        chunks.append(chunk)

    # Step 5: Save chunks cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Chunking complete and cached.")

print(f"✅ Total chunks: {len(chunks)}")


from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert(r"H:\py4e\Doc Analysis\input\Module-3.pdf")

print(f"✅ Pages extracted: {len(result.document.pages)}")
for i, page in enumerate(result.document.pages[:5]):
    print(f"\nPage {i+1} content:\n{page.content[:300]}")  # Print only first 300 chars
