from dotenv import load_dotenv
import os
import hashlib
from tqdm import tqdm
from src.helper import (
    load_all_documents,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
    crawl_website,
)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# === Environment ===
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# === Load and preprocess documents ===
extracted_data = load_all_documents('data/')
print(f"📄 Loaded {len(extracted_data)} documents from disk.")

# === Crawl websites ===
urls = [
    "https://www.adntel.com.bd/",
]
url_docs = []
for url in urls:
    url_docs.extend(crawl_website(url, max_pages=20))
print(f"🌐 Crawled {len(url_docs)} documents from URLs.")

# === Combine and split ===
extracted_data.extend(url_docs)
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)
text_chunks = [doc for doc in text_chunks if len(doc.page_content.encode('utf-8')) < 2000]
print(f"✅ Prepared {len(text_chunks)} total chunks.")

# === Embeddings ===
embeddings = download_hugging_face_embeddings()

# === Pinecone setup ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medibot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# === Generate IDs ===
all_ids = [hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest() for doc in text_chunks]

# === Check existing ===
existing_ids = set()
BATCH_LOOKUP = 100
for i in tqdm(range(0, len(all_ids), BATCH_LOOKUP), desc="🔍 Checking existing IDs"):
    batch_ids = all_ids[i:i + BATCH_LOOKUP]
    response = index.fetch(ids=batch_ids)
    existing_ids.update(response.vectors.keys())

# === Filter new chunks ===
new_docs = []
for doc, doc_id in zip(text_chunks, all_ids):
    if doc_id not in existing_ids:
        doc.metadata["id"] = doc_id
        new_docs.append(doc)

print(f"🆕 {len(new_docs)} new chunks to upload.")

# === Upload to Pinecone ===
batch_size = 50
for i in range(0, len(new_docs), batch_size):
    batch = new_docs[i:i + batch_size]
    PineconeVectorStore.from_documents(
        documents=batch,
        index_name=index_name,
        embedding=embeddings
    )
    print(f"📦 Uploaded batch {i // batch_size + 1} / {(len(new_docs) + batch_size - 1) // batch_size}")

print("✅ Indexing complete.")
