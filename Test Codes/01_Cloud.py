import chromadb
import chromadb.utils.embedding_functions as ef
db = chromadb.PersistentClient(path="../chroma_db")
memories = db.get_or_create_collection("my_facts")
memories.upsert(
    documents=[
        "Final Fantasy 7",
        "I like pizza",
        "The Binding of Isaac",
    ],
    ids=["fact1","fact2","fact3"]
)
print("\nstored:", memories.count(), "my_facts")
retrieved_docs = memories.get()["documents"]
context = "\n".join(f"- {doc}" for doc in retrieved_docs)
question = str(input("Enter the question you'd like to ask the AI: "))
prompt = f"""You have access to the following memories about the user: {context}
Using these memories where relevant, answer the user's question.
Question: {question}"""
results = memories.query(query_texts=[question], n_results=1)
for doc, dist in zip(results["documents"][0],results["distances"][0]):
    print(f"   {dist:.3f}   {doc}")
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GITHUB_TOKEN"),
)
r = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
)
print(r.choices[0].message.content)