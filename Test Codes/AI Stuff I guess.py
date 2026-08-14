import chromadb
import chromadb.utils.embedding_functions as ef
db = chromadb.PersistentClient(path="../chroma_db")
memories = db.get_or_create_collection("my_facts")
memories.upsert(
    documents=[
        "Final Fantasy 7",
        "*Insert Text Here*",
        "The Binding of Isaac is the best roguelite",
    ],
    ids=["fact1","fact2","fact3"]
)
print("\nstored:", memories.count(), "my_facts")
question = "Best video game"
results = memories.query(query_texts=[question], n_results=1)
for doc, dist in zip(results["documents"][0],results["distances"][0]):
    print(f"   {dist:.3f}   {doc}")