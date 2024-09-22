from sentence_transformers import SentenceTransformer

# Load a pre-trained SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Example sentences
sentences = ["This is a sample sentence", "Another example sentence"]

# Generate embeddings
embeddings = model.encode(sentences)

# embeddings will be a list of vectors (one per sentence)
print(embeddings)
