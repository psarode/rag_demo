from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
# Example context, thx ChatGPT
from llama_index.core import Document

# Load environment variables
load_dotenv()

# Set up tracing
langfuse_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([langfuse_callback_handler])


 
doc1 = Document(text="""
Maxwell "Max" Silverstein, a lauded movie director, screenwriter, and producer, was born on October 25, 1978, in Boston, Massachusetts. A film enthusiast from a young age, his journey began with home movies shot on a Super 8 camera. His passion led him to the University of Southern California (USC), majoring in Film Production. Eventually, he started his career as an assistant director at Paramount Pictures. Silverstein's directorial debut, “Doors Unseen,” a psychological thriller, earned him recognition at the Sundance Film Festival and marked the beginning of a successful directing career.
""")
doc2 = Document(text="""
Throughout his career, Silverstein has been celebrated for his diverse range of filmography and unique narrative technique. He masterfully blends suspense, human emotion, and subtle humor in his storylines. Among his notable works are "Fleeting Echoes," "Halcyon Dusk," and the Academy Award-winning sci-fi epic, "Event Horizon's Brink." His contribution to cinema revolves around examining human nature, the complexity of relationships, and probing reality and perception. Off-camera, he is a dedicated philanthropist living in Los Angeles with his wife and two children.
""")

# Example index construction + LLM query
from llama_index.core import VectorStoreIndex
 
index = VectorStoreIndex.from_documents([doc1,doc2])

# Query
#response = index.as_query_engine().query("What did he do growing up?")


# Create a retriever to fetch relevant documents
retriever = index.as_retriever(retrieval_mode='similarity', k=3)

# Define your query
query = "What did he do growing up?"

# Retrieve relevant documents
relevant_docs = retriever.retrieve(query)

print(f"Number of relevant documents: {len(relevant_docs)}")
print("\n" + "="*50 + "\n")

for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}:")
    print(f"Text sample: {doc.node.get_content()[:200]}...")  # Print first 200 characters
    print(f"Metadata: {doc.node.metadata}")
    print(f"Score: {doc.score}")
    print("\n" + "="*50 + "\n")

#langfuse_callback_handler.flush()

#print(response)