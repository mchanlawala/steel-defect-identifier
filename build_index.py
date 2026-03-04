import numpy as np
import faiss
from database import SessionLocal, DefectImage
from models.feature_extractor import extract_embedding

session = SessionLocal()
images = session.query(DefectImage).all()

embeddings = []
image_ids = []

for img in images:
    emb = extract_embedding(img.image_path)
    embeddings.append(emb)
    image_ids.append(img.id)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "vectordb/index.faiss")
np.save("vectordb/image_ids.npy", image_ids)

print("Vector DB built successfully.")