import numpy as np
import faiss
from database import SessionLocal, DefectImage
from embedding import extract_embedding

VECTOR_DB_DIR = "vectordb"

dimension = 2048
index = faiss.IndexFlatL2(dimension)

image_db_ids = []

session = SessionLocal()
images = session.query(DefectImage).all()

print("Total images in DB:", len(images))

for img in images:
    embedding = extract_embedding(img.image_path)
    embedding = np.array([embedding]).astype("float32")

    index.add(embedding)
    image_db_ids.append(img.id)

faiss.write_index(index, f"{VECTOR_DB_DIR}/index.faiss")
np.save(f"{VECTOR_DB_DIR}/image_ids.npy", np.array(image_db_ids))

session.close()

print("FAISS index rebuilt successfully.")