import faiss
import numpy as np
from database import SessionLocal, DefectImage, DefectCategory
from embedding import extract_embedding

index = faiss.read_index("vectordb/index.faiss")
image_ids = np.load("vectordb/image_ids.npy", allow_pickle=True)


def search_defect(image_path):

    session = SessionLocal()

    query_vector = extract_embedding(image_path)
    query_vector = np.array([query_vector]).astype("float32")

    total_vectors = index.ntotal
    D, I = index.search(query_vector, total_vectors)

    results = []

    for distance, idx in zip(D[0], I[0]):

        db_image_id = image_ids[idx]

        image_record = session.query(DefectImage).filter_by(id=int(db_image_id)).first()

        defect = session.query(DefectCategory).filter_by(id=image_record.defect_id).first()

        results.append({
            "defect_name": defect.name,
            "similar_image": image_record.image_path,
            "distance": float(distance)
        })

    session.close()

    return results