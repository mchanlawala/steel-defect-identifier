import os
import numpy as np
import faiss

from database import SessionLocal, DefectImage, DefectCategory
from embedding import extract_embedding

BASE_PATH = "data/images"
VECTOR_DB_DIR = "vectordb"

FOLDER_MAPPING = {
    "slivers": "Slivers / Laminations",
    "seams": "Seams",
    "caster_folds": "Caster Folds / Longitudinal Cracks",
    "annealing_burn": "Annealing Burn",
    "rolled_in_scale": "Rolled in Scale",
    "silicon_streaks": "Silicon Streaks",
    "scabs": "Scabs"
}

dimension = 2048
index = faiss.IndexFlatL2(dimension)

image_db_ids = []

def register_images():
    session = SessionLocal()

    for folder_name in os.listdir(BASE_PATH):
        folder_path = os.path.join(BASE_PATH, folder_name)

        if not os.path.isdir(folder_path):
            continue

        db_defect_name = FOLDER_MAPPING.get(folder_name)

        if not db_defect_name:
            print(f"⚠ No mapping found for folder: {folder_name}")
            continue

        defect = session.query(DefectCategory).filter_by(name=db_defect_name).first()

        if not defect:
            print(f"⚠ Defect not found in DB: {db_defect_name}")
            continue

        for img_file in os.listdir(folder_path):

            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(folder_path, img_file)

            existing = session.query(DefectImage).filter_by(image_path=image_path).first()
            if existing:
                continue

            # 1️⃣ Insert into DB
            new_image = DefectImage(
                defect_id=defect.id,
                image_path=image_path
            )

            session.add(new_image)
            session.commit()   # commit to get ID

            print(f"Registered in DB: {image_path}")

            # 2️⃣ Generate embedding
            embedding = extract_embedding(image_path)
            embedding = np.array([embedding]).astype("float32")

            # 3️⃣ Add to FAISS
            index.add(embedding)

            # 4️⃣ Store DB image ID
            image_db_ids.append(new_image.id)

    # Save FAISS index
    faiss.write_index(index, f"{VECTOR_DB_DIR}/index.faiss")

    # Save DB image IDs
    np.save(f"{VECTOR_DB_DIR}/image_ids.npy", np.array(image_db_ids))

    session.close()
    print("Image registration + vector index completed.")


if __name__ == "__main__":
    register_images()