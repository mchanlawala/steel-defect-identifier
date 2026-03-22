import numpy as np
import faiss
from database import SessionLocal, DefectImage, DefectCategory
from embedding import extract_embedding
import logging
import os

logger = logging.getLogger(__name__)

# Helper function to convert numpy types to Python native types
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# Check if FAISS index exists
FAISS_INDEX_PATH = "vectordb/index.faiss"
IMAGE_IDS_PATH = "vectordb/image_ids.npy"

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(IMAGE_IDS_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    image_ids = np.load(IMAGE_IDS_PATH, allow_pickle=True)
    logger.info("FAISS index loaded successfully")
else:
    logger.warning("FAISS index not found. Run register_images.py or rebuild_index.py first")
    index = None
    image_ids = None

# Initialize LLM analyzer (lazy load to avoid import issues)
llm_analyzer = None

def get_llm_analyzer():
    global llm_analyzer
    if llm_analyzer is None:
        try:
            from llm_defect_analyzer import LLMDefectAnalyzer
            llm_analyzer = LLMDefectAnalyzer(model_name="llava:7b", use_vision=True)
        except ImportError as e:
            logger.error(f"Could not import LLM analyzer: {e}")
            llm_analyzer = None
    return llm_analyzer

def search_defect(image_path, use_llm=True, threshold_distance=100.0):
    """
    Search for similar defects with optional LLM enhancement
    """
    if index is None:
        return {
            "error": "FAISS index not initialized. Please run register_images.py first",
            "similar_defects": []
        }

    session = SessionLocal()

    try:
        # Extract embedding for query image
        query_vector = extract_embedding(image_path)
        query_vector = np.array([query_vector]).astype("float32")

        # Search in FAISS
        total_vectors = index.ntotal
        k = min(total_vectors, 10)
        D, I = index.search(query_vector, k)

        # Collect similar defects - convert numpy types immediately
        similar_defects = []
        for distance, idx in zip(D[0], I[0]):
            if idx == -1:  # No match
                continue

            db_image_id = image_ids[idx]
            # Convert to Python int to avoid numpy type issues
            db_image_id = int(db_image_id) if isinstance(db_image_id, (np.integer, np.int64, np.int32)) else db_image_id

            image_record = session.query(DefectImage).filter_by(id=db_image_id).first()

            if image_record:
                defect = session.query(DefectCategory).filter_by(id=image_record.defect_id).first()
                if defect:
                    # Convert distance to Python float
                    distance_value = float(distance)
                    similarity_score = 1 / (1 + distance_value)

                    similar_defects.append({
                        "defect_name": defect.name,
                        "defect_code": defect.code,
                        "description": defect.description,
                        "similar_image": image_record.image_path,
                        "distance": distance_value,
                        "similarity_score": float(similarity_score)  # Ensure it's float
                    })

        # Prepare base results
        results = {
            "similar_defects": similar_defects,
            "top_match": similar_defects[0] if similar_defects else None,
            "num_matches": len(similar_defects)
        }

        # Enhance with LLM if requested
        if use_llm and similar_defects:
            analyzer = get_llm_analyzer()
            if analyzer and analyzer.available:
                try:
                    llm_analysis = analyzer.analyze_defect_with_context(
                        image_path,
                        similar_defects,
                        threshold_distance
                    )
                    results["llm_analysis"] = convert_to_serializable(llm_analysis)
                    results["recommendation"] = llm_analysis.get("recommendation", "Manual inspection")
                    results["confidence"] = llm_analysis.get("confidence", 0)
                except Exception as e:
                    logger.error(f"LLM enhancement failed: {e}")
                    results["llm_error"] = str(e)
            else:
                results["llm_enhancement"] = "LLM not available"

        # Final conversion to ensure everything is serializable
        return convert_to_serializable(results)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"error": str(e), "similar_defects": []}

    finally:
        session.close()

def search_defect_enhanced(image_path):
    """Enhanced search with LLM analysis"""
    return search_defect(image_path, use_llm=True)