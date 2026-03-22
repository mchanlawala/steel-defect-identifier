from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import shutil
import os
import logging
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Steel Defect Identifier with LLM Enhancement",
    description="API for steel defect detection with optional LLM enhancement",
    version="1.0.0"
)

executor = ThreadPoolExecutor(max_workers=4)

@app.get("/")
def root():
    return {
        "message": "Steel Defect AI Running with LLM Enhancement",
        "models_available": ["resnet50", "llava:7b"],
        "features": ["similarity_search", "llm_analysis", "defect_recommendations"]
    }

@app.post("/predict")
async def predict(
        file: UploadFile = File(..., description="Image file to analyze"),
        use_llm: bool = Query(True, description="Enable LLM enhancement"),
        threshold: float = Query(100.0, description="Similarity threshold")
):
    """Analyze a single steel defect image"""
    temp_path = None
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            temp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        logger.info(f"Processing file: {file.filename}, use_llm={use_llm}")

        # Import search functions here to avoid circular imports
        from search import search_defect_enhanced, search_defect

        # Get prediction
        if use_llm:
            results = search_defect_enhanced(temp_path)
        else:
            results = search_defect(temp_path, use_llm=False)

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "enhanced": use_llm,
            **results
        })

    except Exception as e:
        logger.error(f"Error processing {file.filename}: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "filename": file.filename,
                "error": str(e)
            }
        )

    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/batch_predict")
async def batch_predict(
        files: List[UploadFile] = File(..., description="Multiple image files to analyze"),
        use_llm: bool = Query(True, description="Enable LLM enhancement")
):
    """
    Batch analyze multiple steel defect images

    **How to use in Swagger UI:**
    1. Click "Try it out"
    2. For each file, click "Browse" button next to each string field
    3. Select actual image files from your computer
    4. Click "Execute"

    **Note:** The UI shows "array<string>" but you need to click the file picker buttons
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    logger.info(f"Processing batch of {len(files)} files, use_llm={use_llm}")

    # Process files
    results = []

    for file in files:
        temp_path = None
        try:
            # Create temporary file for each upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                temp_path = tmp.name
                content = await file.read()
                tmp.write(content)

            logger.info(f"Processing: {file.filename}")

            from search import search_defect_enhanced, search_defect

            # Run in thread pool
            loop = asyncio.get_event_loop()
            if use_llm:
                result = await loop.run_in_executor(
                    executor,
                    search_defect_enhanced,
                    temp_path
                )
            else:
                result = await loop.run_in_executor(
                    executor,
                    search_defect,
                    temp_path,
                    False
                )

            results.append({
                "filename": file.filename,
                "status": "success",
                **result
            })

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    # Summary
    successful = sum(1 for r in results if r.get("status") == "success")

    return JSONResponse(content={
        "batch_summary": {
            "total": len(results),
            "successful": successful,
            "failed": len(results) - successful
        },
        "results": results
    })