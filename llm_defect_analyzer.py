import ollama
import json
import logging
from typing import List, Dict, Any
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMDefectAnalyzer:
    def __init__(self, model_name="llava:7b", use_vision=True):
        self.model_name = model_name
        self.use_vision = use_vision
        self.available = self._check_ollama_available()

    def _check_ollama_available(self):
        """Check if Ollama is available and model is loaded"""
        try:
            # Try to list models to check connection
            response = ollama.list()
            models = response.get('models', [])
            model_names = [m.get('name', '') for m in models]
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found. Available: {model_names}")
                return False
            logger.info(f"Ollama is available with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False

    def analyze_defect_with_context(self,
                                    query_image_path: str,
                                    similar_defects: List[Dict],
                                    threshold_distance: float = 100.0) -> Dict[str, Any]:
        """
        Analyze defect using both image and similarity search results
        """

        # If Ollama is not available, return fallback
        if not self.available:
            logger.warning("Ollama not available, using fallback analysis")
            return self._fallback_analysis(similar_defects)

        # Filter high-confidence similar defects and ensure proper types
        top_defects = []
        for d in similar_defects:
            distance = float(d.get('distance', 1000))  # Convert to float
            if distance < threshold_distance:
                top_defects.append(d)
        top_defects = top_defects[:3]

        # Prepare context from similar defects
        defect_context = ""
        for i, defect in enumerate(top_defects, 1):
            similarity = 1 / (1 + float(defect.get('distance', 1)))
            defect_context += f"""
            {i}. Defect Type: {defect.get('defect_name', 'Unknown')}
               Similarity Score: {similarity:.3f}
            """

        # Create prompt for LLM
        prompt = f"""
        You are a steel quality control expert analyzing surface defects.
        
        ### Similar Defects Found in Database:
        {defect_context if defect_context else "No highly similar defects found in database."}
        
        ### Task:
        Based on the similar defects found, provide analysis in JSON format with these fields:
        1. "defect_type": The most likely defect name
        2. "description": Brief description of this defect
        3. "severity": "Low", "Medium", "High", or "Critical"
        4. "root_cause": What likely caused this defect
        5. "recommendation": "Accept", "Repair", or "Reject"
        6. "confidence": A number between 0-100
        
        Return ONLY valid JSON, no other text.
        """

        try:
            # For LLaVA, we need to handle images specially
            if self.use_vision and os.path.exists(query_image_path):
                # Read image as bytes
                with open(query_image_path, 'rb') as f:
                    image_bytes = f.read()

                response = ollama.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [image_bytes]
                    }]
                )
            else:
                # Text-only analysis
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }]
                )

            # Parse response
            content = response['message']['content']

            # Try to extract JSON from response
            try:
                # Find JSON in response (in case model adds extra text)
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                    analysis = json.loads(json_str)
                else:
                    analysis = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, create structured response
                analysis = {
                    "defect_type": top_defects[0].get('defect_name', 'Unknown') if top_defects else "Unknown",
                    "description": content[:200],
                    "severity": "Medium",
                    "root_cause": "Unable to determine",
                    "recommendation": "Manual inspection",
                    "confidence": 50
                }

            # Convert all values to Python native types
            return {
                "defect_type": str(analysis.get("defect_type", "Unknown")),
                "description": str(analysis.get("description", "")),
                "severity": str(analysis.get("severity", "Medium")),
                "root_cause": str(analysis.get("root_cause", "Unable to determine")),
                "recommendation": str(analysis.get("recommendation", "Manual inspection")),
                "confidence": int(analysis.get("confidence", 50))
            }

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(similar_defects)

    def _fallback_analysis(self, similar_defects: List[Dict]) -> Dict[str, Any]:
        """Provide fallback analysis if LLM fails"""
        if similar_defects:
            top_defect = similar_defects[0].get('defect_name', 'Unknown')
            distance = float(similar_defects[0].get('distance', 1))
            similarity_score = 1 / (1 + distance)
            confidence = int(similarity_score * 100)

            return {
                "defect_type": str(top_defect),
                "description": f"Identified as {top_defect} based on visual similarity",
                "severity": "Medium",
                "root_cause": "Refer to similar defect patterns in database",
                "recommendation": "Manual inspection recommended",
                "confidence": min(confidence, 95)
            }
        else:
            return {
                "defect_type": "Unknown",
                "description": "No matching defects found in database",
                "severity": "Unknown",
                "root_cause": "Unable to determine",
                "recommendation": "Manual inspection required",
                "confidence": 0
            }