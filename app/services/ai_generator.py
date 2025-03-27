import google.generativeai as genai
import json
from typing import Dict, Any
from app.core.config import settings

class AIGenerator:
    def __init__(self):
        # Konfigurasi Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt using Gemini API"""
        analysis_prompt = self._create_analysis_prompt(prompt)
        
        try:
            response = self.model.generate_content(analysis_prompt)
            return self._parse_llm_response(response.text)
            
        except Exception as e:
            raise ValueError(f"API request failed: {str(e)}")

    def _create_analysis_prompt(self, user_prompt: str) -> str:
        return f"""
        Kamu adalah AI expert yang akan menganalisis kebutuhan AI dan memberikan konfigurasi yang sesuai.
        
        Analisis kebutuhan AI ini dan berikan konfigurasi detail dalam format JSON:
        {user_prompt}
        
        Berikan response dalam format JSON dengan struktur berikut:
        ```json
        {{
            "ai_type": "tipe AI (classification/regression/clustering/dll)",
            "preprocessing": [
                "list langkah preprocessing yang diperlukan"
            ],
            "algorithm": {{
                "name": "nama algoritma yang disarankan",
                "params": {{
                    "parameter1": "nilai",
                    "parameter2": "nilai"
                }}
            }},
            "evaluation_metrics": [
                "list metrik evaluasi"
            ],
            "validation_strategy": "strategi validasi yang disarankan"
        }}
        ```
        
        Berikan HANYA response dalam format JSON, tanpa penjelasan tambahan.
        """

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        try:
            # Extract JSON from response
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            return json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Failed to parse API response: {str(e)}") 