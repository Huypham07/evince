import time
import json
import logging
from typing import Any, Dict, Optional, Union

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from core.config import Config

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, model_name: str = Config.GEMINI_MODEL, api_key: str = Config.GOOGLE_API_KEY):
        if not api_key:
            raise ValueError("API Key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < Config.REQUEST_INTERVAL:
            sleep_time = Config.REQUEST_INTERVAL - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def generate_content(self, prompt: str, schema: Optional[Any] = None) -> Union[str, Dict]:
        retries = Config.MAX_RETRIES
        
        for attempt in range(retries):
            try:
                self._rate_limit()
                
                generation_config = genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json" # Enforcing JSON for this pipeline
                )

                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings
                )
                
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     logger.warning(f"Blocked: {response.prompt_feedback.block_reason}")
                     return {"error": "blocked", "reason": str(response.prompt_feedback.block_reason)}

                text = response.text.strip()
                
                if text.startswith("```json"):
                    text = text[7:-3].strip()
                elif text.startswith("```"):
                    text = text[3:-3].strip()

                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(f"JSON Decode failed: {text[:50]}...")
                    return {"raw_text": text, "error": "json_decode_error"}

            except Exception as e:
                is_rate_limit = "429" in str(e) or "Resource has been exhausted" in str(e)
                if is_rate_limit:
                    sleep_time = (2 ** attempt) * 5
                    logger.warning(f"Rate Limit 429. Sleeping {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Error attempt {attempt+1}: {e}")
                    time.sleep(2)
        
        return {"error": "max_retries_exceeded"}
