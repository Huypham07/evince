"""
EVINCE: Google Gemini Client

LLM Client for Google Gemini API using the new google.genai package.
Supports Gemini 2.0 Flash, Gemini 1.5 Pro, and other models.

Usage:
    from core import GeminiClient
    
    client = GeminiClient()
    result = client.generate_content("Phân loại câu sau...")
"""

import os
import time
import json
import logging
from typing import Any, Dict, Optional, Union

from .config import Config

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try new package first, fall back to old package
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
    USING_NEW_API = True
    logger.info("Using new google.genai package")
except ImportError:
    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        GENAI_AVAILABLE = True
        USING_NEW_API = False
        logger.warning("Using deprecated google.generativeai. Consider: pip install google-genai")
    except ImportError:
        GENAI_AVAILABLE = False
        USING_NEW_API = False
        logger.warning("Google genai not installed. Run: pip install google-genai")
        genai = None


# Available Gemini models
GEMINI_MODELS = {
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite", 
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
}


class GeminiClient:
    """
    LLM Client for Google Gemini API.
    Supports both new google.genai and deprecated google.generativeai packages.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        request_interval: float = None
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key (uses GOOGLE_API_KEY env var if not specified)
            model: Model name or alias from GEMINI_MODELS
            request_interval: Minimum seconds between requests
        """
        if not GENAI_AVAILABLE:
            raise ImportError("Google genai is required. Install with: pip install google-genai")
        
        # Get API key
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or Config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required. Set in .env or pass as argument.")
        
        # Resolve model name
        model_alias = model or os.getenv("GEMINI_MODEL") or Config.GEMINI_MODEL
        self.model_name = GEMINI_MODELS.get(model_alias, model_alias)
        
        self.request_interval = request_interval or Config.REQUEST_INTERVAL
        self.last_request_time = 0
        
        # Initialize client based on which package is available
        if USING_NEW_API:
            self.client = genai.Client(api_key=self.api_key)
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"GeminiClient initialized: model={self.model_name}, api={'new' if USING_NEW_API else 'legacy'}")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _generate_new_api(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0
    ) -> Union[str, Dict]:
        """Generate using new google.genai package."""
        try:
            # Build config
            config = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json"
            )
            
            # Add system instruction if provided
            if system_prompt:
                config.system_instruction = system_prompt
            
            # Generate
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            text = response.text.strip()
            return self._parse_response(text)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"error": str(e)}
    
    def _generate_legacy_api(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0
    ) -> Union[str, Dict]:
        """Generate using legacy google.generativeai package."""
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        try:
            # Combine prompts
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json"
            )
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return {"error": "blocked", "reason": str(response.prompt_feedback.block_reason)}
            
            text = response.text.strip()
            return self._parse_response(text)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"error": str(e)}
    
    def _parse_response(self, text: str) -> Union[str, Dict]:
        """Parse response text, handling markdown code blocks."""
        # Clean markdown if present
        if text.startswith("```json"):
            text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        elif text.startswith("```"):
            text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        
        # Try to parse as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_text": text}
    
    def generate_content(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0
    ) -> Union[str, Dict]:
        """
        Generate content using Gemini model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0.0 = deterministic)
            
        Returns:
            Parsed JSON dict if response is JSON, otherwise raw text dict
        """
        self._rate_limit()
        
        retries = Config.MAX_RETRIES
        
        for attempt in range(retries):
            try:
                if USING_NEW_API:
                    return self._generate_new_api(prompt, system_prompt, temperature)
                else:
                    return self._generate_legacy_api(prompt, system_prompt, temperature)
                    
            except Exception as e:
                is_rate_limit = "429" in str(e) or "Resource has been exhausted" in str(e)
                if is_rate_limit:
                    sleep_time = (2 ** attempt) * 5
                    logger.warning(f"Rate Limit. Sleeping {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Error attempt {attempt+1}: {e}")
                    time.sleep(2)
        
        return {"error": "max_retries_exceeded"}
    
    def generate_content_with_timing(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0
    ) -> Dict:
        """
        Generate content and return timing information.
        
        Returns:
            Dict with 'result', 'latency_ms', 'model'
        """
        start_time = time.perf_counter()
        result = self.generate_content(prompt, system_prompt, temperature)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        return {
            "result": result,
            "latency_ms": latency_ms,
            "model": self.model_name
        }
