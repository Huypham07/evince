import os
import sys
import time
import json
import base64
import logging
import requests
from typing import Any, Dict, Optional, Union

from .config import Config

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QwenClient:
    """
    LLM Client for Qwen via HTTP API with Basic Auth.
    Uses OpenAI-compatible /chat/completions endpoint.
    """
    def __init__(
        self
    ):
        self.base_url = Config.QWEN_BASE_URL
        self.auth_username = Config.QWEN_AUTH_USERNAME
        self.auth_password = Config.QWEN_AUTH_PASSWORD
        self.model = Config.QWEN_MODEL
        self.max_tokens = Config.QWEN_MAX_TOKENS
        self.request_interval = Config.QWEN_REQUEST_INTERVAL
        self.last_request_time = 0
        
        # Prepare auth header
        auth_string = f"{self.auth_username}:{self.auth_password}"
        self.auth_header = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    
    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        
    def generate_content(self, prompt: str, system_prompt: str = None, temperature: float = 0.0) -> Union[str, Dict]:
        url = self.base_url.rstrip('/')
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {self.auth_header}"
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        # Clean markdown if present
                        text = content.strip()
                        if text.startswith("```json"):
                            text = text[7:-3].strip()
                        elif text.startswith("```"):
                            text = text[3:-3].strip()
                        
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            logger.warning(f"JSON parse failed: {text[:100]}...")
                            return {"raw_text": text, "error": "json_decode_error"}
                    else:
                        return {"error": "no_choices"}
                        
                elif response.status_code == 429:
                    sleep_time = (2 ** attempt) * 5
                    logger.warning(f"429 Rate Limit. Sleeping {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"API Error {response.status_code}: {response.text}")
                    return {"error": f"api_{response.status_code}", "msg": response.text}
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt+1}")
                time.sleep(2)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                time.sleep(2)
        
        return {"error": "max_retries_exceeded"}
