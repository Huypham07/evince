"""
EVINCE: AWS Bedrock Client

LLM Client for AWS Bedrock with support for multiple regions.
Uses boto3 library for Bedrock Runtime.

Supported Models:
- Claude 3.5 Sonnet (anthropic.claude-3-5-sonnet-20241022-v2:0)
- Claude 3 Haiku (anthropic.claude-3-haiku-20240307-v1:0)
- Llama 3.1 (meta.llama3-1-70b-instruct-v1:0)
- Amazon Nova (amazon.nova-pro-v1:0)

Usage:
    from core import BedrockClient
    
    client = BedrockClient(region="us-east-1")
    result = client.generate_content("Phân loại câu sau...")
"""

import os
import json
import time
import logging
from typing import Any, Dict, Optional, Union, List

from .config import Config

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config as BotoConfig
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. Run: pip install boto3")


# Default models for different providers
# Note: Newer models require Cross-Region Inference Profile IDs (prefixed with region like "us." or "apac.")
BEDROCK_MODELS = {
    # Claude models (Anthropic)
    # Older models - use direct model ID
    "claude-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3.5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    
    # Newer models - use Cross-Region Inference Profile ID
    "claude-3.7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-3.7-sonnet-apac": "apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-3.7-sonnet-eu": "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
    
    # Llama models (Meta)
    "llama3-70b": "meta.llama3-1-70b-instruct-v1:0",
    "llama3.1-70b": "meta.llama3-1-70b-instruct-v1:0",
    # Llama 3.3 requires inference profile
    "llama3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    
    # Amazon Nova models
    "nova-pro": "amazon.nova-pro-v1:0",
    "nova-lite": "amazon.nova-lite-v1:0",
    "nova-micro": "amazon.nova-micro-v1:0",
}


class BedrockClient:
    """
    LLM Client for AWS Bedrock.
    Supports multiple regions for latency testing.
    """
    
    def __init__(
        self,
        region: str = None,
        model_id: str = None,
        profile_name: str = None,
        max_tokens: int = 4096,
        request_interval: float = 0.5
    ):
        """
        Initialize Bedrock client.
        
        Args:
            region: AWS region (e.g., 'us-east-1', 'ap-southeast-1')
            model_id: Bedrock model ID or alias from BEDROCK_MODELS
            profile_name: AWS profile name (optional, uses default if not specified)
            max_tokens: Maximum output tokens
            request_interval: Minimum seconds between requests
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required. Install with: pip install boto3")
        
        # Get config from environment or use defaults
        self.region = region or os.getenv("AWS_BEDROCK_REGION", "us-east-1")
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")
        self.max_tokens = max_tokens
        self.request_interval = request_interval
        self.last_request_time = 0
        
        # Resolve model ID
        model_alias = model_id or os.getenv("BEDROCK_MODEL", "claude-haiku")
        self.model_id = BEDROCK_MODELS.get(model_alias, model_alias)
        
        # Create boto3 session and client
        self._init_client()
        
        logger.info(f"BedrockClient initialized: region={self.region}, model={self.model_id}")
    
    def _init_client(self):
        """Initialize boto3 Bedrock Runtime client."""
        # Configure retry and timeout
        boto_config = BotoConfig(
            region_name=self.region,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            read_timeout=120,
            connect_timeout=10
        )
        
        # Get AWS credentials from environment (can be set in .env)
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.getenv("AWS_SESSION_TOKEN")  # Optional, for temporary credentials
        
        # Create session with explicit credentials if provided
        if self.profile_name:
            # Use AWS profile
            session = boto3.Session(profile_name=self.profile_name)
            self.client = session.client('bedrock-runtime', config=boto_config)
        elif aws_access_key and aws_secret_key:
            # Use credentials from environment/.env
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                aws_session_token=aws_session_token,
                region_name=self.region
            )
            self.client = session.client('bedrock-runtime', config=boto_config)
            logger.info("Using AWS credentials from environment")
        else:
            # Fall back to default credential chain (IAM role, ~/.aws/credentials, etc.)
            self.client = boto3.client('bedrock-runtime', config=boto_config)
    
    def switch_region(self, region: str):
        """Switch to a different AWS region."""
        self.region = region
        self._init_client()
        logger.info(f"Switched to region: {region}")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _build_claude_payload(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0
    ) -> Dict:
        """Build payload for Claude models."""
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        return payload
    
    def _build_llama_payload(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0
    ) -> Dict:
        """Build payload for Llama models."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        return {
            "prompt": full_prompt,
            "max_gen_len": self.max_tokens,
            "temperature": temperature
        }
    
    def _build_nova_payload(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0
    ) -> Dict:
        """Build payload for Amazon Nova models."""
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        
        payload = {
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": temperature
            }
        }
        
        if system_prompt:
            payload["system"] = [{"text": system_prompt}]
        
        return payload
    
    def _parse_response(self, response_body: Dict) -> str:
        """Parse response based on model type."""
        try:
            # Claude response
            if "content" in response_body:
                return response_body["content"][0]["text"]
            
            # Llama response
            if "generation" in response_body:
                return response_body["generation"]
            
            # Nova response
            if "output" in response_body:
                return response_body["output"]["message"]["content"][0]["text"]
            
            # Fallback
            return str(response_body)
            
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return str(response_body)
    
    def generate_content(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0
    ) -> Union[str, Dict]:
        """
        Generate content using Bedrock model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 = deterministic)
            
        Returns:
            Parsed JSON dict if response is JSON, otherwise raw text
        """
        self._rate_limit()
        
        # Build payload based on model type
        if "claude" in self.model_id.lower():
            payload = self._build_claude_payload(prompt, system_prompt, temperature)
        elif "llama" in self.model_id.lower():
            payload = self._build_llama_payload(prompt, system_prompt, temperature)
        elif "nova" in self.model_id.lower():
            payload = self._build_nova_payload(prompt, system_prompt, temperature)
        else:
            # Default to Claude format
            payload = self._build_claude_payload(prompt, system_prompt, temperature)
        
        try:
            # Call Bedrock
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)
            )
            
            # Parse response
            response_body = json.loads(response["body"].read())
            text = self._parse_response(response_body)
            
            # Clean markdown if present
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
            
            # Try to parse as JSON
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw_text": text}
                
        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            return {"error": str(e)}
    
    def generate_content_with_timing(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0
    ) -> Dict:
        """
        Generate content and return timing information.
        
        Returns:
            Dict with 'result', 'latency_ms', 'region', 'model'
        """
        start_time = time.perf_counter()
        result = self.generate_content(prompt, system_prompt, temperature)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        return {
            "result": result,
            "latency_ms": latency_ms,
            "region": self.region,
            "model": self.model_id
        }
