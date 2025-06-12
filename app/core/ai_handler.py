"""
AI Handler Module for AI Gate Application

This module manages communication with OpenRouter API and Hugging Face Inference API,
implements model fallback logic, handles rate limiting and error recovery,
and provides response validation and formatting.
"""

import os
import json
import logging
import asyncio
import aiohttp
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import re

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Enumeration for model availability status."""
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ModelInfo:
    """Data class for storing model information and statistics."""
    name: str
    provider: str = "openrouter"  # Added provider field
    is_free: bool = True
    requests_made: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    average_response_time: float = 0.0
    last_used: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    status: ModelStatus = ModelStatus.AVAILABLE
    rate_limit_reset: Optional[datetime] = None


@dataclass
class RequestMetrics:
    """Data class for tracking request metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    total_response_time: float = 0.0
    requests_by_model: Dict[str, int] = field(default_factory=dict)
    requests_by_provider: Dict[str, int] = field(default_factory=dict)  # Added provider tracking
    errors_by_type: Dict[str, int] = field(default_factory=dict)


# Abstract Provider Interface
class AIProvider(ABC):
    """Abstract base class for AI model providers."""
    
    @abstractmethod
    async def make_request(self, payload: Dict[str, Any], model: ModelInfo) -> Tuple[bool, str, Dict[str, Any]]:
        """Make API request to the provider."""
        pass
    
    @abstractmethod
    def build_payload(self, user_message: str, system_prompt: str, model_name: str) -> Dict[str, Any]:
        """Build request payload for the provider."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (has required credentials)."""
        pass


class OpenRouterProvider(AIProvider):
    """OpenRouter API provider implementation."""
    
    def __init__(self, config: Dict[str, Any], session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = config.get('base_url', 'https://openrouter.ai/api/v1/chat/completions')
        self.max_tokens = config.get('max_tokens', 2048)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p')
        self.frequency_penalty = config.get('frequency_penalty')
        self.presence_penalty = config.get('presence_penalty')
        self.rate_limit_window = config.get('rate_limit_window', 60)
    
    def is_available(self) -> bool:
        """Check if OpenRouter is available."""
        return bool(self.api_key)
    
    def build_payload(self, user_message: str, system_prompt: str, model_name: str) -> Dict[str, Any]:
        """Build request payload for OpenRouter API."""
        messages = []
        
        if system_prompt.strip():
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
        
        # Add optional parameters
        if self.top_p is not None:
            payload['top_p'] = self.top_p
        if self.frequency_penalty is not None:
            payload['frequency_penalty'] = self.frequency_penalty
        if self.presence_penalty is not None:
            payload['presence_penalty'] = self.presence_penalty
        
        return payload
    
    async def make_request(self, payload: Dict[str, Any], model: ModelInfo) -> Tuple[bool, str, Dict[str, Any]]:
        """Make API request to OpenRouter."""
        start_time = time.time()
        metadata = {
            'model_used': model.name,
            'provider': 'openrouter',
            'request_time': start_time,
            'tokens_used': 0
        }
        
        try:
            logger.debug(f"Making OpenRouter API request to {model.name}")
            
            # Set Authorization header for this request
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            async with self.session.post(self.base_url, json=payload, headers=headers) as response:
                response_time = time.time() - start_time
                metadata['response_time'] = response_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    if 'choices' in data and data['choices']:
                        response_text = data['choices'][0]['message']['content']
                        
                        if 'usage' in data:
                            metadata['tokens_used'] = data['usage'].get('total_tokens', 0)
                        
                        logger.debug(f"Successful OpenRouter response from {model.name} in {response_time:.2f}s")
                        return True, response_text, metadata
                    else:
                        return False, "Invalid response format", metadata
                
                elif response.status == 429:
                    error_msg = "Rate limit exceeded"
                    logger.warning(f"OpenRouter rate limit hit for {model.name}")
                    return False, error_msg, metadata
                
                else:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get('error', {}).get('message', f'HTTP {response.status}')
                    except:
                        error_msg = f'HTTP {response.status}'
                    
                    logger.warning(f"OpenRouter API error for {model.name}: {error_msg}")
                    return False, f"API Error: {error_msg}", metadata
        
        except asyncio.TimeoutError:
            logger.warning(f"OpenRouter request timeout for {model.name}")
            return False, "Request timeout", metadata
        except aiohttp.ClientError as e:
            logger.warning(f"OpenRouter connection error for {model.name}: {str(e)}")
            return False, f"Connection error: {str(e)}", metadata
        except Exception as e:
            logger.error(f"Unexpected OpenRouter error for {model.name}: {str(e)}")
            return False, f"Unexpected error: {str(e)}", metadata


class HuggingFaceProvider(AIProvider):
    """Hugging Face Inference API provider implementation."""
    
    def __init__(self, config: Dict[str, Any], session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.api_key = os.getenv(config.get('api_key_env_var', 'HF_API_TOKEN'))
        self.base_url_template = "https://api-inference.huggingface.co/models/{}"
        self.timeout = config.get('hf_timeout', 45)
        self.max_new_tokens = config.get('hf_max_new_tokens', 1024)
        self.temperature = config.get('hf_temperature', config.get('temperature', 0.7))
    
    def is_available(self) -> bool:
        """Check if Hugging Face is available."""
        return bool(self.api_key)
    
    def build_payload(self, user_message: str, system_prompt: str, model_name: str) -> Dict[str, Any]:
        """Build request payload for Hugging Face API."""
        # Combine system prompt and user message for Hugging Face
        if system_prompt.strip():
            prompt = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        else:
            prompt = f"User: {user_message}\n\nAssistant:"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "return_full_text": False,
                "do_sample": True
            }
        }
        
        return payload
    
    async def make_request(self, payload: Dict[str, Any], model: ModelInfo) -> Tuple[bool, str, Dict[str, Any]]:
        """Make API request to Hugging Face."""
        start_time = time.time()
        metadata = {
            'model_used': model.name,
            'provider': 'huggingface',
            'request_time': start_time,
            'tokens_used': 0
        }
        
        try:
            logger.debug(f"Making Hugging Face API request to {model.name}")
            
            # Build URL for specific model
            url = self.base_url_template.format(model.name)
            
            # Set Authorization header for this request
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            # Use custom timeout for Hugging Face requests
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with self.session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                response_time = time.time() - start_time
                metadata['response_time'] = response_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Handle different response formats from Hugging Face
                    if isinstance(data, list) and len(data) > 0:
                        # Standard text generation response
                        if 'generated_text' in data[0]:
                            response_text = data[0]['generated_text'].strip()
                        elif 'text' in data[0]:
                            response_text = data[0]['text'].strip()
                        else:
                            response_text = str(data[0]).strip()
                    elif isinstance(data, dict):
                        # Alternative response format
                        response_text = data.get('generated_text', data.get('text', str(data))).strip()
                    else:
                        response_text = str(data).strip()
                    
                    # Estimate tokens used (rough approximation)
                    metadata['tokens_used'] = len(response_text.split()) + len(payload['inputs'].split())
                    
                    logger.debug(f"Successful Hugging Face response from {model.name} in {response_time:.2f}s")
                    return True, response_text, metadata
                
                elif response.status == 429:
                    error_msg = "Rate limit exceeded"
                    logger.warning(f"Hugging Face rate limit hit for {model.name}")
                    return False, error_msg, metadata
                
                elif response.status == 503:
                    error_msg = "Model is currently loading, please try again later"
                    logger.warning(f"Hugging Face model loading for {model.name}")
                    return False, error_msg, metadata
                
                else:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get('error', f'HTTP {response.status}')
                    except:
                        error_msg = f'HTTP {response.status}'
                    
                    logger.warning(f"Hugging Face API error for {model.name}: {error_msg}")
                    return False, f"API Error: {error_msg}", metadata
        
        except asyncio.TimeoutError:
            logger.warning(f"Hugging Face request timeout for {model.name}")
            return False, "Request timeout", metadata
        except aiohttp.ClientError as e:
            logger.warning(f"Hugging Face connection error for {model.name}: {str(e)}")
            return False, f"Connection error: {str(e)}", metadata
        except Exception as e:
            logger.error(f"Unexpected Hugging Face error for {model.name}: {str(e)}")
            return False, f"Unexpected error: {str(e)}", metadata


class AIHandler:
    """
    Handles AI model communication with multiple providers (OpenRouter and Hugging Face).
    
    Features:
    - Multi-provider support with automatic fallback
    - Model fallback logic with automatic retry
    - Rate limiting and error recovery
    - Response validation and formatting
    - Comprehensive logging and metrics
    - Intelligent caching integration
    """
    
    def __init__(self, config: Dict[str, Any], cache_manager=None):
        """
        Initialize the AI Handler.
        
        Args:
            config: Configuration dictionary for AI models and settings (ai_models section from YAML)
            cache_manager: Optional cache manager instance
        """
        self.config = config
        self.cache_manager = cache_manager
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize models from configuration
        self.models: List[ModelInfo] = []
        self._initialize_models()
        
        # Request tracking and metrics
        self.metrics = RequestMetrics()
        
        # Configuration parameters
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.max_consecutive_failures = self.config.get('max_consecutive_failures', 5)
        self.rate_limit_window = self.config.get('rate_limit_window', 60)
        
        # Validation settings
        self.min_response_length = self.config.get('min_response_length', 10)
        self.max_response_length = self.config.get('max_response_length', 8000)
        self.preserve_markdown = self.config.get('preserve_markdown', False)
        
        # Fallback responses
        self.fallback_responses = self.config.get('fallback_responses', [
            "I apologize, but I'm currently experiencing technical difficulties. Please try again in a few moments.",
            "I'm temporarily unable to process your request due to system issues. Please contact support if this persists.",
            "There seems to be a temporary service disruption. Please try your question again shortly."
        ])
        
        # Initialize HTTP session and providers
        self._initialize_session()
        self._initialize_providers()
        
        logger.info(f"AI Handler initialized with {len(self.models)} models across {len(self.providers)} providers")
    
    def _initialize_models(self):
        """Initialize model list from configuration."""
        models_list = []
        
        # OpenRouter models
        primary_model = self.config.get('primary_model')
        fallback_models = self.config.get('fallback_models', [])
        
        if primary_model:
            models_list.append(('openrouter', primary_model))
            logger.info(f"Primary OpenRouter model configured: {primary_model}")
            
            if isinstance(fallback_models, list) and fallback_models:
                for model in fallback_models:
                    models_list.append(('openrouter', model))
                logger.info(f"Fallback OpenRouter models configured: {fallback_models}")
        
        # Hugging Face models (if enabled)
        if self.config.get('direct_fallback_enabled', False):
            hf_config = self.config.get('huggingface_direct_provider', {})
            hf_primary = hf_config.get('primary_model_hf')
            hf_fallbacks = hf_config.get('fallback_models_hf', [])
            
            if hf_primary:
                models_list.append(('huggingface', hf_primary))
                logger.info(f"Primary Hugging Face model configured: {hf_primary}")
                
                if isinstance(hf_fallbacks, list) and hf_fallbacks:
                    for model in hf_fallbacks:
                        models_list.append(('huggingface', model))
                    logger.info(f"Fallback Hugging Face models configured: {hf_fallbacks}")
        
        # Use embedded defaults if no valid configuration found
        if not models_list:
            models_list = [
                ('openrouter', "deepseek/deepseek-prover-v2:free"),
                ('openrouter', "mistralai/mistral-small-3.1-24b-instruct:free"),
                ('openrouter', "microsoft/phi-4-reasoning:free"),
                ('huggingface', "google/gemma-7b-it")  # Default HF fallback
            ]
            logger.warning(f"No models configured in YAML, using embedded defaults")
        
        # Create ModelInfo objects
        for provider, model_name in models_list:
            if not isinstance(model_name, str) or not model_name.strip():
                logger.warning(f"Skipping invalid model name: {model_name}")
                continue
                
            model_name = model_name.strip()
            self.models.append(ModelInfo(
                name=model_name,
                provider=provider,
                is_free=':free' in model_name.lower() or provider == 'huggingface'
            ))
        
        # Ensure we have at least one model
        if not self.models:
            logger.error("No valid models initialized, creating emergency fallback")
            self.models.append(ModelInfo(
                name="google/gemma-7b-it",
                provider="huggingface",
                is_free=True
            ))
        
        logger.info(f"Initialized {len(self.models)} models: {[(m.provider, m.name) for m in self.models]}")
    
    def _initialize_session(self):
        """Initialize aiohttp session with proper configuration."""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Gate-Application/1.0.0',
        }
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        logger.debug("HTTP session initialized")
    
    def _initialize_providers(self):
        """Initialize AI providers."""
        self.providers = {}
        
        # Initialize OpenRouter provider
        self.providers['openrouter'] = OpenRouterProvider(self.config, self.session)
        
        # Initialize Hugging Face provider
        hf_config = self.config.get('huggingface_direct_provider', {})
        self.providers['huggingface'] = HuggingFaceProvider(hf_config, self.session)
        
        # Log provider availability
        for name, provider in self.providers.items():
            available = provider.is_available()
            logger.info(f"Provider {name} {'available' if available else 'not available'}")
    
    async def _ensure_session(self):
        """Ensure HTTP session is available."""
        if self.session is None or self.session.closed:
            self._initialize_session()
            self._initialize_providers()
    
    def _get_available_model(self) -> Optional[ModelInfo]:
        """Get the next available model for requests."""
        current_time = datetime.now()
        
        for model in self.models:
            # Check if provider is available
            provider = self.providers.get(model.provider)
            if not provider or not provider.is_available():
                logger.debug(f"Provider {model.provider} not available for model {model.name}")
                continue
            
            # Check if model is rate limited
            if (model.rate_limit_reset and 
                current_time < model.rate_limit_reset):
                logger.debug(f"Model {model.name} is rate limited until {model.rate_limit_reset}")
                continue
            
            # Check consecutive failures
            if model.consecutive_failures >= self.max_consecutive_failures:
                # Reset after some time
                if (model.last_used and 
                    current_time - model.last_used > timedelta(minutes=10)):
                    model.consecutive_failures = 0
                    model.status = ModelStatus.AVAILABLE
                    logger.info(f"Reset failure count for model {model.name} after cooldown period")
                else:
                    logger.debug(f"Model {model.name} has too many consecutive failures: {model.consecutive_failures}")
                    continue
            
            if model.status == ModelStatus.AVAILABLE:
                return model
        
        # If no models are available, reset the first available model as emergency fallback
        for model in self.models:
            provider = self.providers.get(model.provider)
            if provider and provider.is_available():
                logger.warning(f"No models available, resetting {model.name} as emergency fallback")
                model.consecutive_failures = 0
                model.status = ModelStatus.AVAILABLE
                return model
        
        return None
    
    def _update_model_metrics(self, model: ModelInfo, success: bool, 
                            response_time: float, error: Optional[str] = None):
        """Update model performance metrics."""
        model.requests_made += 1
        model.last_used = datetime.now()
        
        if success:
            model.successful_requests += 1
            model.consecutive_failures = 0
            model.status = ModelStatus.AVAILABLE
            
            # Update average response time
            if model.average_response_time == 0:
                model.average_response_time = response_time
            else:
                model.average_response_time = (
                    (model.average_response_time * (model.successful_requests - 1) + response_time) 
                    / model.successful_requests
                )
        else:
            model.failed_requests += 1
            model.consecutive_failures += 1
            model.last_error = error
            
            # Update model status based on error type
            if error and "rate limit" in error.lower():
                model.status = ModelStatus.RATE_LIMITED
                model.rate_limit_reset = datetime.now() + timedelta(seconds=self.rate_limit_window)
                logger.warning(f"Model {model.name} rate limited, reset at {model.rate_limit_reset}")
            elif model.consecutive_failures >= self.max_consecutive_failures:
                model.status = ModelStatus.FAILED
                logger.warning(f"Model {model.name} marked as failed after {model.consecutive_failures} consecutive failures")
    
    def _validate_response(self, response_text: str) -> Tuple[bool, Optional[str]]:
        """Validate AI response for quality and completeness."""
        if not response_text or not response_text.strip():
            return False, "Empty response received"
        
        response_text = response_text.strip()
        
        # Check length constraints
        if len(response_text) < self.min_response_length:
            return False, f"Response too short (minimum {self.min_response_length} characters)"
        
        if len(response_text) > self.max_response_length:
            return False, f"Response too long (maximum {self.max_response_length} characters)"
        
        # Check for common error patterns
        error_patterns = [
            r"i apologize.*unable to",
            r"i cannot.*provide",
            r"error.*occurred",
            r"something went wrong",
            r"try again later"
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, response_text.lower()):
                return False, "Response indicates an error or inability to help"
        
        # Check for meaningful content (not just repetition)
        words = response_text.lower().split()
        if len(words) > 10 and len(set(words)) < len(words) * 0.3:  # Too much repetition
            return False, "Response appears to be repetitive"
        
        return True, None
    
    def _format_response(self, response_text: str) -> str:
        """Format and clean the AI response."""
        # Remove excessive whitespace
        formatted = re.sub(r'\n\s*\n\s*\n', '\n\n', response_text.strip())
        
        # Ensure proper sentence structure
        formatted = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', formatted)
        
        # Remove any markdown artifacts if not needed
        if not self.preserve_markdown:
            formatted = re.sub(r'\*\*(.*?)\*\*', r'\1', formatted)  # Bold
            formatted = re.sub(r'\*(.*?)\*', r'\1', formatted)      # Italic
        
        return formatted
    
    async def generate_response(self, user_message: str, system_prompt: str, 
                              context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate AI response with multi-provider fallback logic and error handling.
        
        Args:
            user_message: User's input message
            system_prompt: System prompt with institutional context
            context: Optional additional context from website research
            
        Returns:
            str: Generated AI response
        """
        start_time = time.time()
        self.metrics.total_requests += 1
        
        # Check cache first if available
        if self.cache_manager:
            cache_key = self._generate_cache_key(user_message, system_prompt)
            cached_response = await self.cache_manager.get_cached_response(cache_key)
            if cached_response:
                self.metrics.cache_hits += 1
                logger.info("Returning cached AI response")
                return cached_response.get('response', '')
        
        # Enhance system prompt with context if available
        enhanced_prompt = self._enhance_prompt_with_context(system_prompt, context)
        
        # Try each model in order
        last_error = "No models available"
        for attempt in range(self.max_retries):
            model = self._get_available_model()
            if not model:
                logger.error("No available models for request")
                break
            
            logger.info(f"Attempting request with model: {model.name} ({model.provider}) (attempt {attempt + 1})")
            
            # Get provider for this model
            provider = self.providers.get(model.provider)
            if not provider:
                logger.error(f"Provider {model.provider} not found for model {model.name}")
                continue
            
            # Build request payload
            payload = provider.build_payload(user_message, enhanced_prompt, model.name)
            
            # Make API request
            success, response_text, metadata = await provider.make_request(payload, model)
            
            # Update metrics
            request_time = metadata.get('response_time', 0)
            self._update_model_metrics(model, success, request_time, 
                                     None if success else response_text)
            
            # Update global metrics
            self.metrics.requests_by_model[model.name] = self.metrics.requests_by_model.get(model.name, 0) + 1
            self.metrics.requests_by_provider[model.provider] = self.metrics.requests_by_provider.get(model.provider, 0) + 1
            
            if success:
                # Validate response
                is_valid, validation_error = self._validate_response(response_text)
                if is_valid:
                    # Format response
                    formatted_response = self._format_response(response_text)
                    
                    # Update metrics
                    self.metrics.successful_requests += 1
                    self.metrics.total_response_time += time.time() - start_time
                    
                    # Cache response if cache manager is available
                    if self.cache_manager:
                        await self.cache_manager.cache_response(cache_key, {
                            'response': formatted_response,
                            'model_used': model.name,
                            'provider_used': model.provider,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    logger.info(f"Successfully generated response using {model.name} ({model.provider})")
                    return formatted_response
                else:
                    last_error = f"Response validation failed: {validation_error}"
                    logger.warning(f"Response validation failed for {model.name}: {validation_error}")
            else:
                last_error = response_text
                logger.warning(f"Request failed for {model.name} ({model.provider}): {response_text}")
                
                # Add to error metrics
                error_type = "rate_limit" if "rate limit" in response_text.lower() else "api_error"
                self.metrics.errors_by_type[error_type] = self.metrics.errors_by_type.get(error_type, 0) + 1
            
            # Wait before retrying
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # All attempts failed
        self.metrics.failed_requests += 1
        logger.error(f"All AI generation attempts failed. Last error: {last_error}")
        
        # Return fallback response
        return self._get_fallback_response(last_error)
    
    def _generate_cache_key(self, user_message: str, system_prompt: str) -> str:
        """Generate cache key for request."""
        combined = f"{user_message}|{system_prompt}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _enhance_prompt_with_context(self, system_prompt: str, 
                                   context: Optional[List[Dict[str, Any]]]) -> str:
        """Enhance system prompt with additional context."""
        if not context:
            return system_prompt
        
        context_text = "\n\nAdditional Context:\n"
        for item in context[:3]:  # Limit context to avoid token limits
            if 'content' in item:
                context_text += f"- {item['content'][:200]}...\n"
        
        return system_prompt + context_text
    
    def _get_fallback_response(self, error: str) -> str:
        """Generate fallback response when all models fail."""
        if "rate limit" in error.lower():
            return "I'm currently experiencing high demand. Please try again in a few minutes."
        elif "timeout" in error.lower():
            return "The request is taking longer than expected. Please try again with a shorter question."
        else:
            if self.fallback_responses:
                return self.fallback_responses[0]
            else:
                return "I apologize, but I'm currently experiencing technical difficulties. Please try again in a few moments."
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about AI handler performance.
        
        Returns:
            Dict: Statistics and performance metrics
        """
        total_requests = max(self.metrics.total_requests, 1)  # Avoid division by zero
        
        stats = {
            'total_requests': self.metrics.total_requests,
            'successful_requests': self.metrics.successful_requests,
            'failed_requests': self.metrics.failed_requests,
            'cache_hits': self.metrics.cache_hits,
            'success_rate': (self.metrics.successful_requests / total_requests) * 100,
            'cache_hit_rate': (self.metrics.cache_hits / total_requests) * 100,
            'average_response_time': (
                self.metrics.total_response_time / max(self.metrics.successful_requests, 1)
            ),
            'requests_by_model': dict(self.metrics.requests_by_model),
            'requests_by_provider': dict(self.metrics.requests_by_provider),
            'errors_by_type': dict(self.metrics.errors_by_type),
            'models': []
        }
        
        # Add per-model statistics
        for model in self.models:
            model_stats = {
                'name': model.name,
                'provider': model.provider,
                'status': model.status.value,
                'requests_made': model.requests_made,
                'successful_requests': model.successful_requests,
                'failed_requests': model.failed_requests,
                'success_rate': (
                    (model.successful_requests / max(model.requests_made, 1)) * 100
                ),
                'average_response_time': model.average_response_time,
                'total_tokens': model.total_tokens,
                'consecutive_failures': model.consecutive_failures,
                'last_used': model.last_used.isoformat() if model.last_used else None,
                'last_error': model.last_error,
                'is_free': model.is_free
            }
            stats['models'].append(model_stats)
        
        return stats
    
    def is_healthy(self) -> bool:
        """
        Check if the AI handler is healthy and operational.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Check if we have at least one provider with API key
            has_available_provider = any(provider.is_available() for provider in self.providers.values())
            if not has_available_provider:
                logger.debug("Health check failed: No available providers")
                return False
            
            # Check if session is available
            if not self.session or self.session.closed:
                logger.debug("Health check failed: No active session")
                return False
            
            # Check if at least one model is available
            available_models = [m for m in self.models if m.status == ModelStatus.AVAILABLE]
            if not available_models:
                logger.debug("Health check failed: No available models")
                return False
            
            # Check recent success rate
            if self.metrics.total_requests > 10:
                recent_success_rate = (self.metrics.successful_requests / self.metrics.total_requests) * 100
                if recent_success_rate < 50:  # Less than 50% success rate
                    logger.debug(f"Health check failed: Low success rate ({recent_success_rate:.1f}%)")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                logger.info("HTTP session closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def reset_model_failures(self, model_name: Optional[str] = None):
        """
        Reset failure count for models.
        
        Args:
            model_name: Specific model to reset, or None to reset all
        """
        reset_count = 0
        for model in self.models:
            if model_name is None or model.name == model_name:
                model.consecutive_failures = 0
                model.status = ModelStatus.AVAILABLE
                model.last_error = None
                model.rate_limit_reset = None
                reset_count += 1
                logger.info(f"Reset failures for model: {model.name}")
        
        if reset_count == 0 and model_name:
            logger.warning(f"Model '{model_name}' not found for failure reset")
        else:
            logger.info(f"Reset failures for {reset_count} model(s)")
    
    def get_model_priority_info(self) -> List[Dict[str, Any]]:
        """
        Get information about model priority and configuration.
        
        Returns:
            List of dictionaries with model priority information
        """
        priority_info = []
        for i, model in enumerate(self.models):
            info = {
                'priority': i + 1,
                'name': model.name,
                'provider': model.provider,
                'is_primary': i == 0,
                'is_free': model.is_free,
                'status': model.status.value,
                'consecutive_failures': model.consecutive_failures
            }
            priority_info.append(info)
        
        return priority_info