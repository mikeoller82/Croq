"""
Modern Async-First Model Interface with Intelligent Routing
"""
import asyncio
import aiohttp
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator, Union
from dataclasses import dataclass
from enum import Enum

import anthropic
import openai
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings, ModelProvider, ModelConfig


logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant" 
    SYSTEM = "system"


@dataclass
class Message:
    role: MessageRole
    content: str
    thinking: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass 
class ModelResponse:
    content: str
    thinking: Optional[str] = None
    usage: Dict[str, int] = None
    model: str = None
    provider: ModelProvider = None
    latency: float = None
    cost: float = None


class BaseModelClient(ABC):
    """Abstract base for all model clients"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.provider = config.provider
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._total_cost = 0.0
        
    async def __aenter__(self):
        if not self._session:
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout
            )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None
    
    @abstractmethod
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate response from messages"""
        pass
    
    @abstractmethod
    async def stream_generate(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream response generation"""
        pass
    
    def calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost based on token usage"""
        if not usage or self.config.cost_per_token == 0:
            return 0.0
        
        input_tokens = usage.get('input_tokens', 0) + usage.get('prompt_tokens', 0)
        output_tokens = usage.get('output_tokens', 0) + usage.get('completion_tokens', 0)
        total_tokens = input_tokens + output_tokens
        
        cost = total_tokens * self.config.cost_per_token
        self._total_cost += cost
        return cost


class ClaudeClient(BaseModelClient):
    """Optimized Claude client with advanced features"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError))
    )
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate with Claude using thinking and tools"""
        start_time = time.time()
        
        # Format messages for Claude
        claude_messages = []
        system_message = ""
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                claude_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_message,
                messages=claude_messages,
                **kwargs
            )
            
            # Extract content and thinking
            content = ""
            thinking = ""
            
            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == "text":
                        content += block.text
                    elif block.type == "thinking":
                        thinking = block.thinking
                else:
                    content += str(block)
            
            latency = time.time() - start_time
            usage = {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            }
            cost = self.calculate_cost(usage)
            
            self._request_count += 1
            
            return ModelResponse(
                content=content.strip(),
                thinking=thinking,
                usage=usage,
                model=self.config.model_name,
                provider=self.provider,
                latency=latency,
                cost=cost
            )
            
        except Exception as e:
            logger.error(f"Claude generation error: {e}")
            raise
    
    async def stream_generate(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream Claude responses"""
        # Format messages for Claude
        claude_messages = []
        system_message = ""
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                claude_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        try:
            with self.client.messages.stream(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_message,
                messages=claude_messages,
                **kwargs
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            raise


class GPTClient(BaseModelClient):
    """Optimized GPT client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate with GPT"""
        start_time = time.time()
        
        # Format messages for GPT
        gpt_messages = []
        for msg in messages:
            gpt_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=gpt_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs
            )
            
            content = response.choices[0].message.content
            latency = time.time() - start_time
            
            usage = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            }
            cost = self.calculate_cost(usage)
            
            self._request_count += 1
            
            return ModelResponse(
                content=content.strip(),
                usage=usage,
                model=self.config.model_name,
                provider=self.provider,
                latency=latency,
                cost=cost
            )
            
        except Exception as e:
            logger.error(f"GPT generation error: {e}")
            raise
    
    async def stream_generate(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream GPT responses"""
        gpt_messages = []
        for msg in messages:
            gpt_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=gpt_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"GPT streaming error: {e}")
            raise


class GroqClient(BaseModelClient):
    """Optimized Groq client - fastest inference"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate with Groq - optimized for speed"""
        start_time = time.time()
        
        # Format messages for Groq
        groq_messages = []
        for msg in messages:
            groq_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.model_name,
                messages=groq_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs
            )
            
            content = response.choices[0].message.content
            latency = time.time() - start_time
            
            usage = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            }
            
            self._request_count += 1
            
            return ModelResponse(
                content=content.strip(),
                usage=usage,
                model=self.config.model_name,
                provider=self.provider,
                latency=latency,
                cost=0.0  # Groq is free tier
            )
            
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            raise
    
    async def stream_generate(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream Groq responses"""
        # Groq streaming implementation
        groq_messages = []
        for msg in messages:
            groq_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        try:
            # Note: Adjust if Groq client supports async streaming
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=groq_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True,
                **kwargs
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            raise


class GeminiClient(BaseModelClient):
    """Optimized Gemini client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(config.model_name)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate with Gemini"""
        start_time = time.time()
        
        # Format messages for Gemini
        chat_history = []
        prompt = ""
        
        for i, msg in enumerate(messages):
            if msg.role == MessageRole.SYSTEM:
                prompt = f"System: {msg.content}\n\n"
            elif msg.role == MessageRole.USER:
                if i == len(messages) - 1:
                    prompt += msg.content  # Last user message as prompt
                else:
                    chat_history.append({
                        "role": "user",
                        "parts": [msg.content]
                    })
            elif msg.role == MessageRole.ASSISTANT:
                chat_history.append({
                    "role": "model", 
                    "parts": [msg.content]
                })
        
        try:
            chat = self.model.start_chat(history=chat_history)
            
            response = await asyncio.to_thread(
                chat.send_message,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
            )
            
            content = response.text
            latency = time.time() - start_time
            
            # Gemini usage tracking
            usage = {
                'input_tokens': response.usage_metadata.prompt_token_count,
                'output_tokens': response.usage_metadata.candidates_token_count
            }
            cost = self.calculate_cost(usage)
            
            self._request_count += 1
            
            return ModelResponse(
                content=content.strip(),
                usage=usage,
                model=self.config.model_name,
                provider=self.provider,
                latency=latency,
                cost=cost
            )
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    async def stream_generate(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream Gemini responses"""
        # Format messages for Gemini
        chat_history = []
        prompt = ""
        
        for i, msg in enumerate(messages):
            if msg.role == MessageRole.SYSTEM:
                prompt = f"System: {msg.content}\n\n"
            elif msg.role == MessageRole.USER:
                if i == len(messages) - 1:
                    prompt += msg.content
                else:
                    chat_history.append({
                        "role": "user", 
                        "parts": [msg.content]
                    })
            elif msg.role == MessageRole.ASSISTANT:
                chat_history.append({
                    "role": "model",
                    "parts": [msg.content]
                })
        
        try:
            chat = self.model.start_chat(history=chat_history)
            
            response = chat.send_message(
                prompt,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
            )
            
            for chunk in response:
                yield chunk.text
                
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise


class OpenRouterClient(BaseModelClient):
    """OpenRouter client with support for multiple models via unified API"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate with OpenRouter - supports multiple models through one API"""
        start_time = time.time()
        
        # Format messages for OpenRouter (OpenAI-compatible)
        openrouter_messages = []
        for msg in messages:
            # Handle multimodal content if present
            content = msg.content
            if isinstance(content, str):
                openrouter_messages.append({
                    "role": msg.role.value,
                    "content": content
                })
            else:
                # Support for multimodal content (images, etc.)
                openrouter_messages.append({
                    "role": msg.role.value,
                    "content": content
                })
        
        try:
            # OpenRouter-specific headers for tracking
            extra_headers = kwargs.pop('extra_headers', {})
            extra_headers.update({
                "HTTP-Referer": "https://croq-ai.local",  # Optional: Your site URL
                "X-Title": "Croq AI Assistant",  # Optional: Your app name
            })
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openrouter_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                extra_headers=extra_headers,
                **kwargs
            )
            
            content = response.choices[0].message.content
            latency = time.time() - start_time
            
            usage = {
                'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                'output_tokens': response.usage.completion_tokens if response.usage else 0
            }
            cost = self.calculate_cost(usage)
            
            self._request_count += 1
            
            return ModelResponse(
                content=content.strip(),
                usage=usage,
                model=self.config.model_name,
                provider=self.provider,
                latency=latency,
                cost=cost
            )
            
        except Exception as e:
            logger.error(f"OpenRouter generation error: {e}")
            raise
    
    async def stream_generate(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream OpenRouter responses"""
        openrouter_messages = []
        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                openrouter_messages.append({
                    "role": msg.role.value,
                    "content": content
                })
            else:
                openrouter_messages.append({
                    "role": msg.role.value,
                    "content": content
                })
        
        try:
            # OpenRouter-specific headers for tracking
            extra_headers = kwargs.pop('extra_headers', {})
            extra_headers.update({
                "HTTP-Referer": "https://croq-ai.local",
                "X-Title": "Croq AI Assistant",
            })
            
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openrouter_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True,
                extra_headers=extra_headers,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenRouter streaming error: {e}")
            raise


def create_client(provider: ModelProvider) -> BaseModelClient:
    """Factory function to create model clients"""
    config = settings.get_model_config(provider)
    
    clients = {
        ModelProvider.CLAUDE: ClaudeClient,
        ModelProvider.GPT4: GPTClient, 
        ModelProvider.GROQ: GroqClient,
        ModelProvider.GEMINI: GeminiClient,
        ModelProvider.OPENROUTER: OpenRouterClient,
    }
    
    client_class = clients.get(provider)
    if not client_class:
        raise ValueError(f"Unsupported provider: {provider}")
        
    return client_class(config)
