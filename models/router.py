"""
Intelligent Model Router with Automatic Failover and Performance Optimization
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

from models.base import BaseModelClient, create_client, Message, ModelResponse
from config import settings, ModelProvider


logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Track model performance metrics"""
    latency_history: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    total_cost: float = 0.0
    last_used: float = field(default_factory=time.time)
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latency_history) if self.latency_history else 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 1.0
    
    @property
    def score(self) -> float:
        """Calculate overall model score (higher is better)"""
        if not self.latency_history:
            return 0.5  # Neutral score for untested models
        
        # Normalize metrics to 0-1 range
        speed_score = max(0, 1 - (self.avg_latency / 10))  # 10s = 0 score
        reliability_score = self.success_rate
        
        # Cost penalty (lower cost = higher score)
        cost_score = 1.0 if self.total_cost == 0 else max(0, 1 - (self.total_cost / 1.0))
        
        # Weighted combination
        return (speed_score * 0.4 + reliability_score * 0.5 + cost_score * 0.1)


@dataclass 
class RouteConfig:
    """Configuration for routing decisions"""
    prefer_speed: bool = True
    prefer_cost: bool = False
    prefer_quality: bool = False
    max_latency: float = 30.0  # seconds
    min_success_rate: float = 0.8
    enable_fallback: bool = True
    concurrent_requests: bool = True


class ModelRouter:
    """Intelligent router for AI models with automatic failover"""
    
    def __init__(self, config: RouteConfig = None):
        self.config = config or RouteConfig()
        self.clients: Dict[ModelProvider, BaseModelClient] = {}
        self.metrics: Dict[ModelProvider, ModelMetrics] = defaultdict(ModelMetrics)
        self._active_requests: Dict[str, List[asyncio.Task]] = defaultdict(list)
        
        # Initialize available clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available model clients"""
        available_models = settings.get_available_models()
        
        for provider in available_models:
            try:
                self.clients[provider] = create_client(provider)
                logger.info(f"Initialized client for {provider.value}")
            except Exception as e:
                logger.error(f"Failed to initialize {provider.value} client: {e}")
    
    def _select_best_model(self, task_type: str = "general") -> ModelProvider:
        """Select the best model based on current metrics and configuration"""
        available_providers = list(self.clients.keys())
        
        if not available_providers:
            raise RuntimeError("No available model providers")
        
        # Filter models by minimum requirements
        viable_models = []
        for provider in available_providers:
            metrics = self.metrics[provider]
            
            if (metrics.success_rate >= self.config.min_success_rate and 
                metrics.avg_latency <= self.config.max_latency):
                viable_models.append((provider, metrics))
        
        # If no models meet requirements, use all available
        if not viable_models:
            viable_models = [(p, self.metrics[p]) for p in available_providers]
        
        # Score-based selection
        if self.config.prefer_speed:
            # Prioritize fast models
            best = min(viable_models, key=lambda x: x[1].avg_latency or 0.1)
        elif self.config.prefer_cost:
            # Prioritize free/cheap models
            best = min(viable_models, key=lambda x: x[1].total_cost)
        else:
            # Balanced selection using composite score
            best = max(viable_models, key=lambda x: x[1].score)
        
        provider = best[0]
        self.metrics[provider].last_used = time.time()
        
        logger.debug(f"Selected {provider.value} (score: {self.metrics[provider].score:.3f})")
        return provider
    
    def _update_metrics(self, provider: ModelProvider, response: ModelResponse, error: bool = False):
        """Update performance metrics for a model"""
        metrics = self.metrics[provider]
        
        if error:
            metrics.error_count += 1
        else:
            metrics.success_count += 1
            if response.latency:
                metrics.latency_history.append(response.latency)
            if response.cost:
                metrics.total_cost += response.cost
    
    async def _try_model(self, provider: ModelProvider, messages: List[Message], **kwargs) -> ModelResponse:
        """Try a specific model and handle errors"""
        client = self.clients[provider]
        
        try:
            async with client:
                response = await client.generate(messages, **kwargs)
                self._update_metrics(provider, response, error=False)
                return response
                
        except Exception as e:
            logger.warning(f"{provider.value} generation failed: {e}")
            self._update_metrics(provider, None, error=True)
            raise
    
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate response using the best available model with fallback"""
        if self.config.concurrent_requests and len(self.clients) > 1:
            return await self._concurrent_generate(messages, **kwargs)
        else:
            return await self._sequential_generate(messages, **kwargs)
    
    async def _sequential_generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Try models sequentially with fallback"""
        primary_provider = self._select_best_model()
        tried_providers = set()
        
        # Try primary model
        try:
            return await self._try_model(primary_provider, messages, **kwargs)
        except Exception as e:
            tried_providers.add(primary_provider)
            logger.warning(f"Primary model {primary_provider.value} failed: {e}")
        
        # Try fallback models if enabled
        if self.config.enable_fallback:
            for provider in settings.fallback_models:
                if provider in self.clients and provider not in tried_providers:
                    try:
                        return await self._try_model(provider, messages, **kwargs)
                    except Exception as e:
                        tried_providers.add(provider)
                        logger.warning(f"Fallback model {provider.value} failed: {e}")
        
        # Last resort - try any remaining models
        for provider in self.clients:
            if provider not in tried_providers:
                try:
                    return await self._try_model(provider, messages, **kwargs)
                except Exception as e:
                    logger.error(f"Last resort model {provider.value} failed: {e}")
        
        raise RuntimeError("All available models failed to generate response")
    
    async def _concurrent_generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate responses concurrently and return the fastest"""
        tasks = []
        providers = []
        
        # Start with best model + one fallback for speed
        primary_provider = self._select_best_model()
        providers.append(primary_provider)
        
        # Add fastest fallback if different from primary
        fallback_candidates = [p for p in self.clients if p != primary_provider]
        if fallback_candidates:
            fastest_fallback = min(fallback_candidates, 
                                 key=lambda p: self.metrics[p].avg_latency or 0.1)
            providers.append(fastest_fallback)
        
        # Create concurrent tasks
        for provider in providers:
            task = asyncio.create_task(self._try_model(provider, messages, **kwargs))
            tasks.append(task)
        
        try:
            # Wait for first successful response
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
            
            # Return first successful result
            for task in done:
                try:
                    result = await task
                    logger.info(f"Concurrent generation completed by {result.provider.value}")
                    return result
                except Exception as e:
                    logger.warning(f"Concurrent task failed: {e}")
            
            # If all concurrent tasks failed, fall back to sequential
            logger.warning("All concurrent tasks failed, falling back to sequential")
            return await self._sequential_generate(messages, **kwargs)
            
        except Exception as e:
            # Cancel all tasks on error
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
    
    async def stream_generate(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream generation using best available model"""
        provider = self._select_best_model()
        client = self.clients[provider]
        
        try:
            async with client:
                async for chunk in client.stream_generate(messages, **kwargs):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Streaming generation failed for {provider.value}: {e}")
            self._update_metrics(provider, None, error=True)
            
            # Try fallback for streaming
            if self.config.enable_fallback:
                for fallback_provider in settings.fallback_models:
                    if fallback_provider in self.clients and fallback_provider != provider:
                        try:
                            fallback_client = self.clients[fallback_provider]
                            async with fallback_client:
                                async for chunk in fallback_client.stream_generate(messages, **kwargs):
                                    yield chunk
                            return
                        except Exception as fe:
                            logger.warning(f"Fallback streaming failed for {fallback_provider.value}: {fe}")
            
            raise RuntimeError("All streaming models failed")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary for all models"""
        summary = {}
        
        for provider, metrics in self.metrics.items():
            summary[provider.value] = {
                "avg_latency": round(metrics.avg_latency, 3),
                "success_rate": round(metrics.success_rate, 3),
                "total_requests": metrics.success_count + metrics.error_count,
                "total_cost": round(metrics.total_cost, 4),
                "score": round(metrics.score, 3),
                "last_used": time.time() - metrics.last_used
            }
        
        return summary
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.metrics.clear()
        logger.info("Model metrics reset")
    
    async def health_check(self) -> Dict[ModelProvider, bool]:
        """Check health of all available models"""
        health_status = {}
        test_messages = [Message(role="user", content="Hello, respond with 'OK'")]
        
        for provider, client in self.clients.items():
            try:
                async with client:
                    response = await asyncio.wait_for(
                        client.generate(test_messages), 
                        timeout=10.0
                    )
                    health_status[provider] = True
                    logger.debug(f"{provider.value} health check passed")
            except Exception as e:
                health_status[provider] = False
                logger.warning(f"{provider.value} health check failed: {e}")
        
        return health_status


# Global router instance
router = ModelRouter()
