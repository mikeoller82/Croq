"""
Croq Optimized - Advanced AI Code Assistant

A high-performance AI code assistant that competes with Claude Code and Gemini CLI.

Features:
- Multi-model routing with automatic failover
- Intelligent caching for faster responses  
- Advanced code analysis and security scanning
- Real-time streaming responses
- Performance monitoring and optimization
"""

__version__ = "2.0.0"
__author__ = "Croq Team"

from .croq_optimized import croq
from .config import settings, ModelProvider
from .models.router import router
from .core.cache import cache

__all__ = [
    "croq",
    "settings", 
    "ModelProvider",
    "router",
    "cache"
]
