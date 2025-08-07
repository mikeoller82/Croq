"""
Optimized Configuration Management for Croq AI Code Assistant
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
import os
from pathlib import Path


class ModelProvider(str, Enum):
    """Available AI model providers"""
    CLAUDE = "claude"
    GPT4 = "gpt4" 
    GROQ = "groq"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class ModelConfig(BaseModel):
    """Configuration for specific models"""
    provider: ModelProvider
    model_name: str
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 60
    rate_limit: int = 1000  # requests per minute
    cost_per_token: float = 0.0  # for cost tracking


class CacheConfig(BaseModel):
    """Cache configuration"""
    enabled: bool = True
    ttl: int = 3600  # seconds
    max_size: int = 1000
    backend: str = "disk"  # "memory", "disk", "redis"
    redis_url: Optional[str] = None


class SecurityConfig(BaseModel):
    """Security settings"""
    enable_sandbox: bool = True
    max_execution_time: int = 30
    allowed_imports: List[str] = Field(default_factory=lambda: [
        "os", "sys", "json", "re", "math", "datetime", "pathlib",
        "typing", "collections", "itertools", "functools"
    ])
    blocked_functions: List[str] = Field(default_factory=lambda: [
        "eval", "exec", "compile", "__import__", "open", "input"
    ])


class Settings(BaseSettings):
    """Main application settings"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )
    
    # API Keys
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY") 
    groq_api_key: Optional[str] = Field(None, alias="GROQ_API_KEY")
    gemini_api_key: Optional[str] = Field(None, alias="GEMINI_API_KEY")
    openrouter_api_key: Optional[str] = Field(None, alias="OPENROUTER_API_KEY")
    
    # Model Configuration
    primary_model: ModelProvider = ModelProvider.CLAUDE
    fallback_models: List[ModelProvider] = Field(default_factory=lambda: [
        ModelProvider.GPT4, ModelProvider.GROQ, ModelProvider.GEMINI
    ])
    
    # Performance Settings
    max_concurrent_requests: int = 5
    request_timeout: int = 60
    max_retries: int = 3
    backoff_factor: float = 2.0
    
    # Cache Settings  
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    # Security Settings
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Repository Settings
    repo_path: Path = Field(default=Path("code_versions"))
    enable_version_control: bool = True
    auto_commit: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    # UI Settings
    enable_streamlit: bool = False
    streamlit_port: int = 8501
    
    def get_model_config(self, provider: ModelProvider) -> ModelConfig:
        """Get configuration for specific model provider"""
        configs = {
            ModelProvider.CLAUDE: ModelConfig(
                provider=ModelProvider.CLAUDE,
                model_name="claude-3-7-sonnet-20250219", 
                max_tokens=4000,
                temperature=0.1,
                cost_per_token=0.000015
            ),
            ModelProvider.GPT4: ModelConfig(
                provider=ModelProvider.GPT4,
                model_name="gpt-4o-mini",
                max_tokens=4000, 
                temperature=0.1,
                cost_per_token=0.00015
            ),
            ModelProvider.GROQ: ModelConfig(
                provider=ModelProvider.GROQ,
                model_name="llama-3.3-70b-versatile",
                max_tokens=8000,
                temperature=0.1,
                rate_limit=6000,
                cost_per_token=0.0  # Free tier
            ),
            ModelProvider.GEMINI: ModelConfig(
                provider=ModelProvider.GEMINI,
                model_name="gemini-2.0-flash-exp",
                max_tokens=8000,
                temperature=0.1,
                cost_per_token=0.000075
            ),
            ModelProvider.OLLAMA: ModelConfig(
                provider=ModelProvider.OLLAMA,
                model_name="qwen2.5-coder:latest",
                max_tokens=8000,
                temperature=0.1,
                cost_per_token=0.0  # Local model
            ),
            ModelProvider.OPENROUTER: ModelConfig(
                provider=ModelProvider.OPENROUTER,
                model_name="openrouter/horizon-beta",
                max_tokens=4000,
                temperature=0.1,
                cost_per_token=0.0001  # Varies by model
            )
        }
        return configs.get(provider, configs[ModelProvider.CLAUDE])
    
    def get_available_models(self) -> List[ModelProvider]:
        """Get list of available models based on API keys"""
        available = []
        
        if self.anthropic_api_key:
            available.append(ModelProvider.CLAUDE)
        if self.openai_api_key:
            available.append(ModelProvider.GPT4)  
        if self.groq_api_key:
            available.append(ModelProvider.GROQ)
        if self.gemini_api_key:
            available.append(ModelProvider.GEMINI)
        if self.openrouter_api_key:
            available.append(ModelProvider.OPENROUTER)
        
        # Ollama is available if running locally
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                available.append(ModelProvider.OLLAMA)
        except:
            pass
            
        return available or [ModelProvider.GROQ]  # Fallback to Groq


# Global settings instance
settings = Settings()
