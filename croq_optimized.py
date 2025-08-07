"""
Croq Optimized - Advanced AI Code Assistant
Competing with Claude Code and Gemini CLI
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, AsyncIterator
from pathlib import Path
import ast
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel

from config import settings, ModelProvider
from models.router import router, RouteConfig
from models.base import Message, MessageRole
from core.cache import cache
from utils.code_analysis import CodeAnalyzer
from utils.security import SecurityScanner
from utils.version_control import VersionControl


# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

console = Console()


class CroqOptimized:
    """
    Advanced AI Code Assistant with Competitive Features:
    - Multi-model routing with automatic failover
    - Intelligent caching for faster responses
    - Advanced code analysis and security scanning
    - Real-time streaming responses
    - Performance monitoring and optimization
    """
    
    def __init__(self):
        self.router = router
        self.cache = cache
        self.analyzer = CodeAnalyzer()
        self.security = SecurityScanner()
        self.version_control = VersionControl(settings.repo_path)
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "model_usage": {},
            "error_count": 0
        }
        
        # Configure router for optimal performance
        self.router.config = RouteConfig(
            prefer_speed=True,
            prefer_cost=False,
            prefer_quality=False,
            max_latency=15.0,  # 15 second timeout
            min_success_rate=0.85,
            enable_fallback=True,
            concurrent_requests=True
        )
    
    async def generate_code(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        language: str = "python",
        stream: bool = False,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate code with advanced optimizations
        
        Args:
            prompt: Code generation request
            context: Optional context/conversation history
            language: Target programming language
            stream: Enable streaming response
            use_cache: Use intelligent caching
            
        Returns:
            Dictionary with generated code, metrics, and metadata
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Build cache key
            cache_key = {
                "prompt": prompt,
                "context": context,
                "language": language,
                "model": str(self.router._select_best_model())
            }
            
            # Check cache first
            if use_cache:
                cached_result = await self.cache.get(cache_key, ["memory", "disk"])
                if cached_result:
                    self.metrics["cache_hits"] += 1
                    console.print("⚡ [green]Cache hit![/green] Returning cached result")
                    
                    # Add cache metadata
                    cached_result["metadata"]["cached"] = True
                    cached_result["metadata"]["response_time"] = time.time() - start_time
                    return cached_result
            
            # Prepare messages
            messages = self._prepare_messages(prompt, context, language)
            
            # Generate response
            if stream:
                return await self._stream_generate(messages, cache_key, start_time)
            else:
                return await self._generate_complete(messages, cache_key, start_time, use_cache)
                
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Code generation failed: {e}", exc_info=True)
            
            return {
                "code": f"# Error occurred during code generation\\n# Error: {str(e)}\\npass",
                "language": language,
                "metadata": {
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time,
                    "cached": False
                }
            }
    
    def _prepare_messages(self, prompt: str, context: Optional[str], language: str) -> List[Message]:
        """Prepare optimized messages for the model"""
        
        # Enhanced system prompt based on language
        system_prompts = {
            "python": """You are an expert Python developer. Generate clean, efficient, well-documented Python code.
Follow PEP 8 standards, use type hints, include proper error handling, and add comprehensive docstrings.
Focus on performance, security, and maintainability.""",
            
            "javascript": """You are an expert JavaScript/TypeScript developer. Generate modern, clean JavaScript code.
Use ES6+ features, proper async/await patterns, comprehensive error handling, and JSDoc comments.
Focus on performance, security, and browser compatibility.""",
            
            "rust": """You are an expert Rust developer. Generate safe, efficient, idiomatic Rust code.
Use proper ownership patterns, comprehensive error handling with Result types, and detailed documentation.
Focus on memory safety, performance, and zero-cost abstractions.""",
            
            "go": """You are an expert Go developer. Generate clean, idiomatic Go code.
Follow Go conventions, use proper error handling, include comprehensive comments, and focus on simplicity.
Emphasize concurrency patterns, performance, and maintainability."""
        }
        
        system_prompt = system_prompts.get(language, system_prompts["python"])
        
        messages = [Message(role=MessageRole.SYSTEM, content=system_prompt)]
        
        # Add context if provided
        if context:
            messages.append(Message(role=MessageRole.USER, content=f"Context: {context}"))
        
        # Add the main prompt
        messages.append(Message(role=MessageRole.USER, content=f"""
Generate {language} code for the following request:

{prompt}

Requirements:
- Write production-ready code
- Include comprehensive error handling
- Add detailed documentation/comments
- Use appropriate design patterns
- Ensure code is secure and performant
- Include type hints/annotations where applicable

Respond with ONLY the code, no explanations or markdown formatting.
"""))
        
        return messages
    
    async def _generate_complete(
        self, 
        messages: List[Message], 
        cache_key: Dict,
        start_time: float,
        use_cache: bool
    ) -> Dict[str, Any]:
        """Generate complete response with analysis and caching"""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating code...", total=None)
            
            # Generate code
            response = await self.router.generate(messages)
            
            progress.update(task, description="Analyzing code quality...")
            
            # Extract and clean code
            code = self._extract_code(response.content)
            
            # Analyze code quality
            analysis = await self.analyzer.analyze_code(code)
            
            progress.update(task, description="Running security scan...")
            
            # Security scan
            security_issues = await self.security.scan_code(code)
            
            progress.update(task, description="Finalizing...")
        
        # Build result
        result = {
            "code": code,
            "language": "python",  # Default, could be detected
            "analysis": analysis,
            "security": security_issues,
            "metadata": {
                "success": True,
                "model_used": response.provider.value,
                "model_latency": response.latency,
                "total_tokens": (response.usage or {}).get("input_tokens", 0) + 
                               (response.usage or {}).get("output_tokens", 0),
                "cost": response.cost or 0.0,
                "response_time": time.time() - start_time,
                "cached": False,
                "thinking": response.thinking  # Include Claude thinking if available
            }
        }
        
        # Update metrics
        self._update_metrics(response)
        
        # Cache the result
        if use_cache and result["metadata"]["success"]:
            await self.cache.set(
                cache_key, 
                result, 
                ttl=settings.cache.ttl,
                tags=["code_generation", result["language"]]
            )
        
        return result
    
    async def _stream_generate(
        self, 
        messages: List[Message], 
        cache_key: Dict, 
        start_time: float
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream code generation with real-time updates"""
        
        console.print("🚀 [blue]Streaming response...[/blue]")
        
        code_chunks = []
        
        try:
            async for chunk in self.router.stream_generate(messages):
                code_chunks.append(chunk)
                
                # Yield partial result
                yield {
                    "type": "chunk",
                    "content": chunk,
                    "partial_code": "".join(code_chunks),
                    "metadata": {
                        "streaming": True,
                        "response_time": time.time() - start_time
                    }
                }
            
            # Final analysis
            complete_code = "".join(code_chunks)
            clean_code = self._extract_code(complete_code)
            
            # Quick analysis for streaming
            analysis = await self.analyzer.quick_analysis(clean_code)
            
            # Final result
            final_result = {
                "type": "complete",
                "code": clean_code,
                "language": "python",
                "analysis": analysis,
                "metadata": {
                    "success": True,
                    "streaming": True,
                    "response_time": time.time() - start_time,
                    "cached": False
                }
            }
            
            # Cache the final result
            await self.cache.set(cache_key, final_result, ttl=settings.cache.ttl)
            
            yield final_result
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "metadata": {
                    "success": False,
                    "streaming": True,
                    "response_time": time.time() - start_time
                }
            }
    
    def _extract_code(self, content: str) -> str:
        """Extract clean code from model response"""
        # Remove markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Inside code block
                    # Remove language identifier
                    lines = part.strip().split("\n")
                    if lines and lines[0].strip() in ["python", "py", "javascript", "js", "rust", "go"]:
                        return "\n".join(lines[1:])
                    return part.strip()
        
        # Return content as-is if no code blocks
        return content.strip()
    
    def _update_metrics(self, response):
        """Update performance metrics"""
        if response.provider:
            provider_name = response.provider.value
            if provider_name not in self.metrics["model_usage"]:
                self.metrics["model_usage"][provider_name] = 0
            self.metrics["model_usage"][provider_name] += 1
        
        # Update average response time
        if response.latency:
            current_avg = self.metrics["avg_response_time"]
            total_requests = self.metrics["total_requests"]
            self.metrics["avg_response_time"] = (
                (current_avg * (total_requests - 1) + response.latency) / total_requests
            )
    
    async def analyze_existing_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze existing code for quality and security issues"""
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing code...", total=None)
            
            # Quality analysis
            analysis = await self.analyzer.analyze_code(code)
            
            progress.update(task, description="Security scanning...")
            
            # Security scan
            security_issues = await self.security.scan_code(code)
            
            progress.update(task, description="Generating suggestions...")
            
            # Get improvement suggestions
            suggestions = await self._get_improvement_suggestions(code, analysis, security_issues)
        
        return {
            "analysis": analysis,
            "security": security_issues,
            "suggestions": suggestions,
            "metadata": {
                "analysis_time": time.time() - start_time,
                "language": language
            }
        }
    
    async def _get_improvement_suggestions(
        self, 
        code: str, 
        analysis: Dict[str, Any], 
        security_issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate AI-powered improvement suggestions"""
        
        if not code.strip():
            return []
        
        # Build suggestion prompt
        prompt = f"""
Analyze this code and provide specific improvement suggestions:

```python
{code}
```

Quality Score: {analysis.get('quality_score', 0)}/10
Complexity: {analysis.get('complexity_score', 0)}/10
Security Issues: {len(security_issues)}

Provide 3-5 specific, actionable improvement suggestions focusing on:
1. Code quality and maintainability
2. Performance optimizations
3. Security enhancements
4. Best practices adherence

Format as a simple list, no markdown.
"""
        
        try:
            messages = [Message(role=MessageRole.USER, content=prompt)]
            response = await self.router.generate(messages)
            
            suggestions = [
                line.strip().lstrip("- ").lstrip("* ").lstrip("1234567890. ")
                for line in response.content.split("\n")
                if line.strip() and not line.startswith("#")
            ]
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return ["Unable to generate improvement suggestions"]
    
    async def explain_code(self, code: str, focus: str = "general") -> str:
        """Generate detailed code explanations"""
        
        prompt = f"""
Explain this code in detail, focusing on {focus}:

```python
{code}
```

Provide a clear, comprehensive explanation that covers:
- What the code does
- How it works
- Key algorithms or patterns used
- Potential improvements
- Edge cases or limitations

Make it educational and easy to understand.
"""
        
        try:
            messages = [Message(role=MessageRole.USER, content=prompt)]
            response = await self.router.generate(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to explain code: {e}")
            return f"Unable to explain code: {str(e)}"
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        router_metrics = self.router.get_metrics_summary()
        cache_stats = await self.cache.get_stats()
        
        return {
            "requests": {
                "total": self.metrics["total_requests"],
                "cache_hits": self.metrics["cache_hits"],
                "cache_hit_rate": (self.metrics["cache_hits"] / max(self.metrics["total_requests"], 1)) * 100,
                "errors": self.metrics["error_count"]
            },
            "performance": {
                "avg_response_time": round(self.metrics["avg_response_time"], 3),
                "model_usage": self.metrics["model_usage"]
            },
            "models": router_metrics,
            "cache": cache_stats
        }
    
    def display_stats(self):
        """Display performance statistics in a nice table"""
        
        stats = asyncio.run(self.get_performance_stats())
        
        # Create main stats table
        table = Table(title="Croq Performance Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add request stats
        table.add_row("Total Requests", str(stats["requests"]["total"]))
        table.add_row("Cache Hit Rate", f"{stats['requests']['cache_hit_rate']:.1f}%")
        table.add_row("Error Rate", f"{(stats['requests']['errors'] / max(stats['requests']['total'], 1)) * 100:.1f}%")
        table.add_row("Avg Response Time", f"{stats['performance']['avg_response_time']:.3f}s")
        
        console.print(table)
        
        # Model usage table
        if stats["performance"]["model_usage"]:
            model_table = Table(title="Model Usage", show_header=True)
            model_table.add_column("Model", style="yellow")
            model_table.add_column("Requests", style="blue")
            
            for model, count in stats["performance"]["model_usage"].items():
                model_table.add_row(model, str(count))
            
            console.print(model_table)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        
        console.print("🔍 [blue]Running health check...[/blue]")
        
        # Check model availability
        model_health = await self.router.health_check()
        
        # Check cache health
        cache_stats = await self.cache.get_stats()
        cache_healthy = cache_stats.get("total_hits", 0) >= 0  # Basic check
        
        # Check version control
        vc_healthy = self.version_control.is_healthy()
        
        health = {
            "models": model_health,
            "cache": {"healthy": cache_healthy, "stats": cache_stats},
            "version_control": {"healthy": vc_healthy},
            "overall": all(model_health.values()) and cache_healthy and vc_healthy
        }
        
        status_color = "green" if health["overall"] else "red"
        status_text = "Healthy" if health["overall"] else "Issues Detected"
        
        console.print(f"Health Status: [{status_color}]{status_text}[/{status_color}]")
        
        return health


# Global instance
croq = CroqOptimized()
