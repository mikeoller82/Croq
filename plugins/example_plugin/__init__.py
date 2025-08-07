"""
Example Croq Plugin
Demonstrates all plugin system capabilities including hooks, commands, and MCP integration.
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
import json
from pathlib import Path

try:
    from core.plugins import PluginBase, HookType
    from core.hooks import HookContext
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    # Graceful fallback if dependencies aren't available
    class PluginBase:
        pass
    
    class HookType:
        PRE_GENERATION = "pre_generation"
        POST_GENERATION = "post_generation"
        PRE_REQUEST = "pre_request"
        POST_REQUEST = "post_request"
    
    class HookContext:
        def __init__(self):
            self._data = {}
        
        def get(self, key, default=None):
            return self._data.get(key, default)
        
        def set(self, key, value):
            self._data[key] = value

console = Console()

class ExamplePlugin(PluginBase):
    """
    Example plugin demonstrating all Croq plugin capabilities.
    
    This plugin provides:
    - Command examples (greet, status, config)
    - Hook examples (request logging, response enhancement)
    - Configuration management
    - Statistics tracking
    - MCP tool registration
    """
    
    def __init__(self):
        super().__init__()
        self.stats = {
            "commands_executed": 0,
            "hooks_executed": 0,
            "generation_requests": 0,
            "last_activity": None
        }
        self.config = {
            "enabled": True,
            "greeting_message": "Hello from Example Plugin!",
            "log_requests": True,
            "enhance_responses": True,
            "debug_mode": False
        }
    
    # Required plugin properties
    @property
    def name(self) -> str:
        return "example_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Example plugin demonstrating all Croq plugin capabilities"
    
    @property
    def author(self) -> str:
        return "Croq Development Team"
    
    @property
    def dependencies(self) -> List[str]:
        return ["core.hooks", "rich"]
    
    # Plugin lifecycle methods
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            console.print(f"[green]🔌 Initializing {self.name} v{self.version}[/green]")
            
            # Load configuration if it exists
            await self._load_config()
            
            # Initialize statistics
            self.stats["last_activity"] = datetime.now().isoformat()
            
            console.print(f"[green]✅ {self.name} initialized successfully[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize {self.name}: {e}[/red]")
            return False
    
    async def cleanup(self) -> bool:
        """Clean up plugin resources."""
        try:
            console.print(f"[yellow]🧹 Cleaning up {self.name}[/yellow]")
            
            # Save configuration and statistics
            await self._save_config()
            
            console.print(f"[green]✅ {self.name} cleaned up successfully[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to cleanup {self.name}: {e}[/red]")
            return False
    
    # Command system
    def get_commands(self) -> Dict[str, Callable]:
        """Return available commands."""
        return {
            "greet": self.greet_command,
            "status": self.status_command,
            "config": self.config_command,
            "stats": self.stats_command,
            "demo": self.demo_command,
            "test_hook": self.test_hook_command
        }
    
    # Hook system
    def get_hooks(self) -> Dict[HookType, List[Callable]]:
        """Return hook registrations."""
        if not self.config.get("enabled", True):
            return {}
        
        hooks = {}
        
        if self.config.get("log_requests", True):
            hooks[HookType.PRE_GENERATION] = [self.log_generation_request]
        
        if self.config.get("enhance_responses", True):
            hooks[HookType.POST_GENERATION] = [self.enhance_generation_response]
        
        # Always register request/response logging
        hooks[HookType.PRE_REQUEST] = [self.log_request]
        hooks[HookType.POST_REQUEST] = [self.log_response]
        
        return hooks
    
    # MCP tools (if applicable)
    def get_mcp_tools(self) -> Dict[str, Dict[str, Any]]:
        """Return MCP tools provided by this plugin."""
        return {
            "example_tool": {
                "name": "example_tool",
                "description": "Example tool from the example plugin",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to process"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "text", "html"],
                            "default": "text",
                            "description": "Output format"
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    
    async def handle_mcp_call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Handle MCP tool calls."""
        if tool_name == "example_tool":
            message = args.get("message", "Hello!")
            format_type = args.get("format", "text")
            
            result = f"Example plugin processed: {message}"
            
            if format_type == "json":
                return {"result": result, "plugin": self.name, "timestamp": datetime.now().isoformat()}
            elif format_type == "html":
                return f"<p><strong>Example Plugin:</strong> {result}</p>"
            else:
                return result
        
        raise ValueError(f"Unknown tool: {tool_name}")
    
    # Command implementations
    def greet_command(self, name: str = "World") -> str:
        """Greet someone with a friendly message."""
        self._update_stats("commands_executed")
        message = self.config.get("greeting_message", "Hello!")
        return f"{message} Nice to meet you, {name}! 🎉"
    
    def status_command(self) -> Dict[str, Any]:
        """Show plugin status and information."""
        self._update_stats("commands_executed")
        
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "enabled": self.config.get("enabled", True),
            "statistics": self.stats,
            "config": self.config
        }
    
    def config_command(self, key: Optional[str] = None, value: Optional[str] = None) -> Any:
        """Get or set configuration values."""
        self._update_stats("commands_executed")
        
        if key is None:
            return self.config
        
        if value is None:
            return self.config.get(key, "Configuration key not found")
        
        # Set configuration value
        # Convert string values to appropriate types
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)
        
        self.config[key] = value
        asyncio.create_task(self._save_config())
        
        return f"Configuration '{key}' set to '{value}'"
    
    def stats_command(self) -> Dict[str, Any]:
        """Show plugin statistics."""
        self._update_stats("commands_executed")
        return self.stats
    
    def demo_command(self, feature: str = "all") -> str:
        """Demonstrate plugin features."""
        self._update_stats("commands_executed")
        
        demos = {
            "hooks": "Hooks allow plugins to intercept and modify the generation process",
            "commands": "Commands provide CLI functionality through the plugin system",
            "mcp": "MCP tools allow plugins to expose functionality to external systems",
            "config": "Configuration system allows runtime plugin customization",
            "stats": "Statistics tracking helps monitor plugin usage and performance"
        }
        
        if feature == "all":
            return "Available features: " + ", ".join(demos.keys())
        
        return demos.get(feature, f"Unknown feature: {feature}")
    
    def test_hook_command(self) -> str:
        """Test hook execution by triggering a fake generation."""
        self._update_stats("commands_executed")
        
        # Create a test context
        context = HookContext()
        context.set("prompt", "Test prompt from example plugin")
        context.set("plugin_test", True)
        
        # This would normally be called by the hook system
        asyncio.create_task(self.log_generation_request(context))
        
        return "Test hook executed! Check logs for details."
    
    # Hook implementations
    async def log_generation_request(self, context: HookContext):
        """Log generation requests."""
        if not self.config.get("log_requests", True):
            return
        
        self._update_stats("hooks_executed")
        self._update_stats("generation_requests")
        
        prompt = context.get("prompt", "")
        
        if self.config.get("debug_mode", False):
            console.print(Panel(
                f"[blue]Generation Request[/blue]\n"
                f"Plugin: {self.name}\n"
                f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}",
                title="🪝 Example Plugin Hook",
                border_style="blue"
            ))
        
        # Add plugin metadata to context
        context.set("example_plugin_processed", True)
        context.set("example_plugin_timestamp", datetime.now().isoformat())
    
    async def enhance_generation_response(self, context: HookContext):
        """Enhance generation responses."""
        if not self.config.get("enhance_responses", True):
            return
        
        self._update_stats("hooks_executed")
        
        response = context.get("response", "")
        if response and self.config.get("debug_mode", False):
            console.print(f"[green]📝 Example plugin enhanced response ({len(response)} chars)[/green]")
        
        # Add enhancement metadata
        context.set("example_plugin_enhanced", True)
    
    async def log_request(self, context: HookContext):
        """Log all requests."""
        self._update_stats("hooks_executed")
        
        if self.config.get("debug_mode", False):
            request_type = context.get("request_type", "unknown")
            console.print(f"[dim]🔄 {self.name}: {request_type} request[/dim]")
    
    async def log_response(self, context: HookContext):
        """Log all responses."""
        self._update_stats("hooks_executed")
        
        if self.config.get("debug_mode", False):
            response_status = context.get("status", "unknown")
            console.print(f"[dim]✅ {self.name}: Response {response_status}[/dim]")
    
    # Helper methods
    def _update_stats(self, key: str, increment: int = 1):
        """Update plugin statistics."""
        self.stats[key] = self.stats.get(key, 0) + increment
        self.stats["last_activity"] = datetime.now().isoformat()
    
    async def _load_config(self):
        """Load plugin configuration from file."""
        config_path = Path(f"plugins/{self.name}/config.json")
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                    console.print(f"[green]📁 Loaded configuration for {self.name}[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to load config for {self.name}: {e}[/yellow]")
    
    async def _save_config(self):
        """Save plugin configuration to file."""
        config_path = Path(f"plugins/{self.name}")
        config_path.mkdir(exist_ok=True)
        
        config_file = config_path / "config.json"
        
        try:
            with open(config_file, "w") as f:
                json.dump({
                    "config": self.config,
                    "stats": self.stats
                }, f, indent=2)
                
            if self.config.get("debug_mode", False):
                console.print(f"[green]💾 Saved configuration for {self.name}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save config for {self.name}: {e}[/red]")

# Plugin factory function
def create_plugin() -> ExamplePlugin:
    """Create and return plugin instance."""
    return ExamplePlugin()

# Export the plugin class
__all__ = ["ExamplePlugin", "create_plugin"]
