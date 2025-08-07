"""
Advanced Plugin System for Croq AI Assistant
Provides dynamic loading and management of extensions
"""
import asyncio
import importlib
import inspect
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable, Union, Set
import yaml

from core.hooks import hook_manager, HookType, HookContext
from core.mcp_server import mcp_manager
from core.codebase_search import codebase_index, context_extractor

logger = logging.getLogger(__name__)

@dataclass
class PluginInfo:
    """Plugin metadata"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    mcp_tools: List[str] = field(default_factory=list)
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

class PluginBase(ABC):
    """Base class for all plugins"""
    
    def __init__(self, plugin_manager: 'PluginManager'):
        self.plugin_manager = plugin_manager
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"plugin.{self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description"""
        pass
    
    @property
    def author(self) -> str:
        """Plugin author"""
        return "Unknown"
    
    @property
    def dependencies(self) -> List[str]:
        """Plugin dependencies"""
        return []
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the plugin"""
        self.config = config or {}
        return True
    
    async def shutdown(self) -> None:
        """Shutdown the plugin"""
        pass
    
    def get_hooks(self) -> Dict[HookType, List[Callable]]:
        """Get hooks provided by this plugin"""
        return {}
    
    def get_commands(self) -> Dict[str, Callable]:
        """Get commands provided by this plugin"""
        return {}
    
    def get_mcp_tools(self) -> Dict[str, Callable]:
        """Get MCP tools provided by this plugin"""
        return {}

class PluginManager:
    """Manages plugin loading, initialization, and lifecycle"""
    
    def __init__(self, plugins_dir: Path = None):
        self.plugins_dir = plugins_dir or Path("plugins")
        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        self.enabled_plugins: Set[str] = set()
        
        # Create plugins directory if it doesn't exist
        self.plugins_dir.mkdir(exist_ok=True)
    
    async def load_all_plugins(self) -> Dict[str, bool]:
        """Load all plugins from the plugins directory"""
        results = {}
        
        for plugin_path in self.plugins_dir.iterdir():
            if plugin_path.is_dir() and not plugin_path.name.startswith('.'):
                success = await self.load_plugin(plugin_path.name)
                results[plugin_path.name] = success
        
        return results
    
    async def load_plugin(self, plugin_name: str) -> bool:
        """Load a single plugin"""
        plugin_path = self.plugins_dir / plugin_name
        
        if not plugin_path.exists():
            logger.error(f"Plugin directory not found: {plugin_path}")
            return False
        
        try:
            # Load plugin metadata
            metadata_file = plugin_path / "plugin.yaml"
            if not metadata_file.exists():
                metadata_file = plugin_path / "plugin.json"
            
            if metadata_file.exists():
                if metadata_file.suffix == ".yaml":
                    with open(metadata_file) as f:
                        metadata = yaml.safe_load(f)
                else:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                
                plugin_info = PluginInfo(**metadata)
                self.plugin_info[plugin_name] = plugin_info
            else:
                logger.warning(f"No metadata file found for plugin {plugin_name}")
                plugin_info = PluginInfo(
                    name=plugin_name,
                    version="unknown",
                    description="",
                    author="unknown"
                )
                self.plugin_info[plugin_name] = plugin_info
            
            # Check dependencies
            if not await self._check_dependencies(plugin_info):
                logger.error(f"Plugin {plugin_name} dependencies not met")
                return False
            
            # Import plugin module
            plugin_module_path = plugin_path / "__init__.py"
            if not plugin_module_path.exists():
                logger.error(f"Plugin __init__.py not found: {plugin_module_path}")
                return False
            
            # Add plugin path to Python path
            if str(plugin_path.parent) not in sys.path:
                sys.path.insert(0, str(plugin_path.parent))
            
            # Import the plugin module
            try:
                module = importlib.import_module(plugin_name)
                importlib.reload(module)  # Reload if already imported
            except Exception as e:
                logger.error(f"Failed to import plugin {plugin_name}: {e}")
                return False
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginBase) and 
                    obj != PluginBase):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.error(f"No plugin class found in {plugin_name}")
                return False
            
            # Instantiate plugin
            plugin_instance = plugin_class(self)
            
            # Initialize plugin
            config = plugin_info.config if hasattr(plugin_info, 'config') else {}
            if not await plugin_instance.initialize(config):
                logger.error(f"Plugin {plugin_name} initialization failed")
                return False
            
            # Register plugin
            self.plugins[plugin_name] = plugin_instance
            
            # Register hooks
            hooks = plugin_instance.get_hooks()
            for hook_type, hook_functions in hooks.items():
                for hook_func in hook_functions:
                    hook_manager.register_hook(
                        hook_type=hook_type,
                        function=hook_func,
                        name=f"{plugin_name}.{hook_func.__name__}",
                        description=f"Hook from plugin {plugin_name}"
                    )
            
            # Register MCP tools
            mcp_tools = plugin_instance.get_mcp_tools()
            if mcp_tools:
                await self._register_mcp_tools(plugin_name, mcp_tools)
            
            self.enabled_plugins.add(plugin_name)
            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    async def _check_dependencies(self, plugin_info: PluginInfo) -> bool:
        """Check if plugin dependencies are met"""
        for dep in plugin_info.dependencies:
            if dep not in self.enabled_plugins:
                # Try to load the dependency
                if not await self.load_plugin(dep):
                    return False
        return True
    
    async def _register_mcp_tools(self, plugin_name: str, tools: Dict[str, Callable]):
        """Register MCP tools from a plugin"""
        # This would integrate with the MCP server system
        # For now, just log the registration
        logger.info(f"Plugin {plugin_name} registered {len(tools)} MCP tools")
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self.plugins:
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            
            # Shutdown plugin
            await plugin.shutdown()
            
            # Unregister hooks
            hooks = plugin.get_hooks()
            for hook_type, hook_functions in hooks.items():
                for hook_func in hook_functions:
                    hook_manager.unregister_hook(hook_type, f"{plugin_name}.{hook_func.__name__}")
            
            # Remove from loaded plugins
            del self.plugins[plugin_name]
            self.enabled_plugins.discard(plugin_name)
            
            logger.info(f"Successfully unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin"""
        if plugin_name in self.plugins:
            if not await self.unload_plugin(plugin_name):
                return False
        
        return await self.load_plugin(plugin_name)
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """Get a loaded plugin instance"""
        return self.plugins.get(plugin_name)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin metadata"""
        return self.plugin_info.get(plugin_name)
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded plugins with their info"""
        result = {}
        for name, plugin in self.plugins.items():
            info = self.plugin_info.get(name)
            result[name] = {
                "instance": plugin,
                "info": asdict(info) if info else {},
                "enabled": name in self.enabled_plugins
            }
        return result
    
    async def call_plugin_command(self, plugin_name: str, command: str, *args, **kwargs) -> Any:
        """Call a command from a specific plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not loaded")
        
        plugin = self.plugins[plugin_name]
        commands = plugin.get_commands()
        
        if command not in commands:
            raise ValueError(f"Command {command} not found in plugin {plugin_name}")
        
        command_func = commands[command]
        
        # Call command (async or sync)
        if inspect.iscoroutinefunction(command_func):
            return await command_func(*args, **kwargs)
        else:
            return command_func(*args, **kwargs)
    
    def get_all_commands(self) -> Dict[str, Tuple[str, Callable]]:
        """Get all commands from all plugins"""
        commands = {}
        for plugin_name, plugin in self.plugins.items():
            plugin_commands = plugin.get_commands()
            for cmd_name, cmd_func in plugin_commands.items():
                # Use plugin.command format to avoid conflicts
                full_cmd_name = f"{plugin_name}.{cmd_name}"
                commands[full_cmd_name] = (plugin_name, cmd_func)
        return commands

# Global plugin manager
plugin_manager = PluginManager()

# Built-in plugins

class CodeAnalysisPlugin(PluginBase):
    """Built-in code analysis plugin"""
    
    @property
    def name(self) -> str:
        return "code_analysis"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Provides code analysis and search capabilities"
    
    @property
    def author(self) -> str:
        return "Croq Team"
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        self.config = config or {}
        return True
    
    def get_hooks(self) -> Dict[HookType, List[Callable]]:
        return {
            HookType.PRE_GENERATION: [self.enhance_with_code_context]
        }
    
    def get_commands(self) -> Dict[str, Callable]:
        return {
            "index": self.index_codebase,
            "search": self.search_code,
            "stats": self.get_stats,
            "symbols": self.search_symbols
        }
    
    async def enhance_with_code_context(self, context: HookContext):
        """Hook to enhance generation with code context"""
        query = context.get("prompt", "")
        if len(query) > 10:  # Only for substantial queries
            try:
                code_context = await context_extractor.extract_context_for_query(
                    query, Path(".")
                )
                if code_context.strip():
                    context.set("code_context", code_context)
                    self.logger.debug("Enhanced context with code analysis")
            except Exception as e:
                self.logger.warning(f"Failed to enhance with code context: {e}")
    
    async def index_codebase(self, path: str = ".") -> Dict[str, Any]:
        """Index the codebase for search"""
        result = await codebase_index.index_directory(Path(path))
        return result
    
    def search_code(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search the codebase"""
        results = codebase_index.search(query, max_results)
        return [
            {
                "file": str(r.file_path),
                "line": r.line_number,
                "content": r.line_content,
                "relevance": r.relevance_score
            }
            for r in results
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get codebase statistics"""
        stats = codebase_index.get_codebase_stats()
        return {
            "total_files": stats.total_files,
            "total_lines": stats.total_lines,
            "languages": stats.languages,
            "symbols": stats.symbols,
            "complexity": stats.complexity_score
        }
    
    def search_symbols(self, query: str, symbol_type: str = None) -> List[Dict[str, Any]]:
        """Search for symbols"""
        symbols = codebase_index.search_symbols(query, symbol_type)
        return [
            {
                "name": s.name,
                "type": s.type,
                "file": str(s.file_path),
                "line": s.line_number,
                "signature": s.signature,
                "complexity": s.complexity
            }
            for s in symbols
        ]

class MCPIntegrationPlugin(PluginBase):
    """Built-in MCP integration plugin"""
    
    @property
    def name(self) -> str:
        return "mcp_integration"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Integrates with MCP servers for external tool access"
    
    @property
    def author(self) -> str:
        return "Croq Team"
    
    def get_commands(self) -> Dict[str, Callable]:
        return {
            "connect": self.connect_servers,
            "list_tools": self.list_tools,
            "call_tool": self.call_tool,
            "servers": self.list_servers
        }
    
    async def connect_servers(self) -> Dict[str, bool]:
        """Connect to all configured MCP servers"""
        return await mcp_manager.connect_all()
    
    def list_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all available MCP tools"""
        tools = mcp_manager.get_all_tools()
        result = {}
        for server_name, server_tools in tools.items():
            result[server_name] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                }
                for tool in server_tools.values()
            ]
        return result
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], server_name: str = None) -> Any:
        """Call an MCP tool"""
        if server_name:
            return await mcp_manager.call_tool(server_name, tool_name, arguments)
        else:
            return await mcp_manager.find_and_call_tool(tool_name, arguments)
    
    def list_servers(self) -> List[str]:
        """List all configured MCP servers"""
        return list(mcp_manager.servers.keys())

# Register built-in plugins
async def initialize_builtin_plugins():
    """Initialize built-in plugins"""
    # Register code analysis plugin
    code_plugin = CodeAnalysisPlugin(plugin_manager)
    await code_plugin.initialize()
    plugin_manager.plugins["code_analysis"] = code_plugin
    plugin_manager.enabled_plugins.add("code_analysis")
    
    # Register hooks
    hooks = code_plugin.get_hooks()
    for hook_type, hook_functions in hooks.items():
        for hook_func in hook_functions:
            hook_manager.register_hook(
                hook_type=hook_type,
                function=hook_func,
                name=f"code_analysis.{hook_func.__name__}",
                description=f"Hook from code_analysis plugin"
            )
    
    # Register MCP plugin
    mcp_plugin = MCPIntegrationPlugin(plugin_manager)
    await mcp_plugin.initialize()
    plugin_manager.plugins["mcp_integration"] = mcp_plugin
    plugin_manager.enabled_plugins.add("mcp_integration")
    
    logger.info("Built-in plugins initialized")

# Plugin discovery and template creation
def create_plugin_template(plugin_name: str, plugin_path: Path = None) -> bool:
    """Create a template for a new plugin"""
    if not plugin_path:
        plugin_path = plugin_manager.plugins_dir / plugin_name
    
    if plugin_path.exists():
        logger.error(f"Plugin directory already exists: {plugin_path}")
        return False
    
    plugin_path.mkdir(parents=True)
    
    # Create plugin.yaml
    metadata = {
        "name": plugin_name,
        "version": "1.0.0",
        "description": f"{plugin_name} plugin",
        "author": "Unknown",
        "dependencies": [],
        "hooks": [],
        "commands": []
    }
    
    with open(plugin_path / "plugin.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    # Create __init__.py
    template_code = f'''"""
{plugin_name} Plugin
"""
from core.plugins import PluginBase, HookType
from typing import Dict, List, Callable, Any

class {plugin_name.title().replace("_", "")}Plugin(PluginBase):
    """Main plugin class"""
    
    @property
    def name(self) -> str:
        return "{plugin_name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "{plugin_name} plugin description"
    
    @property
    def author(self) -> str:
        return "Your Name"
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the plugin"""
        self.config = config or {{}}
        self.logger.info(f"{{self.name}} plugin initialized")
        return True
    
    async def shutdown(self) -> None:
        """Shutdown the plugin"""
        self.logger.info(f"{{self.name}} plugin shutdown")
    
    def get_hooks(self) -> Dict[HookType, List[Callable]]:
        """Return hooks provided by this plugin"""
        return {{
            # Example: HookType.PRE_GENERATION: [self.my_hook_function]
        }}
    
    def get_commands(self) -> Dict[str, Callable]:
        """Return commands provided by this plugin"""
        return {{
            "hello": self.hello_command,
            # Add more commands here
        }}
    
    def hello_command(self) -> str:
        """Example command"""
        return f"Hello from {{self.name}} plugin!"
'''
    
    with open(plugin_path / "__init__.py", "w") as f:
        f.write(template_code)
    
    logger.info(f"Plugin template created at: {plugin_path}")
    return True
