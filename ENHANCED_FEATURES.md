# 🚀 Croq Enhanced Features Documentation

Welcome to the enhanced Croq AI Assistant! This document covers all the new advanced features that make Croq a truly powerful development companion.

## 🆕 What's New

### 🪝 Hook System
- **Extensible Architecture**: Add custom functionality at key points in the generation process
- **Pre/Post Generation Hooks**: Enhance context before generation and process results after
- **Plugin Integration**: Hooks can be registered by plugins for modular functionality
- **Performance Monitoring**: Track hook execution statistics and success rates

### 🔌 MCP Server Integration
- **Model Context Protocol**: Connect to external tools and services using the MCP standard
- **Multiple Transports**: Support for stdio, HTTP, and WebSocket connections
- **Tool Discovery**: Automatic discovery of available tools and resources
- **Built-in Servers**: File system, Git, and web search integrations

### 🔍 Advanced Codebase Search
- **Intelligent Indexing**: SQLite-based indexing with language-aware analysis
- **Symbol Recognition**: Find functions, classes, variables across your codebase
- **Context Extraction**: Automatically enhance AI prompts with relevant code context
- **Multi-language Support**: Python, JavaScript, TypeScript, Java, Go, Rust, and more

### 🧩 Plugin System
- **Dynamic Loading**: Load and unload plugins at runtime
- **Plugin Templates**: Easy plugin creation with built-in templates
- **Hook Integration**: Plugins can register hooks for custom functionality
- **Command System**: Plugins can expose commands through the CLI

## 📚 Getting Started

### Installation

1. **Install Enhanced Dependencies**:
   ```bash
   pip install -r requirements_optimized.txt
   ```

2. **Initialize Systems**:
   ```bash
   python enhanced_cli.py --help
   ```

### Quick Start

1. **Index Your Codebase**:
   ```bash
   python enhanced_cli.py search index
   ```

2. **Generate Code with Enhanced Context**:
   ```bash
   python enhanced_cli.py generate "create a REST API endpoint" --codebase
   ```

3. **Start Enhanced Interactive Mode**:
   ```bash
   python enhanced_cli.py interactive
   ```

## 🪝 Hook System

### Overview
The hook system allows you to inject custom functionality at specific points during code generation and analysis.

### Available Hook Types
- `PRE_GENERATION`: Before AI generation starts
- `POST_GENERATION`: After AI generation completes
- `PRE_REQUEST`: Before making API requests
- `POST_REQUEST`: After API requests complete
- `ON_ERROR`: When errors occur
- `ON_CACHE_HIT`: When cache is used
- `ON_CACHE_MISS`: When cache is missed
- `PRE_ANALYSIS`: Before code analysis
- `POST_ANALYSIS`: After code analysis
- `ON_FILE_CHANGE`: When files are modified
- `ON_MODEL_SWITCH`: When AI models are switched

### Using Hooks

#### CLI Commands
```bash
# List all registered hooks
python enhanced_cli.py hooks list

# Show hook execution statistics
python enhanced_cli.py hooks stats

# Enable/disable specific hooks
python enhanced_cli.py hooks enable hook_name
python enhanced_cli.py hooks disable hook_name
```

#### Creating Custom Hooks
```python
from core.hooks import hook_manager, HookType, HookContext

@hook_manager.hook(HookType.PRE_GENERATION, description="My custom hook")
async def my_custom_hook(context: HookContext):
    # Access context data
    prompt = context.get("prompt", "")
    
    # Modify context
    context.set("custom_data", "enhanced")
    
    # Your custom logic here
    print(f"Processing prompt: {prompt}")
```

### Built-in Hooks
- **Logging Hooks**: Automatic logging of generation requests and results
- **Context Enhancement**: Adds codebase context to generation prompts
- **Error Handling**: Comprehensive error logging with context

## 🔌 MCP Server Integration

### Overview
MCP (Model Context Protocol) allows Croq to connect to external tools and services, extending its capabilities beyond just AI generation.

### Supported Transports
- **STDIO**: Direct process communication (recommended for local tools)
- **HTTP**: REST API integration
- **WebSocket**: Real-time bidirectional communication

### CLI Commands
```bash
# Connect to all configured MCP servers
python enhanced_cli.py mcp connect

# List all MCP servers
python enhanced_cli.py mcp servers

# List available tools
python enhanced_cli.py mcp tools
python enhanced_cli.py mcp tools --server filesystem

# Call MCP tools
python enhanced_cli.py mcp call read_file '{"path": "README.md"}'
```

### Built-in MCP Servers
1. **Filesystem Server**: File operations and directory traversal
2. **Git Server**: Version control operations
3. **Search Server**: Web search capabilities

### Adding Custom MCP Servers
```python
from core.mcp_server import mcp_manager, MCPServerConfig, MCPTransport

# Add a custom server
config = MCPServerConfig(
    name="my_tool",
    command=["python", "my_mcp_server.py"],
    transport=MCPTransport.STDIO,
    auto_restart=True
)
mcp_manager.add_server(config)
```

### Interactive MCP Usage
In interactive mode, use MCP commands:
```
croq> mcp servers
croq> mcp tools
croq> mcp call read_file {"path": "config.py"}
```

## 🔍 Advanced Codebase Search

### Overview
The codebase search system provides intelligent indexing and search capabilities, allowing AI to understand your codebase context.

### Features
- **Language-Aware Parsing**: Understands Python, JavaScript, TypeScript, and more
- **Symbol Extraction**: Functions, classes, variables, and their relationships
- **Complexity Analysis**: Cyclomatic complexity scoring
- **Relevance Scoring**: Smart ranking of search results
- **Context Preservation**: Maintains code context around matches

### CLI Commands
```bash
# Index your codebase
python enhanced_cli.py search index --path .

# Search for code
python enhanced_cli.py search search "authentication"

# Search for specific symbols
python enhanced_cli.py search symbols "login" --type function

# Show indexing statistics
python enhanced_cli.py search stats
```

### Interactive Search
```
croq> search authentication
croq> search symbols UserManager
croq> search stats
```

### Programmatic Usage
```python
from core.codebase_search import codebase_index, context_extractor
from pathlib import Path

# Index a directory
result = await codebase_index.index_directory(Path("."))

# Search code
results = codebase_index.search("authentication", max_results=10)

# Search symbols
symbols = codebase_index.search_symbols("login", symbol_type="function")

# Extract context for AI
context = await context_extractor.extract_context_for_query("create login", Path("."))
```

### Automatic Context Enhancement
When enabled (default), the codebase search automatically enhances AI generation prompts with relevant code context:

```bash
# This will automatically include relevant code context
python enhanced_cli.py generate "add error handling to the login function"
```

## 🧩 Plugin System

### Overview
The plugin system provides a powerful way to extend Croq's functionality through dynamically loaded modules.

### Plugin Architecture
- **Plugin Base Class**: All plugins inherit from `PluginBase`
- **Hook Integration**: Plugins can register hooks
- **Command System**: Plugins can expose CLI commands
- **Dependency Management**: Automatic plugin dependency resolution
- **Hot Reloading**: Load, unload, and reload plugins at runtime

### CLI Commands
```bash
# List all plugins
python enhanced_cli.py plugins list

# Load a plugin
python enhanced_cli.py plugins load my_plugin

# Unload a plugin
python enhanced_cli.py plugins unload my_plugin

# Reload a plugin
python enhanced_cli.py plugins reload my_plugin

# List plugin commands
python enhanced_cli.py plugins commands

# Call plugin commands
python enhanced_cli.py plugins call code_analysis.search '["authentication"]'

# Create new plugin template
python enhanced_cli.py plugins create my_new_plugin
```

### Creating Plugins

1. **Create Plugin Template**:
   ```bash
   python enhanced_cli.py plugins create my_awesome_plugin
   ```

2. **Edit the Plugin**: Modify `plugins/my_awesome_plugin/__init__.py`

3. **Plugin Structure**:
   ```python
   from core.plugins import PluginBase, HookType
   from typing import Dict, List, Callable, Any

   class MyAwesomePlugin(PluginBase):
       @property
       def name(self) -> str:
           return "my_awesome_plugin"
       
       @property
       def version(self) -> str:
           return "1.0.0"
       
       @property
       def description(self) -> str:
           return "An awesome plugin that does amazing things"
       
       def get_commands(self) -> Dict[str, Callable]:
           return {
               "awesome_command": self.awesome_command,
           }
       
       def get_hooks(self) -> Dict[HookType, List[Callable]]:
           return {
               HookType.PRE_GENERATION: [self.enhance_generation]
           }
       
       def awesome_command(self, message: str = "Hello!") -> str:
           return f"Awesome plugin says: {message}"
       
       async def enhance_generation(self, context):
           context.set("awesome_enhancement", True)
   ```

4. **Load the Plugin**:
   ```bash
   python enhanced_cli.py plugins load my_awesome_plugin
   ```

### Built-in Plugins

#### Code Analysis Plugin
- **Commands**: `index`, `search`, `stats`, `symbols`
- **Hooks**: Automatic context enhancement
- **Purpose**: Provides codebase search and analysis

#### MCP Integration Plugin  
- **Commands**: `connect`, `list_tools`, `call_tool`, `servers`
- **Purpose**: Manages MCP server connections and tool calls

### Plugin Configuration
Create `plugins/my_plugin/plugin.yaml`:
```yaml
name: my_plugin
version: 1.0.0
description: My awesome plugin
author: Your Name
dependencies: []
hooks:
  - pre_generation
commands:
  - awesome_command
config:
  enable_feature: true
  api_endpoint: "https://api.example.com"
```

## 🎯 Enhanced Interactive Mode

### Overview
The enhanced interactive mode provides a comprehensive command-line interface with all advanced features accessible through natural commands.

### Starting Interactive Mode
```bash
python enhanced_cli.py interactive
```

### Available Commands

#### Code Generation
```
croq> create a REST API endpoint for user authentication
```

#### Hook System
```
croq> hook list
croq> hook stats
croq> hook enable my_hook
croq> hook disable my_hook
```

#### MCP Integration
```
croq> mcp servers
croq> mcp tools
croq> mcp call read_file {"path": "README.md"}
```

#### Codebase Search
```
croq> search authentication
croq> search symbols UserManager
croq> search stats
```

#### Plugin System
```
croq> plugin list
croq> plugin commands
croq> plugin call code_analysis.search ["login"]
```

#### General Commands
```
croq> help
croq> exit
```

### Auto-Enhancement
When you start interactive mode, Croq will offer to automatically index your codebase for enhanced context. This dramatically improves the quality of generated code by providing relevant context from your existing codebase.

## 🔧 Configuration

### Environment Variables
```bash
# OpenRouter API Key (NEW!)
export OPENROUTER_API_KEY="your_openrouter_key"

# Existing API keys
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENAI_API_KEY="your_openai_key"
export GROQ_API_KEY="your_groq_key"
export GEMINI_API_KEY="your_gemini_key"
```

### Configuration Files
Create `.env` file in your project root:
```env
# OpenRouter Integration
OPENROUTER_API_KEY=your_openrouter_key_here

# Other API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key

# Optional MCP Configuration
MCP_FILESYSTEM_ROOT=.
MCP_ENABLE_GIT=true
MCP_SEARCH_ENABLED=true
```

## 🚀 Performance & Optimization

### Codebase Indexing
- **Incremental Updates**: Only re-index changed files
- **Gitignore Respect**: Automatically respects .gitignore patterns
- **Language Detection**: Smart file type detection
- **Binary Filtering**: Skips binary files and large files (>1MB)

### Hook System Performance
- **Async Execution**: All hooks run asynchronously
- **Error Isolation**: Hook failures don't break the main process
- **Statistics Tracking**: Monitor hook performance and success rates
- **Conditional Execution**: Hooks can have conditions for when to run

### MCP Server Management
- **Connection Pooling**: Efficient management of server connections
- **Auto-Restart**: Failed servers are automatically restarted
- **Health Monitoring**: Regular health checks for all servers
- **Timeout Handling**: Robust timeout and error handling

### Plugin System Optimization
- **Lazy Loading**: Plugins are loaded only when needed
- **Dependency Resolution**: Smart dependency loading order
- **Memory Management**: Proper cleanup when plugins are unloaded
- **Hot Reloading**: Efficient plugin reloading without restart

## 🐛 Troubleshooting

### Common Issues

#### Codebase Indexing Fails
```bash
# Check permissions
ls -la .croq_index.db

# Re-index with verbose output
python enhanced_cli.py search index --path . -v
```

#### MCP Server Connection Issues
```bash
# Check server status
python enhanced_cli.py mcp servers

# Manually reconnect
python enhanced_cli.py mcp connect
```

#### Plugin Loading Errors
```bash
# Check plugin structure
ls -la plugins/my_plugin/

# View detailed errors
python enhanced_cli.py plugins load my_plugin -v
```

#### Hook Execution Problems
```bash
# Check hook statistics
python enhanced_cli.py hooks stats

# Disable problematic hooks
python enhanced_cli.py hooks disable problematic_hook
```

### Debug Mode
Enable debug logging:
```bash
export CROQ_LOG_LEVEL=DEBUG
python enhanced_cli.py interactive
```

### Database Issues
If the search index gets corrupted:
```bash
# Remove and recreate index
rm .croq_index.db
python enhanced_cli.py search index
```

## 🤝 Contributing

### Adding New Features
1. **Hooks**: Add new hook types in `core/hooks.py`
2. **MCP Servers**: Create new MCP server configurations
3. **Search Languages**: Add language analyzers in `core/codebase_search.py`
4. **Plugins**: Create new built-in plugins

### Code Style
- Follow existing patterns and conventions
- Add comprehensive docstrings
- Include type hints
- Write tests for new functionality

### Testing
```bash
# Run enhanced CLI tests
python -m pytest tests/test_enhanced_features.py

# Test specific components
python -m pytest tests/test_hooks.py
python -m pytest tests/test_mcp.py
python -m pytest tests/test_search.py
python -m pytest tests/test_plugins.py
```

## 📊 Metrics & Monitoring

### Hook Performance
```bash
python enhanced_cli.py hooks stats
```

### Codebase Statistics
```bash
python enhanced_cli.py search stats
```

### MCP Server Health
```bash
python enhanced_cli.py mcp servers
```

### Plugin Status
```bash
python enhanced_cli.py plugins list
```

## 🚀 What's Next?

### Planned Features
- **Vector Search**: Semantic code search using embeddings
- **Code Relationships**: Advanced dependency tracking
- **Multi-Repository**: Support for searching across multiple repos
- **Plugin Registry**: Online plugin repository and distribution
- **Web UI**: Browser-based interface for all features
- **Team Collaboration**: Shared configurations and plugins

### Contributing
We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

🎉 **Enjoy your enhanced Croq AI Assistant!** With hooks, MCP integration, advanced search, and plugins, you now have a truly powerful development companion that adapts to your workflow and extends with your needs.

For more help, run:
```bash
python enhanced_cli.py --help
python enhanced_cli.py interactive
```

Happy coding! 🚀
