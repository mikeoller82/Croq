"""
Enhanced CLI Interface for Croq with Advanced Features
Includes hooks, MCP servers, codebase search, and plugin management
"""
import asyncio
import sys
import typer
from typing import Optional, List, Dict, Any
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Confirm, Prompt
from rich.tree import Tree
import json
import yaml

# Import the enhanced core modules
from core.hooks import hook_manager, HookType, HookContext
from core.mcp_server import mcp_manager
from core.codebase_search import codebase_index, context_extractor
from core.plugins import plugin_manager, initialize_builtin_plugins, create_plugin_template

# Import existing modules
from croq_optimized import croq
from config import settings, ModelProvider

app = typer.Typer(
    name="croq",
    help="🚀 Croq - Enhanced AI Code Assistant with Hooks, MCP, and Plugin Support",
    rich_markup_mode="rich"
)

console = Console()

# Subcommands for new features
hooks_app = typer.Typer(name="hooks", help="🪝 Hook system management")
mcp_app = typer.Typer(name="mcp", help="🔌 MCP server integration")
search_app = typer.Typer(name="search", help="🔍 Codebase search and indexing")
plugin_app = typer.Typer(name="plugins", help="🧩 Plugin management")

app.add_typer(hooks_app)
app.add_typer(mcp_app)
app.add_typer(search_app)
app.add_typer(plugin_app)

# Initialize systems
@app.callback()
def initialize():
    """Initialize the enhanced Croq system"""
    asyncio.run(_initialize_systems())

async def _initialize_systems():
    """Initialize hooks, MCP, and plugins"""
    try:
        # Initialize built-in plugins
        await initialize_builtin_plugins()
        
        # Connect to MCP servers
        await mcp_manager.connect_all()
        
        console.print("✅ [green]Enhanced Croq systems initialized[/green]", err=True)
    except Exception as e:
        console.print(f"⚠️ [yellow]Warning: Failed to initialize some systems: {e}[/yellow]", err=True)

# Original commands (enhanced)
@app.command("generate")
def generate_code(
    prompt: str = typer.Argument(..., help="Code generation prompt"),
    language: str = typer.Option("python", "--lang", "-l", help="Target programming language"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Enable streaming output"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    analyze: bool = typer.Option(True, "--analyze/--no-analyze", help="Run code analysis"),
    use_codebase: bool = typer.Option(True, "--codebase/--no-codebase", help="Use codebase context")
):
    """🚀 Generate code from a prompt with enhanced context"""
    
    console.print(f"[bold blue]Enhanced Croq Code Generation[/bold blue]")
    console.print(f"Prompt: {prompt}")
    console.print(f"Language: [green]{language}[/green]")
    
    if use_codebase:
        console.print("🔍 [cyan]Searching codebase for context...[/cyan]")
    
    if stream:
        asyncio.run(_enhanced_stream_generate(prompt, language, context, not no_cache, use_codebase))
    else:
        asyncio.run(_enhanced_generate_complete(prompt, language, context, not no_cache, analyze, output, use_codebase))

async def _enhanced_stream_generate(prompt: str, language: str, context: Optional[str], use_cache: bool, use_codebase: bool):
    """Enhanced streaming generation with hooks and context"""
    
    # Create hook context
    hook_context = HookContext(
        hook_type=HookType.PRE_GENERATION,
        data={
            "prompt": prompt,
            "language": language,
            "context": context,
            "use_cache": use_cache,
            "use_codebase": use_codebase,
            "prompt_length": len(prompt)
        }
    )
    
    # Execute pre-generation hooks
    await hook_manager.execute_hooks(HookType.PRE_GENERATION, hook_context)
    
    # Get enhanced context
    enhanced_context = context or ""
    if hook_context.get("code_context"):
        enhanced_context += "\n\n" + hook_context.get("code_context")
    
    console.print("🚀 [blue]Streaming response...[/blue]")
    
    partial_code = ""
    
    async for chunk in croq.generate_code(prompt, enhanced_context, language, stream=True, use_cache=use_cache):
        if chunk["type"] == "chunk":
            console.print(chunk["content"], end="")
            partial_code = chunk["partial_code"]
        elif chunk["type"] == "complete":
            console.print("\n" + "="*50)
            console.print("[green]✅ Generation complete![/green]")
            
            # Execute post-generation hooks
            post_context = HookContext(
                hook_type=HookType.POST_GENERATION,
                data={"result": chunk, "prompt": prompt}
            )
            await hook_manager.execute_hooks(HookType.POST_GENERATION, post_context)
            
            _display_generation_result(chunk)
        elif chunk["type"] == "error":
            console.print(f"\n[red]❌ Error: {chunk['error']}[/red]")

async def _enhanced_generate_complete(
    prompt: str, 
    language: str, 
    context: Optional[str], 
    use_cache: bool, 
    analyze: bool,
    output: Optional[str],
    use_codebase: bool
):
    """Enhanced complete generation with all features"""
    
    # Create hook context
    hook_context = HookContext(
        hook_type=HookType.PRE_GENERATION,
        data={
            "prompt": prompt,
            "language": language,
            "context": context,
            "use_cache": use_cache,
            "use_codebase": use_codebase,
            "prompt_length": len(prompt)
        }
    )
    
    # Execute pre-generation hooks
    await hook_manager.execute_hooks(HookType.PRE_GENERATION, hook_context)
    
    # Get enhanced context
    enhanced_context = context or ""
    if hook_context.get("code_context"):
        enhanced_context += "\n\n" + hook_context.get("code_context")
        console.print("✨ [cyan]Enhanced with codebase context[/cyan]")
    
    result = await croq.generate_code(
        prompt, 
        context=enhanced_context,
        language=language, 
        use_cache=use_cache
    )
    
    # Execute post-generation hooks
    post_context = HookContext(
        hook_type=HookType.POST_GENERATION,
        data={"result": result, "prompt": prompt}
    )
    await hook_manager.execute_hooks(HookType.POST_GENERATION, post_context)
    
    _display_generation_result(result)
    
    # Save to file if requested
    if output:
        try:
            with open(output, 'w') as f:
                f.write(result["code"])
            console.print(f"✅ [green]Code saved to {output}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to save file: {e}[/red]")

# Hook management commands
@hooks_app.command("list")
def list_hooks():
    """📋 List all registered hooks"""
    hooks = hook_manager.list_hooks()
    
    for hook_type, hook_list in hooks.items():
        if hook_list:
            console.print(f"\n[bold cyan]{hook_type.value.replace('_', ' ').title()} Hooks:[/bold cyan]")
            
            table = Table(show_header=True)
            table.add_column("Name", style="green")
            table.add_column("Priority", style="yellow")
            table.add_column("Enabled", style="blue")
            table.add_column("Description", style="white")
            
            for hook in hook_list:
                enabled = "✅" if hook.enabled else "❌"
                table.add_row(
                    hook.name,
                    str(hook.priority.value),
                    enabled,
                    hook.description or "No description"
                )
            
            console.print(table)

@hooks_app.command("stats")
def hook_stats():
    """📊 Show hook execution statistics"""
    stats = hook_manager.get_hook_stats()
    
    if not stats:
        console.print("[yellow]No hook statistics available[/yellow]")
        return
    
    table = Table(title="Hook Execution Statistics", show_header=True)
    table.add_column("Hook Name", style="cyan")
    table.add_column("Executions", style="green")
    table.add_column("Errors", style="red")
    table.add_column("Success Rate", style="blue")
    
    for hook_name, hook_stats in stats.items():
        executed = hook_stats["executed"]
        errors = hook_stats["errors"]
        success_rate = ((executed - errors) / executed * 100) if executed > 0 else 0
        
        table.add_row(
            hook_name,
            str(executed),
            str(errors),
            f"{success_rate:.1f}%"
        )
    
    console.print(table)

@hooks_app.command("enable")
def enable_hook(hook_name: str = typer.Argument(..., help="Hook name to enable")):
    """✅ Enable a hook"""
    if hook_manager.enable_hook(hook_name):
        console.print(f"✅ [green]Hook '{hook_name}' enabled[/green]")
    else:
        console.print(f"❌ [red]Hook '{hook_name}' not found[/red]")

@hooks_app.command("disable")
def disable_hook(hook_name: str = typer.Argument(..., help="Hook name to disable")):
    """❌ Disable a hook"""
    if hook_manager.disable_hook(hook_name):
        console.print(f"❌ [yellow]Hook '{hook_name}' disabled[/yellow]")
    else:
        console.print(f"❌ [red]Hook '{hook_name}' not found[/red]")

# MCP server commands
@mcp_app.command("connect")
def connect_servers():
    """🔌 Connect to all configured MCP servers"""
    console.print("🔌 [cyan]Connecting to MCP servers...[/cyan]")
    
    results = asyncio.run(mcp_manager.connect_all())
    
    table = Table(title="MCP Server Connection Results", show_header=True)
    table.add_column("Server", style="cyan")
    table.add_column("Status", style="green")
    
    for server_name, success in results.items():
        status = "✅ Connected" if success else "❌ Failed"
        table.add_row(server_name, status)
    
    console.print(table)

@mcp_app.command("servers")
def list_servers():
    """📋 List all configured MCP servers"""
    servers = list(mcp_manager.servers.keys())
    
    if not servers:
        console.print("[yellow]No MCP servers configured[/yellow]")
        return
    
    table = Table(title="MCP Servers", show_header=True)
    table.add_column("Server Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Tools", style="blue")
    table.add_column("Resources", style="yellow")
    
    for server_name in servers:
        client = mcp_manager.servers[server_name]
        status = "✅ Connected" if client.connected else "❌ Disconnected"
        tools_count = len(client.tools)
        resources_count = len(client.resources)
        
        table.add_row(
            server_name,
            status,
            str(tools_count),
            str(resources_count)
        )
    
    console.print(table)

@mcp_app.command("tools")
def list_tools(server: Optional[str] = typer.Option(None, "--server", "-s", help="Specific server name")):
    """🛠️ List available MCP tools"""
    all_tools = mcp_manager.get_all_tools()
    
    if server and server not in all_tools:
        console.print(f"[red]Server '{server}' not found or not connected[/red]")
        return
    
    servers_to_show = {server: all_tools[server]} if server else all_tools
    
    for server_name, tools in servers_to_show.items():
        if tools:
            console.print(f"\n[bold cyan]{server_name} Tools:[/bold cyan]")
            
            table = Table(show_header=True)
            table.add_column("Tool Name", style="green")
            table.add_column("Description", style="white")
            table.add_column("Tags", style="yellow")
            
            for tool in tools.values():
                tags = ", ".join(tool.tags) if tool.tags else "None"
                table.add_row(tool.name, tool.description, tags)
            
            console.print(table)

@mcp_app.command("call")
def call_tool(
    tool_name: str = typer.Argument(..., help="Tool name to call"),
    arguments: str = typer.Option("{}", "--args", "-a", help="JSON arguments for the tool"),
    server: Optional[str] = typer.Option(None, "--server", "-s", help="Specific server name")
):
    """⚡ Call an MCP tool"""
    try:
        args_dict = json.loads(arguments) if arguments != "{}" else {}
    except json.JSONDecodeError:
        console.print(f"[red]Invalid JSON arguments: {arguments}[/red]")
        return
    
    console.print(f"🛠️ [cyan]Calling tool '{tool_name}'...[/cyan]")
    
    result = asyncio.run(_call_mcp_tool(tool_name, args_dict, server))
    
    if result is not None:
        console.print("[bold green]Tool Result:[/bold green]")
        if isinstance(result, (dict, list)):
            console.print(json.dumps(result, indent=2))
        else:
            console.print(str(result))
    else:
        console.print("[red]Tool call failed or returned no result[/red]")

async def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any], server: Optional[str] = None):
    """Call MCP tool with error handling"""
    try:
        if server:
            return await mcp_manager.call_tool(server, tool_name, arguments)
        else:
            return await mcp_manager.find_and_call_tool(tool_name, arguments)
    except Exception as e:
        console.print(f"[red]Error calling tool: {e}[/red]")
        return None

# Codebase search commands
@search_app.command("index")
def index_codebase(
    path: str = typer.Option(".", "--path", "-p", help="Path to index"),
    patterns: Optional[List[str]] = typer.Option(None, "--ignore", help="Ignore patterns")
):
    """📚 Index the codebase for search"""
    console.print(f"📚 [cyan]Indexing codebase at: {path}[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Indexing files...", total=None)
        
        result = asyncio.run(codebase_index.index_directory(Path(path), patterns))
        
        progress.update(task, completed=100)
    
    console.print(f"✅ [green]Indexed {result['indexed_files']} files[/green]")
    console.print(f"⚠️ [yellow]Failed: {result['failed_files']} files[/yellow]")
    console.print(f"⏱️ Duration: {result['duration_seconds']:.2f}s")
    
    # Show stats
    stats = result['stats']
    console.print(f"\n[bold]Codebase Statistics:[/bold]")
    console.print(f"📁 Total files: {stats.total_files}")
    console.print(f"📝 Total lines: {stats.total_lines}")
    console.print(f"💻 Languages: {', '.join(stats.languages.keys())}")

@search_app.command("search")
def search_code(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(20, "--limit", "-l", help="Maximum results"),
    file_pattern: Optional[str] = typer.Option(None, "--file", "-f", help="File pattern filter")
):
    """🔍 Search the indexed codebase"""
    console.print(f"🔍 [cyan]Searching for: {query}[/cyan]")
    
    results = codebase_index.search(query, max_results, file_pattern)
    
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    console.print(f"\n[bold green]Found {len(results)} results:[/bold green]")
    
    for i, result in enumerate(results, 1):
        console.print(f"\n[bold]{i}. {result.file_path}:{result.line_number}[/bold] (score: {result.relevance_score:.2f})")
        
        # Show context
        if result.context_before:
            for line in result.context_before[-2:]:  # Last 2 lines
                console.print(f"     {line}", style="dim")
        
        console.print(f"  >> {result.line_content}", style="bold yellow")
        
        if result.context_after:
            for line in result.context_after[:2]:  # First 2 lines  
                console.print(f"     {line}", style="dim")
        
        if result.symbol:
            console.print(f"     [cyan]Symbol: {result.symbol.type} `{result.symbol.name}`[/cyan]")

@search_app.command("symbols")
def search_symbols(
    query: str = typer.Argument(..., help="Symbol search query"),
    symbol_type: Optional[str] = typer.Option(None, "--type", "-t", help="Symbol type filter")
):
    """🎯 Search for code symbols (functions, classes, etc.)"""
    console.print(f"🎯 [cyan]Searching symbols for: {query}[/cyan]")
    
    symbols = codebase_index.search_symbols(query, symbol_type)
    
    if not symbols:
        console.print("[yellow]No symbols found[/yellow]")
        return
    
    table = Table(title=f"Found {len(symbols)} symbols", show_header=True)
    table.add_column("Name", style="green")
    table.add_column("Type", style="blue") 
    table.add_column("File", style="cyan")
    table.add_column("Line", style="yellow")
    table.add_column("Complexity", style="red")
    
    for symbol in symbols:
        table.add_row(
            symbol.name,
            symbol.type,
            str(symbol.file_path),
            str(symbol.line_number),
            str(symbol.complexity) if symbol.complexity else "N/A"
        )
    
    console.print(table)

@search_app.command("stats")
def search_stats():
    """📊 Show codebase indexing statistics"""
    stats = codebase_index.get_codebase_stats()
    
    table = Table(title="Codebase Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Files", str(stats.total_files))
    table.add_row("Total Lines", str(stats.total_lines))
    table.add_row("Average Complexity", f"{stats.complexity_score:.2f}")
    table.add_row("Last Updated", stats.last_updated.strftime("%Y-%m-%d %H:%M:%S"))
    
    console.print(table)
    
    if stats.languages:
        console.print("\n[bold]Languages:[/bold]")
        lang_table = Table(show_header=True)
        lang_table.add_column("Language", style="blue")
        lang_table.add_column("Files", style="green")
        
        for lang, count in sorted(stats.languages.items()):
            lang_table.add_row(lang, str(count))
        
        console.print(lang_table)

# Plugin management commands
@plugin_app.command("list")
def list_plugins():
    """📋 List all plugins"""
    plugins = plugin_manager.list_plugins()
    
    if not plugins:
        console.print("[yellow]No plugins loaded[/yellow]")
        return
    
    table = Table(title="Loaded Plugins", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Status", style="blue")
    table.add_column("Description", style="white")
    
    for name, plugin_data in plugins.items():
        info = plugin_data["info"]
        status = "✅ Enabled" if plugin_data["enabled"] else "❌ Disabled"
        
        table.add_row(
            name,
            info.get("version", "unknown"),
            status,
            info.get("description", "No description")
        )
    
    console.print(table)

@plugin_app.command("load")
def load_plugin(plugin_name: str = typer.Argument(..., help="Plugin name to load")):
    """⬇️ Load a plugin"""
    console.print(f"⬇️ [cyan]Loading plugin: {plugin_name}[/cyan]")
    
    success = asyncio.run(plugin_manager.load_plugin(plugin_name))
    
    if success:
        console.print(f"✅ [green]Plugin '{plugin_name}' loaded successfully[/green]")
    else:
        console.print(f"❌ [red]Failed to load plugin '{plugin_name}'[/red]")

@plugin_app.command("unload")
def unload_plugin(plugin_name: str = typer.Argument(..., help="Plugin name to unload")):
    """⬆️ Unload a plugin"""
    console.print(f"⬆️ [cyan]Unloading plugin: {plugin_name}[/cyan]")
    
    success = asyncio.run(plugin_manager.unload_plugin(plugin_name))
    
    if success:
        console.print(f"✅ [green]Plugin '{plugin_name}' unloaded successfully[/green]")
    else:
        console.print(f"❌ [red]Failed to unload plugin '{plugin_name}'[/red]")

@plugin_app.command("reload")
def reload_plugin(plugin_name: str = typer.Argument(..., help="Plugin name to reload")):
    """🔄 Reload a plugin"""
    console.print(f"🔄 [cyan]Reloading plugin: {plugin_name}[/cyan]")
    
    success = asyncio.run(plugin_manager.reload_plugin(plugin_name))
    
    if success:
        console.print(f"✅ [green]Plugin '{plugin_name}' reloaded successfully[/green]")
    else:
        console.print(f"❌ [red]Failed to reload plugin '{plugin_name}'[/red]")

@plugin_app.command("commands")
def plugin_commands():
    """📋 List all plugin commands"""
    commands = plugin_manager.get_all_commands()
    
    if not commands:
        console.print("[yellow]No plugin commands available[/yellow]")
        return
    
    table = Table(title="Plugin Commands", show_header=True)
    table.add_column("Command", style="cyan")
    table.add_column("Plugin", style="green")
    table.add_column("Function", style="blue")
    
    for cmd_name, (plugin_name, cmd_func) in commands.items():
        table.add_row(
            cmd_name,
            plugin_name,
            cmd_func.__name__
        )
    
    console.print(table)

@plugin_app.command("call")
def call_plugin_command(
    command: str = typer.Argument(..., help="Plugin command to call (format: plugin.command)"),
    arguments: str = typer.Option("[]", "--args", "-a", help="JSON arguments for the command")
):
    """⚡ Call a plugin command"""
    try:
        if "." not in command:
            console.print(f"[red]Invalid command format. Use 'plugin.command'[/red]")
            return
        
        plugin_name, cmd_name = command.split(".", 1)
        args_list = json.loads(arguments) if arguments != "[]" else []
        
        console.print(f"⚡ [cyan]Calling {plugin_name}.{cmd_name}...[/cyan]")
        
        result = asyncio.run(plugin_manager.call_plugin_command(plugin_name, cmd_name, *args_list))
        
        if result is not None:
            console.print("[bold green]Command Result:[/bold green]")
            if isinstance(result, (dict, list)):
                console.print(json.dumps(result, indent=2))
            else:
                console.print(str(result))
        
    except json.JSONDecodeError:
        console.print(f"[red]Invalid JSON arguments: {arguments}[/red]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Command execution error: {e}[/red]")

@plugin_app.command("create")
def create_plugin(plugin_name: str = typer.Argument(..., help="Name for the new plugin")):
    """🆕 Create a new plugin template"""
    console.print(f"🆕 [cyan]Creating plugin template: {plugin_name}[/cyan]")
    
    success = create_plugin_template(plugin_name)
    
    if success:
        console.print(f"✅ [green]Plugin template created successfully[/green]")
        console.print(f"📁 Location: {plugin_manager.plugins_dir / plugin_name}")
        console.print(f"📝 Edit the __init__.py file to implement your plugin")
    else:
        console.print(f"❌ [red]Failed to create plugin template[/red]")

# Enhanced interactive mode
@app.command("interactive")  
def enhanced_interactive():
    """🎯 Start enhanced interactive session with all features"""
    
    console.print("[bold green]🚀 Enhanced Croq Interactive Mode[/bold green]")
    console.print("New features: hooks, MCP tools, codebase search, plugins")
    console.print("Type 'help' for commands, 'exit' to quit")
    console.print()
    
    # Auto-index current directory
    if Confirm.ask("Index current directory for enhanced context?", default=True):
        console.print("📚 [cyan]Indexing current directory...[/cyan]")
        try:
            result = asyncio.run(codebase_index.index_directory(Path(".")))
            console.print(f"✅ Indexed {result['indexed_files']} files")
        except Exception as e:
            console.print(f"⚠️ [yellow]Indexing failed: {e}[/yellow]")
    
    console.print()
    
    while True:
        try:
            prompt = console.input("[blue]croq\u003e[/blue] ")
            
            if prompt.lower() == 'exit':
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            elif prompt.lower() == 'help':
                _show_enhanced_help()
            elif prompt.lower().startswith('hook '):
                _handle_hook_command(prompt[5:])
            elif prompt.lower().startswith('mcp '):
                _handle_mcp_command(prompt[4:])
            elif prompt.lower().startswith('search '):
                _handle_search_command(prompt[7:])
            elif prompt.lower().startswith('plugin '):
                _handle_plugin_command(prompt[7:])
            else:
                # Enhanced code generation with all context
                result = asyncio.run(_enhanced_interactive_generate(prompt))
                _display_generation_result(result)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def _show_enhanced_help():
    """Show enhanced interactive help"""
    
    help_text = """
[bold]Enhanced Interactive Commands:[/bold]

[cyan]Code Generation:[/cyan]
• [green]\u003cprompt\u003e[/green] - Generate code (auto-enhanced with codebase context)

[cyan]Hook System:[/cyan]
• [green]hook list[/green] - List all hooks
• [green]hook stats[/green] - Show hook statistics
• [green]hook enable \u003cname\u003e[/green] - Enable a hook
• [green]hook disable \u003cname\u003e[/green] - Disable a hook

[cyan]MCP Integration:[/cyan]
• [green]mcp servers[/green] - List MCP servers
• [green]mcp tools[/green] - List available tools
• [green]mcp call \u003ctool\u003e [\u003cargs\u003e][/green] - Call MCP tool

[cyan]Codebase Search:[/cyan]
• [green]search \u003cquery\u003e[/green] - Search code
• [green]search symbols \u003cquery\u003e[/green] - Search symbols
• [green]search stats[/green] - Show search statistics

[cyan]Plugin System:[/cyan]
• [green]plugin list[/green] - List plugins
• [green]plugin commands[/green] - List plugin commands
• [green]plugin call \u003cplugin.command\u003e[/green] - Call plugin command

[cyan]General:[/cyan]
• [green]help[/green] - Show this help
• [green]exit[/green] - Exit interactive mode
    """
    
    console.print(Panel(help_text, title="Enhanced Help", border_style="yellow"))

def _handle_hook_command(cmd: str):
    """Handle hook commands in interactive mode"""
    parts = cmd.strip().split()
    if not parts:
        console.print("[red]Usage: hook \u003clist|stats|enable|disable\u003e[/red]")
        return
    
    if parts[0] == "list":
        list_hooks()
    elif parts[0] == "stats":
        hook_stats()
    elif parts[0] == "enable" and len(parts) > 1:
        enable_hook(parts[1])
    elif parts[0] == "disable" and len(parts) > 1:
        disable_hook(parts[1])
    else:
        console.print("[red]Invalid hook command[/red]")

def _handle_mcp_command(cmd: str):
    """Handle MCP commands in interactive mode"""
    parts = cmd.strip().split()
    if not parts:
        console.print("[red]Usage: mcp \u003cservers|tools|call\u003e[/red]")
        return
    
    if parts[0] == "servers":
        list_servers()
    elif parts[0] == "tools":
        list_tools()
    elif parts[0] == "call" and len(parts) > 1:
        tool_name = parts[1]
        args = "{}"
        if len(parts) > 2:
            args = " ".join(parts[2:])
        call_tool(tool_name, args)
    else:
        console.print("[red]Invalid MCP command[/red]")

def _handle_search_command(cmd: str):
    """Handle search commands in interactive mode"""
    parts = cmd.strip().split()
    if not parts:
        console.print("[red]Usage: search \u003cquery|symbols|stats\u003e[/red]")
        return
    
    if parts[0] == "stats":
        search_stats()
    elif parts[0] == "symbols" and len(parts) > 1:
        search_symbols(" ".join(parts[1:]))
    else:
        search_code(" ".join(parts))

def _handle_plugin_command(cmd: str):
    """Handle plugin commands in interactive mode"""
    parts = cmd.strip().split()
    if not parts:
        console.print("[red]Usage: plugin \u003clist|commands|call\u003e[/red]")
        return
    
    if parts[0] == "list":
        list_plugins()
    elif parts[0] == "commands":
        plugin_commands()
    elif parts[0] == "call" and len(parts) > 1:
        call_plugin_command(parts[1])
    else:
        console.print("[red]Invalid plugin command[/red]")

async def _enhanced_interactive_generate(prompt: str):
    """Enhanced interactive generation with all context"""
    # Create hook context
    hook_context = HookContext(
        hook_type=HookType.PRE_GENERATION,
        data={
            "prompt": prompt,
            "language": "python",  # Default
            "interactive": True,
            "prompt_length": len(prompt)
        }
    )
    
    # Execute pre-generation hooks
    await hook_manager.execute_hooks(HookType.PRE_GENERATION, hook_context)
    
    # Get enhanced context
    enhanced_context = ""
    if hook_context.get("code_context"):
        enhanced_context = hook_context.get("code_context")
        console.print("✨ [dim cyan]Enhanced with codebase context[/dim cyan]")
    
    result = await croq.generate_code(prompt, context=enhanced_context)
    
    # Execute post-generation hooks
    post_context = HookContext(
        hook_type=HookType.POST_GENERATION,
        data={"result": result, "prompt": prompt}
    )
    await hook_manager.execute_hooks(HookType.POST_GENERATION, post_context)
    
    return result

# Utility functions from original CLI
def _display_generation_result(result: dict):
    """Display code generation result (from original CLI)"""
    
    metadata = result.get("metadata", {})
    
    # Show generated code
    console.print(Panel(
        Syntax(result["code"], result.get("language", "python"), theme="monokai"),
        title=f"Generated {result.get('language', 'Python')} Code",
        border_style="green" if metadata.get("success", False) else "red"
    ))
    
    # Show metadata
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="white")
    
    if metadata.get("cached"):
        info_table.add_row("Source", "✅ Cache")
    else:
        info_table.add_row("Model", metadata.get("model_used", "unknown"))
        
    info_table.add_row("Response Time", f"{metadata.get('response_time', 0):.3f}s")
    
    if metadata.get("total_tokens"):
        info_table.add_row("Tokens", str(metadata["total_tokens"]))
        
    if metadata.get("cost"):
        info_table.add_row("Cost", f"${metadata['cost']:.6f}")
    
    console.print(info_table)

if __name__ == "__main__":
    app()
