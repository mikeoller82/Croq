"""
High-Performance CLI Interface for Croq Optimized
"""
import asyncio
import sys
import typer
from typing import Optional, List
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import json

from croq_optimized import croq
from config import settings, ModelProvider

app = typer.Typer(
    name="croq",
    help="🚀 Croq - Advanced AI Code Assistant",
    rich_markup_mode="rich"
)
console = Console()


@app.command("generate")
def generate_code(
    prompt: str = typer.Argument(..., help="Code generation prompt"),
    language: str = typer.Option("python", "--lang", "-l", help="Target programming language"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Enable streaming output"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    analyze: bool = typer.Option(True, "--analyze/--no-analyze", help="Run code analysis")
):
    """🚀 Generate code from a prompt"""
    
    console.print(f"[bold blue]Croq Code Generation[/bold blue]")
    console.print(f"Prompt: {prompt}")
    console.print(f"Language: [green]{language}[/green]")
    
    if stream:
        asyncio.run(_stream_generate(prompt, language, context, not no_cache))
    else:
        asyncio.run(_generate_complete(prompt, language, context, not no_cache, analyze, output))


@app.command("analyze")
def analyze_code(
    file_path: str = typer.Argument(..., help="Path to code file to analyze"),
    language: str = typer.Option("python", "--lang", "-l", help="Programming language"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format (table/json)")
):
    """🔍 Analyze existing code for quality and security issues"""
    
    try:
        code = open(file_path).read()
    except FileNotFoundError:
        console.print(f"[red]Error: File '{file_path}' not found[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold blue]Analyzing code: {file_path}[/bold blue]")
    
    result = asyncio.run(croq.analyze_existing_code(code, language))
    
    if output_format == "json":
        console.print(json.dumps(result, indent=2))
    else:
        _display_analysis_results(result)


@app.command("explain")
def explain_code(
    file_path: str = typer.Argument(..., help="Path to code file to explain"),
    focus: str = typer.Option("general", "--focus", "-f", help="Focus area (general/performance/security)")
):
    """📖 Get detailed explanations of code"""
    
    try:
        code = open(file_path).read()
    except FileNotFoundError:
        console.print(f"[red]Error: File '{file_path}' not found[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold blue]Explaining code: {file_path}[/bold blue]")
    
    explanation = asyncio.run(croq.explain_code(code, focus))
    
    console.print(Panel(
        explanation, 
        title=f"Code Explanation ({focus})",
        border_style="blue"
    ))


@app.command("stats")
def show_stats():
    """📊 Display performance statistics"""
    
    console.print("[bold blue]Croq Performance Statistics[/bold blue]")
    croq.display_stats()


@app.command("health")
def health_check():
    """🏥 Run comprehensive health check"""
    
    health = asyncio.run(croq.health_check())
    
    if health["overall"]:
        console.print("✅ [green]All systems healthy[/green]")
    else:
        console.print("❌ [red]Issues detected[/red]")
        
        # Show detailed issues
        for system, status in health.items():
            if system != "overall":
                if isinstance(status, dict) and not status.get("healthy", True):
                    console.print(f"  • {system}: [red]Unhealthy[/red]")


@app.command("models")
def list_models():
    """📋 List available AI models and their status"""
    
    health = asyncio.run(croq.router.health_check())
    metrics = croq.router.get_metrics_summary()
    
    table = Table(title="Available AI Models", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Avg Latency", style="yellow") 
    table.add_column("Success Rate", style="blue")
    table.add_column("Requests", style="magenta")
    
    for provider in ModelProvider:
        if provider.value in health:
            status = "✅ Healthy" if health[provider] else "❌ Offline"
            
            model_metrics = metrics.get(provider.value, {})
            latency = f"{model_metrics.get('avg_latency', 0):.3f}s"
            success_rate = f"{model_metrics.get('success_rate', 0) * 100:.1f}%"
            requests = str(model_metrics.get('total_requests', 0))
            
            table.add_row(
                provider.value.upper(),
                status,
                latency,
                success_rate, 
                requests
            )
    
    console.print(table)


@app.command("cache")
def cache_operations(
    action: str = typer.Argument(..., help="Cache action (stats/clear)"),
):
    """💾 Cache management operations"""
    
    if action == "stats":
        stats = asyncio.run(croq.cache.get_stats())
        
        table = Table(title="Cache Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Hit Rate", f"{stats.get('hit_rate', 0)}%")
        table.add_row("Total Hits", str(stats.get('total_hits', 0)))
        table.add_row("Total Misses", str(stats.get('total_misses', 0)))
        table.add_row("Memory Hits", str(stats.get('memory_hits', 0)))
        table.add_row("Disk Hits", str(stats.get('disk_hits', 0)))
        table.add_row("Redis Hits", str(stats.get('redis_hits', 0)))
        
        console.print(table)
        
    elif action == "clear":
        asyncio.run(croq.cache.clear())
        console.print("✅ [green]Cache cleared successfully[/green]")
        
    else:
        console.print(f"[red]Unknown cache action: {action}[/red]")
        console.print("Available actions: stats, clear")
        raise typer.Exit(1)


@app.command("interactive")  
def interactive_mode():
    """🎯 Start interactive coding session"""
    
    console.print("[bold green]🚀 Croq Interactive Mode[/bold green]")
    console.print("Type 'exit' to quit, 'help' for commands")
    console.print()
    
    while True:
        try:
            prompt = console.input("[blue]croq>[/blue] ")
            
            if prompt.lower() == 'exit':
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            elif prompt.lower() == 'help':
                _show_interactive_help()
            elif prompt.lower() == 'stats':
                croq.display_stats()
            elif prompt.lower().startswith('analyze '):
                file_path = prompt[8:].strip()
                if file_path:
                    try:
                        code = open(file_path).read()
                        result = asyncio.run(croq.analyze_existing_code(code))
                        _display_analysis_results(result)
                    except FileNotFoundError:
                        console.print(f"[red]File not found: {file_path}[/red]")
            else:
                # Generate code
                result = asyncio.run(croq.generate_code(prompt))
                _display_generation_result(result)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def _show_interactive_help():
    """Show interactive mode help"""
    
    help_text = """
[bold]Interactive Commands:[/bold]

• [cyan]<prompt>[/cyan] - Generate code from prompt
• [cyan]analyze <file>[/cyan] - Analyze code file
• [cyan]stats[/cyan] - Show performance statistics  
• [cyan]help[/cyan] - Show this help
• [cyan]exit[/cyan] - Exit interactive mode
    """
    
    console.print(Panel(help_text, title="Help", border_style="yellow"))


async def _stream_generate(prompt: str, language: str, context: Optional[str], use_cache: bool):
    """Stream code generation with real-time output"""
    
    console.print("🚀 [blue]Streaming response...[/blue]")
    
    partial_code = ""
    
    async for chunk in croq.generate_code(prompt, context, language, stream=True, use_cache=use_cache):
        if chunk["type"] == "chunk":
            console.print(chunk["content"], end="")
            partial_code = chunk["partial_code"]
        elif chunk["type"] == "complete":
            console.print("\n" + "="*50)
            console.print("[green]✅ Generation complete![/green]")
            _display_generation_result(chunk)
        elif chunk["type"] == "error":
            console.print(f"\n[red]❌ Error: {chunk['error']}[/red]")


async def _generate_complete(
    prompt: str, 
    language: str, 
    context: Optional[str], 
    use_cache: bool, 
    analyze: bool,
    output: Optional[str]
):
    """Complete code generation with analysis"""
    
    result = await croq.generate_code(
        prompt, 
        context=context,
        language=language, 
        use_cache=use_cache
    )
    
    _display_generation_result(result)
    
    # Save to file if requested
    if output:
        try:
            with open(output, 'w') as f:
                f.write(result["code"])
            console.print(f"✅ [green]Code saved to {output}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to save file: {e}[/red]")


def _display_generation_result(result: dict):
    """Display code generation result"""
    
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
    
    # Show analysis if available
    analysis = result.get("analysis", {})
    if analysis:
        console.print("\n[bold]Code Analysis:[/bold]")
        
        analysis_table = Table(show_header=False, box=None)
        analysis_table.add_column("Metric", style="yellow")  
        analysis_table.add_column("Score", style="green")
        
        for metric in ["quality_score", "complexity_score", "maintainability_score"]:
            if metric in analysis:
                analysis_table.add_row(
                    metric.replace("_", " ").title(),
                    f"{analysis[metric]}/10"
                )
        
        console.print(analysis_table)
    
    # Show security issues
    security = result.get("security", [])
    if security:
        console.print(f"\n[bold red]⚠️  Security Issues Found: {len(security)}[/bold red]")
        for issue in security[:3]:  # Show first 3
            console.print(f"  • {issue.get('description', 'Unknown issue')}")


def _display_analysis_results(result: dict):
    """Display code analysis results"""
    
    analysis = result.get("analysis", {})
    security = result.get("security", [])
    suggestions = result.get("suggestions", [])
    
    # Main analysis table
    table = Table(title="Code Analysis Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green") 
    table.add_column("Issues", style="yellow")
    
    metrics = [
        ("Quality", analysis.get("quality_score", 0)),
        ("Complexity", analysis.get("complexity_score", 0)), 
        ("Maintainability", analysis.get("maintainability_score", 0)),
        ("Documentation", analysis.get("documentation_score", 0)),
        ("Style", analysis.get("style_score", 0)),
        ("Performance", analysis.get("performance_score", 0))
    ]
    
    for name, score in metrics:
        issues_key = f"{name.lower()}_issues"
        issues = analysis.get(issues_key, [])
        issue_text = f"{len(issues)} issues" if issues else "None"
        
        table.add_row(name, f"{score}/10", issue_text)
    
    console.print(table)
    
    # Security issues
    if security:
        console.print(f"\n[bold red]🔒 Security Issues ({len(security)}):[/bold red]")
        for issue in security:
            severity = issue.get("severity", "medium").upper()
            color = "red" if severity == "HIGH" else "yellow" if severity == "MEDIUM" else "blue"
            console.print(f"  • [{color}]{severity}[/{color}]: {issue.get('description', 'Unknown issue')}")
    
    # Suggestions
    if suggestions:
        console.print(f"\n[bold blue]💡 Improvement Suggestions:[/bold blue]")
        for i, suggestion in enumerate(suggestions, 1):
            console.print(f"  {i}. {suggestion}")


if __name__ == "__main__":
    app()
