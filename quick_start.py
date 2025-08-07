#!/usr/bin/env python3
"""
Croq Enhanced Features Quick Start Script
Sets up and verifies the enhanced Croq environment.
"""

import os
import sys
import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
import subprocess
import json

console = Console()

class QuickStart:
    """Quick start setup for enhanced Croq features."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_completed = []
        
    async def run(self):
        """Run the complete quick start process."""
        console.print(Panel.fit(
            "[bold blue]🚀 Croq Enhanced Features Quick Start[/bold blue]",
            subtitle="Setting up your advanced AI development environment",
            border_style="bright_blue"
        ))
        
        # Check prerequisites
        if not await self.check_prerequisites():
            return False
        
        # Setup environment
        if not await self.setup_environment():
            return False
        
        # Initialize systems
        if not await self.initialize_systems():
            return False
        
        # Run verification
        if not await self.verify_setup():
            return False
        
        # Show completion message
        self.show_completion()
        return True
    
    async def check_prerequisites(self):
        """Check system prerequisites."""
        console.print("\n[yellow]🔍 Checking Prerequisites[/yellow]")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            console.print("[red]❌ Python 3.8+ required[/red]")
            return False
        console.print(f"[green]✅ Python {python_version.major}.{python_version.minor}[/green]")
        
        # Check required files
        required_files = [
            "requirements_optimized.txt",
            "enhanced_cli.py",
            "core/hooks.py",
            "core/mcp_server.py",
            "core/codebase_search.py",
            "core/plugins.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            console.print(f"[red]❌ Missing files: {', '.join(missing_files)}[/red]")
            return False
        
        console.print("[green]✅ All required files present[/green]")
        return True
    
    async def setup_environment(self):
        """Set up the development environment."""
        console.print("\n[yellow]🛠️  Setting Up Environment[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Install dependencies
            task1 = progress.add_task("Installing dependencies...", total=None)
            if await self.install_dependencies():
                self.setup_completed.append("dependencies")
                console.print("[green]✅ Dependencies installed[/green]")
            else:
                console.print("[yellow]⚠️  Some dependencies may need manual installation[/yellow]")
            progress.update(task1, completed=True)
            
            # Create configuration
            task2 = progress.add_task("Creating configuration files...", total=None)
            if await self.create_config_files():
                self.setup_completed.append("config")
                console.print("[green]✅ Configuration files created[/green]")
            progress.update(task2, completed=True)
            
            # Set up directories
            task3 = progress.add_task("Setting up directories...", total=None)
            if await self.setup_directories():
                self.setup_completed.append("directories")
                console.print("[green]✅ Directories created[/green]")
            progress.update(task3, completed=True)
        
        return True
    
    async def initialize_systems(self):
        """Initialize enhanced systems."""
        console.print("\n[yellow]⚡ Initializing Enhanced Systems[/yellow]")
        
        # Check if user wants to index codebase
        if Confirm.ask("Would you like to index your codebase for enhanced context?", default=True):
            try:
                console.print("[blue]📊 Indexing codebase...[/blue]")
                result = subprocess.run([
                    sys.executable, "enhanced_cli.py", "search", "index", "--path", "."
                ], capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode == 0:
                    console.print("[green]✅ Codebase indexed successfully[/green]")
                    self.setup_completed.append("codebase_index")
                else:
                    console.print(f"[yellow]⚠️  Indexing warning: {result.stderr}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not index codebase: {e}[/yellow]")
        
        # Initialize plugin system
        try:
            console.print("[blue]🧩 Loading example plugin...[/blue]")
            result = subprocess.run([
                sys.executable, "enhanced_cli.py", "plugins", "load", "example_plugin"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                console.print("[green]✅ Example plugin loaded[/green]")
                self.setup_completed.append("plugins")
            else:
                console.print(f"[yellow]⚠️  Plugin loading: {result.stderr}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not load plugins: {e}[/yellow]")
        
        return True
    
    async def verify_setup(self):
        """Verify the setup by running tests."""
        console.print("\n[yellow]🧪 Verifying Setup[/yellow]")
        
        if Confirm.ask("Run comprehensive test suite?", default=True):
            try:
                console.print("[blue]🔬 Running test suite...[/blue]")
                result = subprocess.run([
                    sys.executable, "test_enhanced_features.py"
                ], cwd=self.project_root)
                
                if result.returncode == 0:
                    console.print("[green]✅ Test suite completed[/green]")
                    self.setup_completed.append("tests")
                else:
                    console.print("[yellow]⚠️  Some tests may have failed (this is normal for initial setup)[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not run tests: {e}[/yellow]")
        
        return True
    
    async def install_dependencies(self):
        """Install required dependencies."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements_optimized.txt"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            return result.returncode == 0
        except Exception:
            return False
    
    async def create_config_files(self):
        """Create configuration files if they don't exist."""
        try:
            # Create .env if it doesn't exist
            env_file = self.project_root / ".env"
            if not env_file.exists() and (self.project_root / "config_example.env").exists():
                console.print("[blue]📝 Creating .env file from template...[/blue]")
                
                # Copy example to .env
                example_content = (self.project_root / "config_example.env").read_text()
                
                # Prompt for API keys
                if Confirm.ask("Would you like to configure API keys now?", default=False):
                    lines = []
                    for line in example_content.split('\n'):
                        if '=' in line and 'API_KEY' in line and not line.strip().startswith('#'):
                            key, example_value = line.split('=', 1)
                            if 'your_' in example_value.lower():
                                new_value = Prompt.ask(f"Enter {key}", default="", show_default=False)
                                if new_value:
                                    lines.append(f"{key}={new_value}")
                                else:
                                    lines.append(line)
                            else:
                                lines.append(line)
                        else:
                            lines.append(line)
                    example_content = '\n'.join(lines)
                
                env_file.write_text(example_content)
                console.print("[green]✅ .env file created[/green]")
            
            return True
        except Exception as e:
            console.print(f"[red]❌ Error creating config files: {e}[/red]")
            return False
    
    async def setup_directories(self):
        """Set up required directories."""
        try:
            directories = [
                "plugins",
                "logs",
                "cache",
                "data"
            ]
            
            for directory in directories:
                (self.project_root / directory).mkdir(exist_ok=True)
            
            return True
        except Exception:
            return False
    
    def show_completion(self):
        """Show completion message and next steps."""
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]🎉 Setup Complete![/bold green]",
            subtitle="Your enhanced Croq environment is ready",
            border_style="bright_green"
        ))
        
        # Show what was completed
        console.print("\n[bold blue]✅ Completed Setup Steps:[/bold blue]")
        completed_items = {
            "dependencies": "📦 Dependencies installed",
            "config": "⚙️  Configuration files created", 
            "directories": "📁 Directory structure set up",
            "codebase_index": "📊 Codebase indexed for enhanced context",
            "plugins": "🧩 Plugin system initialized",
            "tests": "🧪 Verification tests completed"
        }
        
        for item in self.setup_completed:
            if item in completed_items:
                console.print(f"  {completed_items[item]}")
        
        # Show next steps
        console.print("\n[bold blue]🚀 Next Steps:[/bold blue]")
        console.print("  1. [cyan]Try the enhanced CLI:[/cyan]")
        console.print("     python enhanced_cli.py --help")
        console.print("\n  2. [cyan]Start interactive mode:[/cyan]")
        console.print("     python enhanced_cli.py interactive")
        console.print("\n  3. [cyan]Generate code with context:[/cyan]")
        console.print('     python enhanced_cli.py generate "create a REST API" --codebase')
        console.print("\n  4. [cyan]Explore plugins:[/cyan]")
        console.print("     python enhanced_cli.py plugins list")
        console.print("     python enhanced_cli.py plugins call example_plugin.greet")
        console.print("\n  5. [cyan]Check system status:[/cyan]")
        console.print("     python enhanced_cli.py hooks stats")
        console.print("     python enhanced_cli.py search stats")
        
        console.print(f"\n[bold blue]📚 Documentation:[/bold blue] [link]./ENHANCED_FEATURES.md[/link]")
        console.print(f"[bold blue]🧪 Test Suite:[/bold blue] python test_enhanced_features.py")
        
        console.print("\n[dim]Happy coding with your enhanced Croq AI assistant! 🤖✨[/dim]")

async def main():
    """Main quick start function."""
    quick_start = QuickStart()
    success = await quick_start.run()
    
    if not success:
        console.print("\n[red]❌ Quick start encountered issues. Please check the documentation.[/red]")
        sys.exit(1)
    
    console.print("\n[green]🎉 Quick start completed successfully![/green]")

if __name__ == "__main__":
    asyncio.run(main())
