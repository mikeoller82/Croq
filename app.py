from typing import List, Optional, Dict, Any, Tuple, Union, Callable
import ast
import sys
import time
import os
import re
import anthropic
import warnings
import asyncio
import hashlib
from git import Repo
from git.exc import GitError, GitCommandError
import requests
from packaging.requirements import Requirement
from functools import lru_cache
import json
from dotenv import load_dotenv
import logging
import itertools
from pathlib import Path
import traceback
import tempfile
import subprocess
from contextlib import contextmanager
import importlib.util
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
import openai
from rich.console import Console
from rich.syntax import Syntax
from rich.logging import RichHandler

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("AICodeAssistantLogger")
console = Console()

load_dotenv()
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Environment Setup
class CodeEnvironment:
    """Manages sandbox environment for code execution"""
    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[SyntaxError]]:
        """Validate code syntax with detailed error reporting"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as se:
            return False, se

    @contextmanager
    def sandbox(self):
        """Creates an isolated environment for code execution"""
        original_paths = sys.path.copy()
        original_modules = set(sys.modules.keys())

        try:
            sys.path.insert(0, str(self.temp_dir))
            yield self.temp_dir
        finally:
            sys.path = original_paths
            # Remove any newly imported modules
            for mod in set(sys.modules.keys()) - original_modules:
                sys.modules.pop(mod, None)

    def test_imports(self, code: str) -> Dict[str, bool]:
        """Tests if all imports in the code are available"""
        try:
            tree = ast.parse(code)
            imports = [
                node.names[0].name
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]

            results = {}
            for imp in imports:
                try:
                    importlib.import_module(imp)
                    results[imp] = True
                except ImportError:
                    results[imp] = False
            return results
        except SyntaxError:
            return {"parsing_error": False}
        

# Add ModelProvider to select between available models
class ModelProvider(Enum):
    GROQ = "groq"
    OLLAMA = "ollama"
    CLAUDE = "claude"

# Data Models
class CodeQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"

class CodeAnalysis(BaseModel):
    quality: CodeQuality
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    complexity_score: float = Field(ge=0, le=10)
    security_issues: List[dict] = Field(default_factory=list)
    documentation_coverage: float = Field(ge=0, le=100)
    test_coverage: Optional[float] = Field(None, ge=0, le=100)

    @classmethod
    def validate_complexity_score(cls, value: float) -> float:
        if value > 10:
            return 10
        elif value < 0:
            return 0
        return value

class CodeTestResult(BaseModel):
    passed: bool
    error_message: Optional[str] = None
    execution_time: float
    memory_usage: Optional[float] = None

    @classmethod
    def validate_execution_time(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Execution time cannot be negative")
        return value

    @classmethod
    def validate_memory_usage(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("Memory usage cannot be negative")
        return value

class GeneratedCode(BaseModel):
    code: str
    language: str = "python"
    description: str
    test_cases: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    version_hash: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    documentation: Optional[str] = None

    @classmethod
    def get_fallback_code(cls) -> 'GeneratedCode':
        """Return a safe fallback when generation fails"""
        return cls(
            code="# Code generation failed\n# Default safe code\npass",
            description="Fallback code",
            language="python",
            metrics={"error": "Generation failed"}
        )

    @classmethod
    def _validate_code(self, code: str):
        """Comprehensive code validation with improved string handling"""
        # First, scan for unterminated strings
        lines = code.splitlines()
        open_quotes = ""
        
        for i, line in enumerate(lines):
            for char in line:
                if char in ['"', "'"]:
                    if not open_quotes:
                        open_quotes = char
                    elif char == open_quotes:
                        open_quotes = ""
            
            # If we have open quotes at end of line, try to fix
            if open_quotes:
                lines[i] += open_quotes
                open_quotes = ""
        
        # Reconstructed code with fixed quotes
        fixed_code = "\n".join(lines)
        
        # Syntax check
        try:
            ast.parse(fixed_code)
            return fixed_code  # Return the fixed code
        except SyntaxError as e:
            raise ValueError(f"Invalid syntax: {str(e)}")

# Add Conversation class for memory
class Conversation(BaseModel):
    """Stores conversation history between user and assistant"""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    generated_codes: List[GeneratedCode] = Field(default_factory=list)
    model_provider: ModelProvider = Field(default=ModelProvider.CLAUDE)

    # Add this configuration
    model_config = {
        "protected_namespaces": ()
    }

    def add_user_message(self, content: str):
        """Add a user message to the conversation history"""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, code: Optional[GeneratedCode] = None, thinking: Optional[str] = None):
        """Add an assistant message to the conversation history with optional code and thinking"""
        message = {"role": "assistant", "content": content}
        if thinking:
            message["thinking"] = thinking
        self.messages.append(message)
        if code:
            self.generated_codes.append(code)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history"""
        return self.messages

    def get_latest_code(self) -> Optional[GeneratedCode]:
        """Get the most recently generated code"""
        return self.generated_codes[-1] if self.generated_codes else None

    def clear(self):
        """Clear the conversation history"""
        self.messages = []
        self.generated_codes = []

class MetaPrompt(BaseModel):
    context: str
    objectives: List[str]
    constraints: List[str]
    examples: Optional[List[str]] = None

    def to_string(self) -> str:
        return f"""
Context: {self.context}

Objectives:
{chr(10).join(f'- {obj}' for obj in self.objectives)}

Constraints:
{chr(10).join(f'- {con}' for con in self.constraints)}

{f"Examples:{chr(10)}{chr(10).join(self.examples)}" if self.examples else ""}
"""

class CodeVersioning:
    """Git-based version control with empty repo handling"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        # Create the directory if it doesn't exist
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self._init_repo()
        
    def _create_initial_commit(self):
        """Create initial commit with README file"""
        try:
            # Create and add README file
            readme_path = self.repo_path / "README.md"
            readme_path.write_text("# Generated Code Repository\nThis repository contains generated code versions.")
            
            # Configure git user if not set
            if not self.repo.config_reader().has_section("user"):
                self.repo.config_writer().set_value("user", "name", "AI Code Assistant").release()
                self.repo.config_writer().set_value("user", "email", "mikeoller82@gmail.com").release()
            
            # Add and commit README
            self.repo.index.add([str(readme_path)])
            self.repo.index.commit("Initial commit")
            
            # Create main branch if it doesn't exist
            if 'main' not in self.repo.heads:
                self.repo.create_head('main')
                self.repo.heads.main.checkout()
                
        except GitError as e:
            logger.error(f"Failed to create initial commit: {str(e)}")
            raise
    
    def _init_repo(self):
        """Initialize repository with proper branch setup"""
        try:
            self.repo = Repo(self.repo_path)
            
            # Create initial commit if needed
            if not self.repo.heads:
                self._create_initial_commit()
                
            # Ensure main branch exists
            if 'main' not in self.repo.heads:
                self.repo.git.checkout('-b', 'main')
                
        except GitError:
            self.repo = Repo.init(self.repo_path)
            self._create_initial_commit()
            
    def commit_version(self, code: GeneratedCode, message: str = "Auto-commit"):
        """Safe commit with branch handling"""
        try:
            # Create directory and file
            code_file = self.repo_path / "generated_code.py"
            # Ensure repo_path exists
            self.repo_path.mkdir(parents=True, exist_ok=True)
            
            # Write the file (create parent directories if needed)
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code.code)
            
            # Verify file exists before git operations
            if not code_file.exists():
                raise FileNotFoundError(f"Failed to create file: {code_file}")
            
            # Git operations
            self.repo.index.add([str(code_file)])
            commit = self.repo.index.commit(message)
            return commit.hexsha
            
        except Exception as e:
            logger.error(f"Version control error: {str(e)}")
            return None

class DependencyManager:
    """Advanced dependency resolution and vulnerability checking"""
    
    def __init__(self):
        self.requirements = set()
        self.vulnerability_db = self._load_vulnerability_db()
        
    def _load_vulnerability_db(self):
        try:
            response = requests.get("https://pypi.org/safety-db/")
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to load vulnerability DB: {str(e)}")
            return {}
            
    def add_requirement(self, package: str):
        """Add a package requirement with version validation"""
        try:
            req = Requirement(package)
            self.requirements.add(str(req))
        except ValueError:
            logger.error(f"Invalid requirement: {package}")
            
    def check_vulnerabilities(self):
        """Check requirements against known vulnerabilities"""
        vulnerable = []
        for req in self.requirements:
            pkg = req.split("==")[0]
            if vulns := self.vulnerability_db.get(pkg.lower()):
                vulnerable.extend(vulns)
        return vulnerable
        
    def generate_requirements_file(self, path: Path):
        """Generate requirements.txt file"""
        with open(path / "requirements.txt", "w") as f:
            f.write("\n".join(sorted(self.requirements)))

class DocumentationGenerator:
    """Auto-generates documentation from code"""
    
    def generate_html_docs(self, code: GeneratedCode, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Generate HTML documentation using pdoc"""
        try:
            if output_dir is None:
                output_dir = Path("code_versions/docs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            code_path = output_dir / "temp_module.py"
            code_path.write_text(code.code)
            
            result = subprocess.run(
                ["pdoc", "-o", str(output_dir), str(code_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return output_dir / "temp_module.html"
            return None
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            return None

class AICodeAssistant:
    def __init__(self, model_provider: ModelProvider = ModelProvider.CLAUDE):
        self.tokens_used = 0
        self.last_reset = datetime.now()
        self.rate_limit = 90000  # Tokens per minute
        self.code_history = []
        self.fix_attempts = 0
        self.max_fixes = 3
        self.environment = CodeEnvironment()
        self.user_feedback = ""
        self.version_control = CodeVersioning(Path("code_versions"))
        self.dependency_manager = DependencyManager()
        self.documentation_gen = DocumentationGenerator()
        self.cache = {}
        self.model_provider = model_provider
        self.conversation = Conversation(model_provider=model_provider)
        
        # Initialize the appropriate API clients
        self.openai_client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )
        
        # Initialize Claude client if API key is available
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            self.anthropic_client = None
            if model_provider == ModelProvider.CLAUDE:
                logger.warning("ANTHROPIC_API_KEY not found but Claude model requested. Defaulting to GROQ.")
                self.model_provider = ModelProvider.GROQ
        
        self._verify_repo_health()
        self._init_git_branch()
        
    def _init_git_branch(self):
        """Ensure main branch exists"""
        if not self.version_control.repo.heads:
            self.version_control.repo.git.checkout('-b', 'main')
        if 'main' not in self.version_control.repo.heads:
            self.version_control.repo.create_head('main')

    def create_meta_prompt(self, task_description: str) -> MetaPrompt:
        """Creates a structured meta-prompt for code generation"""
        return MetaPrompt(
            context=f"""You are a professional Python developer tasked with: {task_description}
            Generate production-quality code with proper error handling, logging, and best practices.""",
            objectives=[
                "Generate well-structured, maintainable code",
                "Include comprehensive error handling",
                "Add detailed documentation and type hints",
                "Implement logging for important operations",
                "Follow PEP 8 style guidelines"
            ],
            constraints=[
                "Code must be executable in Python 3.8+",
                "Use only standard library and specified dependencies",
                "Include proper exception handling for all operations",
                "Maintain separation of concerns",
                "Ensure code is testable and maintainable"
            ]
        )
        
    # Define tools for Claude
    def _get_claude_tools(self) -> List[Dict[str, Any]]:
        """Define tools that Claude can call"""
        return [
            {
                "name": "validate_syntax",
                "description": "Validate Python code syntax",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to validate"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "check_imports",
                "description": "Check if all imports in code are available",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code with imports to check"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "test_code_execution",
                "description": "Test if code executes without errors",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to test"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "check_security",
                "description": "Check code for security issues",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to check for security issues"
                        }
                    },
                    "required": ["code"]
                }
            }
        ]

    # Function to execute the tools
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool and return the result"""
        try:
            if tool_name == "validate_syntax":
                valid, error = self.environment.validate_syntax(tool_input["code"])
                return {"valid": valid, "error": str(error) if error else None}
                
            elif tool_name == "check_imports":
                import_results = self.environment.test_imports(tool_input["code"])
                return {"imports": import_results}
                
            elif tool_name == "test_code_execution":
                with self.environment.sandbox() as temp_dir:
                    test_file = temp_dir / "test_code.py"
                    test_file.write_text(tool_input["code"])
                    
                    try:
                        result = subprocess.run(
                            [sys.executable, str(test_file)],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        return {
                            "success": result.returncode == 0,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "return_code": result.returncode
                        }
                    except Exception as e:
                        return {"success": False, "error": str(e)}
                        
            elif tool_name == "check_security":
                security_issues = self._check_security(tool_input["code"])
                return {"security_issues": security_issues}
                
            return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            return {"error": f"Tool execution failed: {str(e)}"}

    async def _call_anthropic_api(self, messages: List[Dict[str, Any]], system_prompt: str) -> Dict[str, Any]:
        """Call Claude's API with thinking and tools"""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized. Make sure ANTHROPIC_API_KEY is set.")
            
        # Define Claude model parameters
        claude_model = "claude-3-7-sonnet-20250219"
        max_tokens = 4000
        thinking_budget = 2000
        
        # Format messages for Claude API
        claude_messages = []
        for msg in messages:
            # Handle user messages
            if msg["role"] == "user":
                claude_messages.append({
                    "role": "user",
                    "content": msg["content"]
                })
            # Handle assistant messages - might contain thinking blocks
            elif msg["role"] == "assistant":
                # If content is a list (from previous tool calls)
                if isinstance(msg["content"], list):
                    claude_messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
                else:
                    # Regular text content
                    claude_messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
        
        try:
            # Get tools
            tools = self._get_claude_tools()
            
            # Make the API call to Claude
            response = self.anthropic_client.messages.create(
                model=claude_model,
                max_tokens=max_tokens,
                system=system_prompt,
                thinking={
                    "type": "enabled", 
                    "budget_tokens": thinking_budget
                },
                tools=tools,
                messages=claude_messages
            )
            
            # Handle tool calling if needed
            if response.stop_reason == "tool_use":
                # Extract response blocks
                assistant_blocks = []
                for block in response.content:
                    if block.type in ["thinking", "redacted_thinking", "tool_use"]:
                        assistant_blocks.append(block)
                
                # Find the tool_use block and execute it
                tool_use_block = next((block for block in response.content if block.type == "tool_use"), None)
                
                if tool_use_block:
                    # Execute the requested tool
                    tool_result = self._execute_tool(tool_use_block.name, tool_use_block.input)
                    
                    # Add the assistant message with tool call to messages
                    claude_messages.append({
                        "role": "assistant",
                        "content": assistant_blocks
                    })
                    
                    # Add tool result to messages
                    claude_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": json.dumps(tool_result)
                        }]
                    })
                    
                    # Make another API call with the tool result
                    response = self.anthropic_client.messages.create(
                        model=claude_model,
                        max_tokens=max_tokens,
                        system=system_prompt,
                        thinking={
                            "type": "enabled",
                            "budget_tokens": thinking_budget
                        },
                        tools=tools,
                        messages=claude_messages
                    )
            
            # Extract thinking and content
            thinking_content = ""
            final_content = ""
            
            for block in response.content:
                if block.type == "thinking":
                    thinking_content = block.thinking
                elif block.type == "text":
                    final_content = block.text
            
            return {
                "response": final_content,
                "thinking": thinking_content
            }
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise

            
    async def _call_groq_api(self, messages: List[Dict[str, Any]], system_prompt: str) -> Dict[str, Any]:
        """Call Groq API"""
        try:
            # Add system message at the beginning
            formatted_messages = [{"role": "system", "content": system_prompt}]
            
            # Add the rest of the messages
            for msg in messages:
                if isinstance(msg["content"], str):
                    formatted_messages.append({"role": msg["role"], "content": msg["content"]})
            
            response = self.openai_client.chat.completions.create(
                model="qwen-qwq-32b",
                messages=formatted_messages,
                temperature=0.1
            )
            
            return {
                "response": response.choices[0].message.content,
                "thinking": ""  # Groq doesn't support thinking blocks
            }
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise
            
    

    def _verify_repo_health(self):
        """Ensure repository is in working state"""
        if not hasattr(self, 'version_control') or not hasattr(self.version_control, 'repo'):
            logger.error("Version control or repo not properly initialized")
            raise RuntimeError("Version control system not properly initialized")
            
        try:
            if not self.version_control.repo.heads:
                logger.info("No branches found, creating initial commit")
                self.version_control._create_initial_commit()
                
            # Verify HEAD reference
            try:
                self.version_control.repo.git.rev_parse('--verify', 'HEAD')
                logger.info("HEAD reference verified")
            except GitCommandError:
                logger.warning("No HEAD reference found, creating repair commit")
                self.version_control.repo.index.commit("Repair commit", allow_empty=True)
                
            # Ensure main branch exists
            if 'main' not in self.version_control.repo.heads:
                logger.info("Creating main branch")
                self.version_control.repo.create_head('main')
                self.version_control.repo.heads.main.checkout()
                
        except Exception as e:
            logger.error(f"Repository verification failed: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    async def _fix_specific_issues(self, code: str, error: SyntaxError) -> tuple[str, bool]:
        """Enhanced syntax error correction with better tracking and string handling"""
        fixed = False
        original = code

        # Handle unterminated string literals more robustly
        if "unterminated string literal" in str(error):
            lines = code.splitlines()
            if 0 <= error.lineno - 1 < len(lines):
                error_line = lines[error.lineno - 1]
                
                # Check for unbalanced quotes
                single_quotes = error_line.count("'") % 2
                double_quotes = error_line.count('"') % 2
                triple_single = error_line.count("'''") % 2
                triple_double = error_line.count('"""') % 2
                
                # Fix the issue based on quote type
                if triple_double:
                    lines[error.lineno - 1] = error_line + '"""'
                    fixed = True
                elif triple_single:
                    lines[error.lineno - 1] = error_line + "'''"
                    fixed = True
                elif double_quotes:
                    lines[error.lineno - 1] = error_line + '"'
                    fixed = True
                elif single_quotes:
                    lines[error.lineno - 1] = error_line + "'"
                    fixed = True
                    
                code = '\n'.join(lines)

        # Handle indentation
        if not fixed and "unexpected indent" in str(error):
            lines = code.splitlines()
            if 0 <= error.lineno - 1 < len(lines):
                error_line = lines[error.lineno - 1]
                lines[error.lineno - 1] = error_line.lstrip()
                code = '\n'.join(lines)
                fixed = True

        # Clean special characters if not already fixed
        if not fixed:
            cleaned = ''.join(char for char in code if ord(char) >= 32 or char in '\n\r\t')
            if cleaned != code:
                code = cleaned
                fixed = True

        # Check if fixing was successful by trying to parse the fixed code
        if fixed:
            try:
                ast.parse(code)
                return code, True
            except SyntaxError:
                # If still invalid but we made changes, return anyway
                return code, code != original
        
        return code, fixed and code != original

    async def _call_ollama_with_rate_limit(self, **kwargs) -> Any:
        """Rate-limited Ollama API calls"""
        current_time = datetime.now()
        time_since_reset = (current_time - self.last_reset).total_seconds()
        
        # Reset token counter if minute has passed
        if time_since_reset > 60:
            self.tokens_used = 0
            self.last_reset = current_time

        try:
            # Ollama endpoint
            url = "http://localhost:11434/api/generate"
            
            # Format the messages into a prompt
            messages = kwargs.get('messages', [])
            prompt = "\n".join(msg["content"] for msg in messages)
            
            data = {
                "model": "QwQ:latest",
                "prompt": prompt,
                "stream": False,
                "temperature": kwargs.get('temperature', 0.1)
            }
            
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise

    async def generate_code(self, task_description: str) -> GeneratedCode:
        """Generation process with improved error handling and validation"""
        # Add user request to conversation history
        self.conversation.add_user_message(task_description)
            
        max_attempts = 25  # Increased from 10 to 25 attempts
        backoff_time = 1  # Starting backoff time in seconds
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Attempt {attempt + 1} to generate code")
                
                # Read system prompt from file
                prompt_path = Path(__file__).parent / "prompt.txt"
                try:
                    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
                except FileNotFoundError:
                    logger.warning("prompt.txt not found, using default prompt")
                    system_prompt = """You are a Python code generator. DO NOT explain your thinking process.
                    Generate ONLY the Python code following these rules:
                    1. Use proper string termination
                    2. Include proper error handling
                    3. Use type hints
                    4. Add docstrings
                    5. Return ONLY the code, no explanations
                    6. Do not use markdown formatting"""
                
                # Add exponential backoff for rate limiting
                if attempt > 0:
                    wait_time = backoff_time * (2 ** (attempt - 1))  # Exponential backoff
                    wait_time = min(wait_time, 60)  # Cap at 60 seconds
                    logger.info(f"Waiting {wait_time:.1f}s before next attempt")
                    await asyncio.sleep(wait_time)
                
                # Include conversation history in messages
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.conversation.get_conversation_history())
                
                response = await self._call_ollama_with_rate_limit(
                    messages=messages,
                    temperature=0.1 + (attempt * 0.02)  # More gradual temperature increase
                )
                
                # Extract code from Ollama response
                raw_code = response.get('response', '').strip()
                
                # Try parsing first to get any syntax errors
                try:
                    ast.parse(raw_code)
                    cleaned_code, is_valid = raw_code, True
                except SyntaxError as se:
                    # Apply fixes with the actual syntax error
                    cleaned_code, is_valid = await self._fix_specific_issues(raw_code, se)
                
                if is_valid:
                    logger.info(f"Successfully generated valid code on attempt {attempt + 1}")
                    generated_code = GeneratedCode(
                        code=cleaned_code,
                        description=task_description,
                        language="python"
                    )
                    
                    # Process successful code
                    self._process_successful_code(generated_code)
                    
                    # Add the generated code to conversation history
                    self.conversation.add_assistant_message(
                        f"Generated code for: {task_description}", 
                        generated_code
                    )
                    
                    return generated_code
                else:
                    logger.warning(f"Attempt {attempt + 1} failed to produce valid code")
                    continue
                    
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    logger.error("All attempts failed, returning fallback code")
                    fallback_code = self._handle_generation_failure()
                    
                    # Add to conversation
                    self.conversation.add_assistant_message(
                        "Failed to generate valid code. Using fallback code.", 
                        fallback_code
                    )
                    
                    return fallback_code
                continue
                    
        logger.error(f"Failed to generate valid code after {max_attempts} attempts")
        fallback_code = self._handle_generation_failure()
        
        # Add to conversation
        self.conversation.add_assistant_message(
            "Failed to generate valid code after multiple attempts. Using fallback code.", 
            fallback_code
        )
        
        return fallback_code

    def _validate_code(self, code: str):
        """Comprehensive code validation"""
        # Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid syntax: {str(e)}")
            
        # Security check
        if self._check_security(code):
            raise SystemError("Security issues detected")
            
        # Import check
        missing_imports = [k for k,v in self.environment.test_imports(code).items() if not v]
        if missing_imports:
            raise ImportError(f"Missing imports: {missing_imports}")

    def _process_successful_code(self, code: GeneratedCode):
        """Post-generation workflow"""
        # Version control
        self.version_control.commit_version(code)
        
        # Documentation
        code.documentation = self.documentation_gen.generate_html_docs(code)
        
        # Architecture diagram
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code.metrics["diagram_path"] = self.generate_architecture_diagram(code)

    def _handle_generation_failure(self) -> GeneratedCode:
        """Graceful failure handling"""
        fallback = GeneratedCode.get_fallback_code()
        self.version_control.commit_version(fallback, "Failed generation snapshot")
        return fallback

    async def improve_code(
        self,
        original_code: GeneratedCode,
        analysis: CodeAnalysis,
        max_attempts: int = 5
    ) -> GeneratedCode:
        """Improved code generation with better error handling"""
        self.fix_attempts = 0
        current_code = original_code
        
        while self.fix_attempts < max_attempts:
            try:
                # Try parsing the code
                ast.parse(current_code)
                self.code_history.append(current_code)
                return current_code
            except SyntaxError as se:
                if self.fix_attempts >= self.max_fixes:
                    logger.warning(f"Max fix attempts ({self.max_fixes}) reached")
                    break
                    
                # Apply fixes and track if anything changed
                fixed_code, was_fixed = await self._fix_specific_issues(current_code, se)
                
                if was_fixed:
                    current_code = fixed_code
                    self.fix_attempts += 1
                    continue
                else:
                    # If no fixes worked, try regenerating with context
                    try:
                        response = await self._call_ollama_with_rate_limit(
                            messages=[{
                                "role": "system",
                                "content": f"Fix this code. Original error: {str(se)}\n\nCode:\n{current_code}"
                            }],
                            temperature=0.7
                        )
                        current_code = response.get('response', '').strip()
                        self.fix_attempts += 1
                    except Exception as e:
                        logger.error(f"Failed to regenerate code: {str(e)}")
                        break

        # If we get here, return the best version we have
        self.code_history.append(current_code)
        return current_code

    def run_advanced_tests(self, code: GeneratedCode) -> Dict[str, Any]:
        """Runs comprehensive tests on the generated code"""
        results = {
            "syntax_check": True,
            "static_analysis": {},
            "runtime_tests": [],
            "security_checks": [],
            "performance_metrics": {}
        }

        # Static Analysis
        try:
            tree = ast.parse(code.code)
            results["static_analysis"].update({
                "complexity": code.metrics["complexity"],
                "maintainability_index": self._calculate_maintainability_index(code.code),
                "documentation_coverage": self._calculate_doc_coverage(tree),
                "type_hint_coverage": self._calculate_type_hint_coverage(tree)
            })
        except SyntaxError as e:
            results["syntax_check"] = False
            results["static_analysis"]["error"] = str(e)

        # Security Analysis
        results["security_checks"] = self._run_security_checks(code.code)

        # Runtime Tests
        with self.environment.sandbox() as temp_dir:
            for test_case in code.test_cases:
                test_result = self._run_single_test(test_case, temp_dir)
                results["runtime_tests"].append(test_result)

        # Performance Testing
        results["performance_metrics"] = self._measure_performance(code.code)

        return results

    def _calculate_maintainability_index(self, code: str) -> float:
        """Calculate maintainability index based on various metrics"""
        lines = code.splitlines()
        loc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])

        # Halstead metrics (simplified)
        operators = set()
        operands = set()
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.operator):
                operators.add(type(node).__name__)
            elif isinstance(node, ast.Name):
                operands.add(node.id)

        n1 = len(operators)
        n2 = len(operands)
        N = ast.unparse(tree).count(' ') + 1

        # Calculate maintainability index
        volume = N * (n1 + n2).bit_length()
        difficulty = (n1 / 2) * (len(operands) / n2) if n2 > 0 else 0
        effort = volume * difficulty

        maintainability = max(0, (171 - 5.2 * volume.bit_length() - 0.23 * effort - 16.2 * loc.bit_length()) * 100 / 171)
        return round(maintainability, 2)

    def _calculate_doc_coverage(self, tree: ast.AST) -> float:
        """Calculate documentation coverage percentage with improved whitespace handling"""
        total_nodes = 0
        documented_nodes = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                total_nodes += 1
                # Use clean=True to properly handle whitespace in docstrings
                if ast.get_docstring(node, clean=True):
                    documented_nodes += 1

        return round(documented_nodes / total_nodes * 100, 2) if total_nodes > 0 else 0

    def _calculate_type_hint_coverage(self, tree: ast.AST) -> float:
        """Calculate type hint coverage percentage"""
        total_annotations = 0
        total_annotatable = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check return type annotation
                total_annotatable += 1
                if node.returns:
                    total_annotations += 1

                # Check argument annotations
                for arg in node.args.args:
                    total_annotatable += 1
                    if arg.annotation:
                        total_annotations += 1

        return round(total_annotations / total_annotatable * 100, 2) if total_annotatable > 0 else 0

    def _run_security_checks(self, code: str) -> List[Dict[str, str]]:
        """Run basic security checks on the code"""
        security_issues = []
        tree = ast.parse(code)

        # Check for common security issues
        for node in ast.walk(tree):
            # Check for eval() usage
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'eval':
                    security_issues.append({
                        "severity": "high",
                        "issue": "Use of eval() detected",
                        "description": "eval() can execute arbitrary code and should be avoided"
                    })

            # Check for hardcoded credentials
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if any(sensitive in target.id.lower()
                            for sensitive in ['password', 'secret', 'key', 'token']):
                            security_issues.append({
                                "severity": "medium",
                                "issue": f"Possible hardcoded credential in variable '{target.id}'",
                                "description": "Avoid storing sensitive data in code"
                            })

        return security_issues

    def _measure_performance(self, code: str) -> Dict[str, float]:
        """Measure code performance metrics"""
        import timeit

        # Execution time
        try:
            execution_time = timeit.timeit(lambda: exec(code), number=100) / 100
            return {"avg_execution_time": execution_time}
        except Exception as e:
            return {"execution_error": str(e)}

    def test_code(self, code: GeneratedCode) -> List[CodeTestResult]:
        """Executes test cases and returns results"""
        results = []

        with self.environment.sandbox() as temp_dir:
            test_file = temp_dir / "test_code.py"

            # Normalize code before writing to file
            normalized_code = code.code.strip()
            test_file.write_text(normalized_code + '\n')  # Ensure single newline at end

            for test_case in code.test_cases:
                # Normalize test case
                normalized_test = test_case.strip()
                start_time = datetime.now()
                try:
                    coverage_result = subprocess.run(
                        ["coverage", "run", "-m", "pytest", str(test_file)],
                        capture_output=True,
                        text=True
                    )

                    # Parse coverage results
                    coverage_report = subprocess.run(
                        ["coverage", "report", "--format=json"],
                        capture_output=True,
                        text=True
                    )

                    if coverage_report.returncode == 0:
                        coverage_data = json.loads(coverage_report.stdout)
                        code.metrics["test_coverage"] = coverage_data.get("totals", {}).get("percent_covered", 0)

                    # Execute test in subprocess for isolation
                    result = subprocess.run(
                        [sys.executable, "-c", normalized_test],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}  # Ensure consistent encoding
                    )

                    execution_time = (datetime.now() - start_time).total_seconds()

                    # Filter out common harmless warnings and encoding messages from stderr
                    error_message = None
                    if result.returncode != 0:
                        error_lines = [
                            line for line in result.stderr.splitlines()
                            if not any(harmless in line.lower() for harmless in [
                                'resourcewarning',
                                'deprecation',
                                'encoding declaration',
                                'whitespace'
                            ])
                        ]
                        if error_lines:
                            error_message = '\n'.join(error_lines)

                    results.append(CodeTestResult(
                        passed=result.returncode == 0,
                        error_message=error_message,
                        execution_time=execution_time,
                        memory_usage=None
                    ))

                except subprocess.TimeoutExpired:
                    results.append(CodeTestResult(
                        passed=False,
                        error_message="Test execution timed out",
                        execution_time=5.0,
                        memory_usage=None
                    ))
                except Exception as e:
                    results.append(CodeTestResult(
                        passed=False,
                        error_message=str(e),
                        execution_time=0.0,
                        memory_usage=None
                    ))

        return results

    def display_code(self, code: GeneratedCode):
        """Displays code with syntax highlighting"""
        console.print("\n[bold]Generated Code:[/bold]")
        console.print(Syntax(code.code, "python", theme="monokai"))

        if code.test_cases:
            console.print("\n[bold]Test Cases:[/bold]")
            for i, test in enumerate(code.test_cases, 1):
                console.print(f"\nTest {i}:")
                console.print(Syntax(test, "python", theme="monokai"))

    def generate_architecture_diagram(self, code: GeneratedCode):
        """Generate UML diagram using pyreverse with improved error handling"""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                try:
                    # First, validate the code is syntactically correct
                    valid, _ = self.environment.validate_syntax(code.code)
                    if not valid:
                        logger.warning("Cannot generate diagram for code with syntax errors")
                        return None
                        
                    # Write to the temp file
                    f.write(code.code)
                    f.flush()
                    
                    # Create diagram directory
                    output_dir = self.version_control.repo_path / "diagrams"
                    output_dir.mkdir(exist_ok=True)
                    
                    # Try to run pyreverse
                    result = subprocess.run(
                        ["pyreverse", "-o", "png", "-p", "GeneratedCode", "-d", str(output_dir), f.name],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if result.returncode != 0:
                        logger.warning(f"Pyreverse error: {result.stderr}")
                        return None
                        
                    diagram_path = output_dir / "classes_GeneratedCode.png"
                    if diagram_path.exists():
                        return diagram_path
                    return None
                    
                finally:
                    # Ensure we clean up the temp file
                    try:
                        os.unlink(f.name)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Diagram generation failed: {str(e)}")
            return None

    def save_feedback(self, feedback: str):
        """Store user feedback for future improvements"""
        feedback_file = self.version_control.repo_path / "feedback.log"
        timestamp = datetime.now().isoformat()
        feedback_file.write_text(f"{timestamp}: {feedback}\n", encoding="utf-8")

    def _check_security(self, code: str) -> List[dict]:
        """Enhanced security checks using bandit"""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
                f.write(code)
                f.flush()
                
                result = subprocess.run(
                    ["bandit", "-f", "json", "-r", f.name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode in [0, 1]:
                    return json.loads(result.stdout).get("results", [])
                return []
        except Exception as e:
            logger.error(f"Security scan failed: {str(e)}")
            return []

async def main():
    # Initialize the AI assistant with Claude as the default model
    model_choice = os.environ.get("MODEL_CHOICE", "groq").lower()
    
    try:
        if model_choice == "ollama":
            assistant = AICodeAssistant(model_provider=ModelProvider.OLLAMA)
        elif model_choice == "groq":
            assistant = AICodeAssistant(model_provider=ModelProvider.GROQ)
        else:
            assistant = AICodeAssistant(model_provider=ModelProvider.CLAUDE)
    except Exception as e:
        logger.error(f"Failed to initialize with {model_choice}, falling back to GROQ: {str(e)}")
        assistant = AICodeAssistant(model_provider=ModelProvider.GROQ)
    
    console.print(f"[bold green]AI Code Assistant initialized using {assistant.model_provider.value} model[/bold green]")
    
    while True:
        try:
            # Check if this is a follow-up or initial prompt
            if not assistant.conversation.get_conversation_history():
                task = input("Enter your code generation task: ")
                console.print("[bold]Generating code...[/bold]")
            else:
                task = input("\nEnter a follow-up request (or type 'exit' to quit, 'new' for a new conversation): ")
                
            if task.lower() == 'exit':
                break
            elif task.lower() == 'new':
                assistant.conversation.clear()
                console.print("[bold]Starting new conversation[/bold]")
                continue
                
            # Generate code using conversation history
            code = await assistant.generate_code(task)
            
            # Ensure code_versions directory exists
            Path("code_versions").mkdir(parents=True, exist_ok=True)
            
            # Validate syntax before running advanced tests
            try:
                # Try to fix any syntax errors
                valid, error = assistant.environment.validate_syntax(code.code)
                if not valid:
                    logger.warning(f"Syntax error in generated code: {error}")
                    # Try to fix the code
                    fixed_code, success = await assistant._fix_specific_issues(code.code, error)
                    if success:
                        logger.info("Successfully fixed syntax errors")
                        code.code = fixed_code
                    else:
                        logger.warning("Could not fix syntax errors automatically")
                
                # Run advanced tests only if code is valid
                test_results = assistant.run_advanced_tests(code)
                
                # Display system
                console.print("\n[bold]Code Generation Result:[/bold]")
                assistant.display_code(code)
                
                # Show test results summary
                console.print("\n[bold]Test Results:[/bold]")
                console.print(f"Syntax Check: {'✅ Passed' if test_results['syntax_check'] else '❌ Failed'}")
                if 'maintainability_index' in test_results['static_analysis']:
                    console.print(f"Maintainability: {test_results['static_analysis']['maintainability_index']:.1f}/100")
                if 'documentation_coverage' in test_results['static_analysis']:
                    console.print(f"Documentation Coverage: {test_results['static_analysis']['documentation_coverage']:.1f}%")
                
                # Security check results
                if test_results.get('security_checks'):
                    console.print("\n[bold red]Security Issues Found:[/bold red]")
                    for issue in test_results['security_checks']:
                        console.print(f"- {issue.get('issue', 'Unknown issue')}")
                else:
                    console.print("[green]No security issues found[/green]")
            
            except Exception as e:
                logger.error(f"Error in code validation: {str(e)}")
                console.print(f"[bold red]Code validation error:[/bold red] {str(e)}")
                test_results = {
                    "syntax_check": False,
                    "static_analysis": {},
                    "runtime_tests": [],
                    "security_checks": [],
                    "performance_metrics": {}
                }
            
            # Run code tests if test cases exist (even if validation failed)
            if code.test_cases:
                console.print("\n[bold]Running Tests:[/bold]")
                try:
                    code_test_results = assistant.test_code(code)
                    for i, result in enumerate(code_test_results, 1):
                        status = "[green]✅ Passed[/green]" if result.passed else f"[red]❌ Failed: {result.error_message}[/red]"
                        console.print(f"Test {i}: {status} (Time: {result.execution_time:.3f}s)")
                except Exception as e:
                    logger.error(f"Error running tests: {str(e)}")
                    console.print(f"[bold red]Test execution error:[/bold red] {str(e)}")
            
            # Generate meta prompt for reference
            meta_prompt = assistant.create_meta_prompt(task)
            console.print("\n[bold]Generated Using:[/bold]")
            console.print(f"Objectives: {len(meta_prompt.objectives)} | Constraints: {len(meta_prompt.constraints)}")
            
            # Version history (with error handling)
            try:
                if assistant.version_control.repo.heads:
                    commits = list(assistant.version_control.repo.iter_commits())
                    console.print(f"\n[bold]Version History:[/bold] ({len(commits)} commits)")
                    for commit in commits[:3]:
                        console.print(f"- {commit.hexsha[:7]}: {commit.message}")
            except Exception as e:
                logger.debug(f"Version history error: {str(e)}")

            # Generate architecture diagram (with error handling)
            try:
                diagram_path = assistant.generate_architecture_diagram(code)
                if diagram_path and Path(diagram_path).exists():
                    console.print(f"\n[bold]Architecture diagram generated:[/bold] {diagram_path}")
            except Exception as e:
                logger.error(f"Diagram generation error: {str(e)}")
                
            # Generate documentation (with error handling)
            try:
                doc_path = assistant.documentation_gen.generate_html_docs(code)
                if doc_path:
                    console.print(f"\n[bold]Documentation generated:[/bold] {doc_path}")
            except Exception as e:
                logger.error(f"Documentation generation error: {str(e)}")
                
            # Collect user feedback
            feedback = input("\nAny feedback on the generated code? (Press Enter to skip): ")
            if feedback.strip():
                assistant.save_feedback(feedback)
                console.print("[green]Feedback saved[/green]")
                
            # Try to improve code if it needs improvement
            if test_results.get('static_analysis', {}).get('maintainability_index', 100) < 65 or \
               test_results.get('static_analysis', {}).get('documentation_coverage', 100) < 50:
                console.print("\n[bold yellow]Code quality issues detected. Attempting improvement...[/bold yellow]")
                try:
                    code_analysis = CodeAnalysis(
                        quality=CodeQuality.NEEDS_IMPROVEMENT,
                        complexity_score=test_results.get('static_analysis', {}).get('complexity', 5),
                        documentation_coverage=test_results.get('static_analysis', {}).get('documentation_coverage', 50),
                        suggestions=["Improve documentation", "Simplify complex sections"]
                    )
                    
                    improved_code = await assistant.improve_code(code, code_analysis)
                    if improved_code.code != code.code:
                        console.print("\n[bold green]Code successfully improved![/bold green]")
                        assistant.display_code(improved_code)
                        
                        # Create a new version with the improved code
                        assistant.version_control.commit_version(improved_code, "Automatic code improvement")
                        
                        # Add the improvement to conversation history
                        assistant.conversation.add_assistant_message(
                            "I've improved the code based on quality analysis.",
                            improved_code
                        )
                        
                        # Update code for dependency processing
                        code = improved_code
                except Exception as e:
                    logger.error(f"Code improvement error: {str(e)}")
            
            # Analyze and generate requirements file
            try:
                for match in re.finditer(r"import\s+(\w+)|from\s+(\w+)", code.code):
                    package = match.group(1) or match.group(2)
                    if package and package not in ["os", "sys", "re", "json", "time", "datetime", "typing"]:
                        assistant.dependency_manager.add_requirement(f"{package}")
                
                assistant.dependency_manager.generate_requirements_file(Path("code_versions"))
                console.print("\n[bold]Requirements file generated[/bold]")
            except Exception as e:
                logger.error(f"Dependency management error: {str(e)}")
                
        except Exception as e:
            code = GeneratedCode.get_fallback_code()
            logger.error(f"Critical failure: {str(e)}")
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())