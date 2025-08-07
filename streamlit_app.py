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
from pydantic import BaseModel, Field, field_validator, ConfigDict # Updated import
import openai
from rich.console import Console as RichConsole # Renamed to avoid conflict with Streamlit Console
from rich.syntax import Syntax
from rich.logging import RichHandler
import streamlit as st # Added Streamlit

# Configure rich logging (will output to terminal where Streamlit runs)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", # Added timestamp etc.
    handlers=[RichHandler(rich_tracebacks=True, console=RichConsole(stderr=True))] # Log to stderr
)
logger = logging.getLogger("AICodeAssistantLogger")
# console = RichConsole() # Removed, use st elements for UI output

load_dotenv()

# Environment Setup
class CodeEnvironment:
    """Manages sandbox environment for code execution"""
    def __init__(self, temp_dir: Optional[Path] = None):
        # Use Streamlit's temp directory if possible, falling back to system default
        try:
            self.temp_dir = temp_dir or Path(st.temporary_directory())
        except Exception: # Catch potential errors if running outside Streamlit context briefly
             self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Code environment initialized at: {self.temp_dir}")

    @classmethod
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[SyntaxError]]:
        """Validate code syntax with detailed error reporting"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as se:
            logger.warning(f"Syntax validation failed: {se}")
            return False, se

    @contextmanager
    def sandbox(self):
        """Creates an isolated environment for code execution"""
        original_paths = sys.path.copy()
        original_modules = set(sys.modules.keys())
        logger.info(f"Entering sandbox: {self.temp_dir}")
        try:
            sys.path.insert(0, str(self.temp_dir))
            yield self.temp_dir
        finally:
            logger.info("Exiting sandbox.")
            sys.path = original_paths
            # Remove any newly imported modules
            mods_to_remove = set(sys.modules.keys()) - original_modules
            if mods_to_remove:
                logger.debug(f"Removing modules imported in sandbox: {mods_to_remove}")
            for mod in mods_to_remove:
                sys.modules.pop(mod, None)

    def test_imports(self, code: str) -> Dict[str, bool]:
        """Tests if all imports in the code are available"""
        imports_found = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_found.add(alias.name.split('.')[0]) # Get top-level package
                elif isinstance(node, ast.ImportFrom):
                    if node.module: # Handle relative imports (level > 0) if needed, but focus on absolute
                         imports_found.add(node.module.split('.')[0])

            results = {}
            logger.info(f"Testing imports: {imports_found}")
            for imp in imports_found:
                 # Skip built-in modules efficiently
                if imp in sys.builtin_module_names:
                    results[imp] = True
                    continue
                try:
                    # Try finding the spec first - more robust
                    spec = importlib.util.find_spec(imp)
                    results[imp] = spec is not None
                    if spec is None:
                        logger.warning(f"Import '{imp}' not found.")
                except ModuleNotFoundError:
                    results[imp] = False
                    logger.warning(f"Import '{imp}' not found (ModuleNotFoundError).")
                except Exception as e: # Catch other potential import errors
                    results[imp] = False
                    logger.error(f"Error testing import '{imp}': {e}")

            logger.info(f"Import test results: {results}")
            return results
        except SyntaxError:
            logger.error("Syntax error during import testing.")
            return {"parsing_error": False}
        except Exception as e:
            logger.error(f"Unexpected error during import testing: {e}")
            return {"unexpected_error": False}


# Add ModelProvider to select between available models
class ModelProvider(Enum):
    GROQ = "groq"
    # OLLAMA = "ollama" # Commented out Ollama for simplicity, can be re-added
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
    complexity_score: float = Field(default=5.0, ge=0, le=10) # Provide default
    security_issues: List[dict] = Field(default_factory=list)
    documentation_coverage: float = Field(default=0.0, ge=0, le=100) # Provide default
    test_coverage: Optional[float] = Field(None, ge=0, le=100)

    # Pydantic v2 uses field_validator
    @field_validator('complexity_score')
    @classmethod
    def validate_complexity_score(cls, value: float) -> float:
        if value > 10:
            return 10.0
        elif value < 0:
            return 0.0
        return value

class CodeTestResult(BaseModel):
    passed: bool
    error_message: Optional[str] = None
    execution_time: float
    memory_usage: Optional[float] = None

    @field_validator('execution_time')
    @classmethod
    def validate_execution_time(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Execution time cannot be negative")
        return value

    @field_validator('memory_usage')
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
    documentation: Optional[str] = None # Store path as string

    model_config = ConfigDict(protected_namespaces=()) # Pydantic v2 config

    @classmethod
    def get_fallback_code(cls) -> 'GeneratedCode':
        """Return a safe fallback when generation fails"""
        logger.warning("Returning fallback code due to generation failure.")
        return cls(
            code="# Code generation failed\n# Default safe code\npass",
            description="Fallback code due to generation failure",
            language="python",
            metrics={"error": "Generation failed"}
        )

    # _validate_code is complex and potentially error-prone, relying on AST parse is safer
    # Let's remove the custom _validate_code and rely on CodeEnvironment.validate_syntax

# Conversation class for memory
class Conversation(BaseModel):
    """Stores conversation history between user and assistant"""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    generated_codes: List[GeneratedCode] = Field(default_factory=list)
    model_provider: ModelProvider = Field(default=ModelProvider.CLAUDE)

    model_config = ConfigDict(protected_namespaces=()) # Pydantic v2 config

    def add_user_message(self, content: str):
        """Add a user message to the conversation history"""
        logger.info(f"Adding user message: {content[:100]}...")
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, code: Optional[GeneratedCode] = None, thinking: Optional[str] = None):
        """Add an assistant message to the conversation history with optional code and thinking"""
        logger.info(f"Adding assistant message: {content[:100]}...")
        message = {"role": "assistant", "content": content}
        if thinking:
            # Truncate long thinking messages for logging
            thinking_preview = thinking[:200] + "..." if len(thinking) > 200 else thinking
            logger.debug(f"Assistant thinking: {thinking_preview}")
            message["thinking"] = thinking
        self.messages.append(message)
        if code:
            # Maybe log a hash or snippet of the code, not the whole thing
            code_hash = hashlib.sha256(code.code.encode()).hexdigest()[:8]
            logger.info(f"Associated generated code (hash: {code_hash}) with assistant message.")
            self.generated_codes.append(code)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history"""
        return self.messages

    def get_latest_code(self) -> Optional[GeneratedCode]:
        """Get the most recently generated code"""
        return self.generated_codes[-1] if self.generated_codes else None

    def clear(self):
        """Clear the conversation history"""
        logger.info("Clearing conversation history.")
        self.messages = []
        self.generated_codes = []

class MetaPrompt(BaseModel):
    context: str
    objectives: List[str]
    constraints: List[str]
    examples: Optional[List[str]] = None

    def to_string(self) -> str:
        # Use f-string formatting directly for cleaner representation
        objectives_str = "\n".join(f"- {obj}" for obj in self.objectives)
        constraints_str = "\n".join(f"- {con}" for con in self.constraints)
        examples_str = f"Examples:\n\n" + "\n\n".join(self.examples) if self.examples else ""

        return f"""
Context: {self.context}

Objectives:
{objectives_str}

Constraints:
{constraints_str}

{examples_str}
"""

class CodeVersioning:
    """Git-based version control with empty repo handling"""
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        # Create the directory if it doesn't exist
        self.repo_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing version control at: {self.repo_path}")
        self._init_repo()

    def _create_initial_commit(self):
        """Create initial commit with README file"""
        try:
            logger.info("Creating initial commit in repository.")
            # Create and add README file
            readme_path = self.repo_path / "README.md"
            if not readme_path.exists(): # Avoid overwriting
                 readme_path.write_text("# Generated Code Repository\nThis repository contains generated code versions.", encoding='utf-8')

            # Configure git user if not set globally or locally
            try:
                self.repo.config_reader().get_value("user", "name")
                self.repo.config_reader().get_value("user", "email")
            except Exception:
                 logger.info("Git user name/email not configured. Setting default.")
                 with self.repo.config_writer() as cw:
                    cw.set_value("user", "name", "AI Code Assistant").release()
                    cw.set_value("user", "email", "ai.assistant@example.com").release() # Use a generic email

            # Add and commit README
            if readme_path.exists() and str(readme_path) not in self.repo.git.ls_files():
                self.repo.index.add([str(readme_path)])
                self.repo.index.commit("Initial commit: Add README")
                logger.info("Initial commit created successfully.")
            else:
                 logger.info("README already exists or is tracked. Skipping initial commit steps.")

            # Create main branch if it doesn't exist
            if 'main' not in self.repo.heads:
                self.repo.create_head('main')
                logger.info("Created 'main' branch.")
            if not self.repo.active_branch or self.repo.active_branch.name != 'main':
                 self.repo.heads.main.checkout()
                 logger.info("Checked out 'main' branch.")

        except GitError as e:
            logger.error(f"Failed to create initial commit: {str(e)}", exc_info=True)
            raise

    def _init_repo(self):
        """Initialize repository with proper branch setup"""
        try:
            self.repo = Repo(self.repo_path)
            logger.info("Existing Git repository found.")

            # Check if repo is empty (no commits)
            try:
                self.repo.head.commit
            except ValueError: # Catches "Reference at 'HEAD' does not exist"
                logger.warning("Repository is empty. Creating initial commit.")
                self._create_initial_commit()

            # Ensure main branch exists and is checked out
            if 'main' not in self.repo.heads:
                logger.warning("'main' branch not found. Creating and checking out.")
                if self.repo.heads: # If other branches exist, create main from current HEAD
                    self.repo.create_head('main')
                else: # If truly empty (should have been caught above, but safety first)
                    self._create_initial_commit() # This will create main
                self.repo.heads.main.checkout()
            elif self.repo.active_branch.name != 'main':
                 logger.info("Checking out 'main' branch.")
                 self.repo.heads.main.checkout()

        except GitError as e: # Covers InvalidGitRepositoryError, NoSuchPathError
            logger.warning(f"Git repository not found or invalid at {self.repo_path}. Initializing new repository. Error: {e}")
            self.repo = Repo.init(self.repo_path)
            self._create_initial_commit()

    def commit_version(self, code: GeneratedCode, message: str = "Auto-commit generated code"):
        """Safe commit with branch handling"""
        try:
            if not hasattr(self, 'repo'):
                logger.error("Repository not initialized. Cannot commit.")
                return None

            # Ensure we are on the main branch
            if self.repo.active_branch.name != 'main':
                 logger.warning(f"Not on 'main' branch (current: {self.repo.active_branch.name}). Checking out 'main'.")
                 self.repo.heads.main.checkout()

            # Create directory and file
            code_file = self.repo_path / "generated_code.py"
            self.repo_path.mkdir(parents=True, exist_ok=True) # repo_path should already exist, but belt-and-suspenders

            # Write the file
            logger.info(f"Writing code to {code_file}")
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code.code)

            # Verify file exists before git operations
            if not code_file.exists():
                logger.error(f"Failed to create file: {code_file}")
                raise FileNotFoundError(f"Failed to create file: {code_file}")

            # Git operations
            logger.info(f"Adding {code_file} to index.")
            self.repo.index.add([str(code_file)])

            # Check if there are changes to commit
            if self.repo.is_dirty(index=True, working_tree=False):
                commit_message = f"{message} (desc: {code.description[:50]}...)"
                logger.info(f"Committing changes with message: '{commit_message}'")
                commit = self.repo.index.commit(commit_message)
                logger.info(f"Commit successful: {commit.hexsha}")
                return commit.hexsha
            else:
                 logger.info("No changes detected in generated_code.py. Skipping commit.")
                 # Return the latest commit hash if no changes
                 try:
                     return self.repo.head.commit.hexsha
                 except ValueError: # Handle case of repo existing but having no commits (should be rare after init)
                     logger.warning("No commit history found after attempting commit.")
                     return None


        except GitError as e:
             logger.error(f"Git commit error: {str(e)}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Unexpected error during version control commit: {str(e)}", exc_info=True)
            return None


class DependencyManager:
    """Advanced dependency resolution and vulnerability checking"""
    def __init__(self):
        self.requirements = set()
        # self.vulnerability_db = self._load_vulnerability_db() # Defer loading until needed

    # Removed vulnerability DB loading for simplicity, can be added back if needed
    # def _load_vulnerability_db(self):
    #     try:
    #         # Use a cached session for potential retries or multiple calls
    #         session = requests.Session()
    #         response = session.get("https://raw.githubusercontent.com/pyupio/safety-db/master/data/insecure_full.json", timeout=10)
    #         response.raise_for_status()
    #         logger.info("Successfully loaded vulnerability database.")
    #         return response.json()
    #     except requests.exceptions.RequestException as e:
    #         logger.warning(f"Failed to load vulnerability DB: {str(e)}. Continuing without vulnerability checks.")
    #         return {}
    #     except json.JSONDecodeError as e:
    #          logger.warning(f"Failed to parse vulnerability DB JSON: {str(e)}. Continuing without vulnerability checks.")
    #          return {}
    #     except Exception as e:
    #         logger.warning(f"An unexpected error occurred while loading vulnerability DB: {str(e)}")
    #         return {}

    def add_requirement(self, package: str):
        """Add a package requirement with basic validation"""
        package = package.strip()
        if not package:
            return
        # Basic check to avoid adding obviously invalid package names
        if re.match(r'^[a-zA-Z0-9._-]+$', package):
            try:
                # Use packaging.requirements to potentially normalize
                req = Requirement(package)
                self.requirements.add(str(req))
                logger.debug(f"Added requirement: {str(req)}")
            except ValueError:
                logger.warning(f"Invalid requirement format: {package}. Adding as is.")
                self.requirements.add(package) # Add even if invalid format, maybe user intended it
        else:
             logger.warning(f"Potentially invalid package name skipped: {package}")

    # Removed vulnerability check function
    # def check_vulnerabilities(self): ...

    def generate_requirements_file(self, path: Path):
        """Generate requirements.txt file"""
        if not self.requirements:
            logger.info("No requirements found to generate requirements.txt.")
            return

        output_file = path / "requirements.txt"
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(output_file, "w", encoding='utf-8') as f:
                f.write("\n".join(sorted(list(self.requirements))))
            logger.info(f"Generated requirements file at: {output_file}")
        except IOError as e:
            logger.error(f"Failed to write requirements file {output_file}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error generating requirements file: {e}")

class DocumentationGenerator:
    """Auto-generates documentation from code"""
    def generate_html_docs(self, code: GeneratedCode, output_dir: Optional[Path] = None) -> Optional[str]:
        """Generate HTML documentation using pdoc"""
        if not code or not code.code.strip():
            logger.warning("Cannot generate documentation for empty code.")
            return None

        output_dir = output_dir or Path("code_versions/docs")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Attempting to generate documentation in: {output_dir}")

        # Use a temporary file within a controlled environment if possible
        # Using CodeEnvironment's sandbox might be safer
        code_env = CodeEnvironment() # Create a temporary env for this
        try:
            with code_env.sandbox() as temp_sandbox_dir:
                temp_module_path = temp_sandbox_dir / "temp_module.py"
                logger.debug(f"Writing code to temporary file: {temp_module_path}")
                temp_module_path.write_text(code.code, encoding='utf-8')

                # Define expected output path
                # pdoc creates a directory named after the module inside the output_dir
                expected_html_path = output_dir / "temp_module.html"

                # Run pdoc
                cmd = ["pdoc", "--html", str(temp_module_path), "-o", str(output_dir)]
                logger.info(f"Running pdoc command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8', # Specify encoding
                    timeout=60 # Add a timeout
                )

                logger.debug(f"pdoc stdout:\n{result.stdout}")
                logger.debug(f"pdoc stderr:\n{result.stderr}")

                if result.returncode == 0 and expected_html_path.exists():
                    logger.info(f"Documentation generated successfully: {expected_html_path}")
                    return str(expected_html_path) # Return path as string
                else:
                    logger.error(f"pdoc failed (return code {result.returncode}). Stderr: {result.stderr}")
                    return None
        except FileNotFoundError:
            logger.error("pdoc command not found. Ensure pdoc is installed and in PATH.")
            return None
        except subprocess.TimeoutExpired:
             logger.error("pdoc execution timed out.")
             return None
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}", exc_info=True)
            return None


class AICodeAssistant:
    def __init__(self, model_provider: ModelProvider = ModelProvider.CLAUDE, repo_path: Path = Path("code_versions")):
        logger.info(f"Initializing AICodeAssistant with model: {model_provider.value}")
        self.tokens_used = 0
        self.last_reset = datetime.now()
        self.rate_limit = 90000  # Tokens per minute (adjust as needed)
        # self.code_history = [] # Less useful with Conversation history
        self.fix_attempts = 0
        self.max_fixes = 3 # Max attempts to auto-fix syntax
        self.environment = CodeEnvironment()
        # self.user_feedback = "" # Feedback handled per session via Streamlit input
        self.version_control = CodeVersioning(repo_path)
        self.dependency_manager = DependencyManager()
        self.documentation_gen = DocumentationGenerator()
        # self.cache = {} # Caching can be complex with state, removed for now
        self.model_provider = model_provider
        self.conversation = Conversation(model_provider=model_provider)

        # API Client Initialization
        self.openai_client = None
        self.anthropic_client = None

        if self.model_provider == ModelProvider.GROQ:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if groq_api_key:
                self.openai_client = openai.OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=groq_api_key
                )
                logger.info("Groq client initialized.")
            else:
                logger.error("GROQ_API_KEY not found. Groq model will not function.")
                raise ValueError("GROQ_API_KEY environment variable not set.")

        elif self.model_provider == ModelProvider.CLAUDE:
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                try:
                    self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                    logger.info("Anthropic client initialized.")
                except Exception as e:
                    logger.error(f"Failed to initialize Anthropic client: {e}")
                    raise ValueError(f"Failed to initialize Anthropic client: {e}")
            else:
                logger.error("ANTHROPIC_API_KEY not found. Claude model will not function.")
                raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

        # Removed Ollama initialization
        # elif self.model_provider == ModelProvider.OLLAMA:
        #     logger.info("Ollama model selected. Ensure Ollama service is running.")
            # No specific client init needed here, uses requests

        # self._verify_repo_health() # Called within CodeVersioning init
        # self._init_git_branch() # Called within CodeVersioning init


    def create_meta_prompt(self, task_description: str) -> MetaPrompt:
        """Creates a structured meta-prompt for code generation"""
        logger.debug("Creating meta prompt.")
        # Simplified context for clarity
        context = f"Generate Python code for the following task: {task_description}"
        objectives = [
            "Produce clean, well-structured, and maintainable Python code.",
            "Implement robust error handling using try-except blocks.",
            "Include clear docstrings (Google style) for modules, classes, and functions.",
            "Use type hints for function signatures and variables where appropriate.",
            "Adhere to PEP 8 style guidelines.",
            "Ensure code is reasonably modular and avoids unnecessary complexity.",
            "Add logging for key operations or potential issues (use the 'logging' module)."
        ]
        constraints = [
            "Target Python version: 3.8+",
            "Only use the Python standard library unless specific packages are requested or obviously necessary.",
            "Handle potential exceptions gracefully (e.g., file not found, network errors, invalid input).",
            "Avoid hardcoding sensitive information (like API keys or passwords).",
            "Generated code must be syntactically correct Python.",
            "Ensure all string literals (single, double, triple quotes) are properly terminated.",
            "Focus on correctness and clarity."
        ]
        return MetaPrompt(
            context=context,
            objectives=objectives,
            constraints=constraints
        )

    def _get_claude_tools(self) -> List[Dict[str, Any]]:
        """Define tools that Claude can call (Simplified for core validation)"""
        logger.debug("Defining Claude tools.")
        return [
            {
                "name": "validate_python_syntax", # Renamed for clarity
                "description": "Checks if the provided Python code has valid syntax using AST parsing. Returns true if valid, false with error details if invalid.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code snippet to validate."
                        }
                    },
                    "required": ["code"]
                }
            },
             {
                 "name": "check_python_imports", # Renamed for clarity
                 "description": "Checks if the top-level modules imported in the Python code seem installable or are built-in. Returns a dictionary mapping import names to a boolean indicating availability.",
                 "input_schema": {
                     "type": "object",
                     "properties": {
                         "code": {
                             "type": "string",
                             "description": "Python code containing import statements to check."
                         }
                     },
                     "required": ["code"]
                 }
             }
            # Removed execution and security tools for simplicity, focus on generation first
        ]

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool and return the result"""
        logger.info(f"Executing tool: {tool_name} with input keys: {list(tool_input.keys())}")
        try:
            if tool_name == "validate_python_syntax":
                 code_to_validate = tool_input.get("code", "")
                 if not code_to_validate:
                      return {"valid": False, "error": "No code provided to validate."}
                 valid, error = self.environment.validate_syntax(code_to_validate)
                 result = {"valid": valid, "error": str(error) if error else None}
                 logger.debug(f"Tool result (validate_python_syntax): {result}")
                 return result

            elif tool_name == "check_python_imports":
                 code_to_check = tool_input.get("code", "")
                 if not code_to_check:
                     return {"imports": {}, "error": "No code provided to check imports."}
                 # Run import test in a sandbox to avoid polluting main env
                 with self.environment.sandbox():
                    import_results = self.environment.test_imports(code_to_check)
                 result = {"imports": import_results}
                 logger.debug(f"Tool result (check_python_imports): {result}")
                 return result

            # Removed execution/security tool handling
            # elif tool_name == "test_code_execution": ...
            # elif tool_name == "check_security": ...

            else:
                logger.warning(f"Unknown tool requested: {tool_name}")
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Tool execution failed for '{tool_name}': {str(e)}", exc_info=True)
            return {"error": f"Tool execution failed: {str(e)}"}


    async def _call_anthropic_api(self, messages: List[Dict[str, Any]], system_prompt: str) -> Dict[str, Any]:
        """Call Claude's API with thinking and tools"""
        if not self.anthropic_client:
            logger.error("Anthropic client not initialized.")
            raise ValueError("Anthropic client not initialized. Make sure ANTHROPIC_API_KEY is set.")

        # Ensure Claude model name is correct (check Anthropic docs for latest)
        # claude_model = "claude-3-sonnet-20240229" # Example Sonnet
        claude_model = "claude-3-haiku-20240307" # Faster/Cheaper Haiku for iteration
        max_tokens = 4000 # Max output tokens
        # thinking_budget = 2000 # Optional thinking budget

        # Format messages for Claude API (handle potential tool results)
        claude_messages = []
        for msg in messages:
            # Ensure content is correctly formatted (string or list of blocks)
            if isinstance(msg.get("content"), list):
                 # If content is already a list (likely from tool use response), use it directly
                 claude_messages.append({
                      "role": msg["role"],
                      "content": msg["content"]
                 })
            elif isinstance(msg.get("content"), str):
                # If content is a string, wrap it in a text block
                 claude_messages.append({
                      "role": msg["role"],
                      "content": [{"type": "text", "text": msg["content"]}]
                 })
            # Skip messages with unexpected content format
            else:
                 logger.warning(f"Skipping message with unexpected content format: {msg}")


        try:
            tools = self._get_claude_tools()
            logger.info(f"Calling Anthropic API ({claude_model}) with {len(claude_messages)} messages.")
            # logger.debug(f"Claude messages: {json.dumps(claude_messages, indent=2)}") # Be careful logging full messages

            # Initial API call
            response = self.anthropic_client.messages.create(
                model=claude_model,
                max_tokens=max_tokens,
                system=system_prompt,
                # thinking={"type": "enabled", "budget_tokens": thinking_budget}, # Optional: enable thinking
                tools=tools,
                tool_choice={"type": "auto"}, # Let Claude decide when to use tools
                messages=claude_messages
            )

            logger.debug(f"Anthropic initial response stop reason: {response.stop_reason}")

            # Handle tool usage
            while response.stop_reason == "tool_use":
                logger.info("Claude requested tool use.")
                tool_calls = [block for block in response.content if block.type == "tool_use"]
                if not tool_calls:
                     logger.warning("Stop reason is 'tool_use' but no tool_use blocks found.")
                     break # Avoid infinite loop

                tool_results = []
                # Prepare the assistant's response message containing the tool calls
                assistant_message = {"role": "assistant", "content": response.content}
                claude_messages.append(assistant_message)


                # Execute tools and gather results
                for tool_call in tool_calls:
                    tool_name = tool_call.name
                    tool_input = tool_call.input
                    tool_use_id = tool_call.id
                    logger.info(f"Executing tool: {tool_name} (ID: {tool_use_id})")
                    tool_result_content = self._execute_tool(tool_name, tool_input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(tool_result_content) # Ensure content is JSON stringified
                         # Add error flag if necessary based on tool_result_content
                         # "is_error": "error" in tool_result_content
                    })

                # Add the tool results message for the next API call
                tool_result_message = {"role": "user", "content": tool_results}
                claude_messages.append(tool_result_message)

                logger.info("Re-calling Anthropic API with tool results.")
                # Make the follow-up API call
                response = self.anthropic_client.messages.create(
                    model=claude_model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    # thinking={"type": "enabled", "budget_tokens": thinking_budget},
                    tools=tools,
                    tool_choice={"type": "auto"},
                    messages=claude_messages
                )
                logger.debug(f"Anthropic response stop reason after tool use: {response.stop_reason}")

            # Extract final response and thinking
            final_content = ""
            thinking_content = "" # Not currently enabled

            # Process final response content blocks
            for block in response.content:
                if block.type == "text":
                    final_content += block.text
                # elif block.type == "thinking": # If thinking enabled
                #     thinking_content += block.thinking # Adapt based on actual structure

            logger.info("Anthropic API call successful.")
            return {
                "response": final_content.strip(),
                "thinking": thinking_content.strip()
            }

        except anthropic.APIConnectionError as e:
             logger.error(f"Anthropic API connection error: {e}", exc_info=True)
             raise ConnectionError(f"Anthropic API connection error: {e}") from e
        except anthropic.RateLimitError as e:
             logger.error(f"Anthropic API rate limit exceeded: {e}", exc_info=True)
             # Implement backoff strategy here if needed
             raise TimeoutError("Anthropic API rate limit exceeded.") from e # Map to TimeoutError for retry logic?
        except anthropic.APIStatusError as e:
             logger.error(f"Anthropic API status error ({e.status_code}): {e.message}", exc_info=True)
             raise SystemError(f"Anthropic API error ({e.status_code}): {e.message}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic API call: {str(e)}", exc_info=True)
            raise


    # Removed Ollama call function
    # async def _call_ollama_with_rate_limit(self, **kwargs) -> Any: ...

    async def _call_groq_api(self, messages: List[Dict[str, Any]], system_prompt: str) -> Dict[str, Any]:
        """Call Groq API"""
        if not self.openai_client:
            logger.error("Groq (OpenAI compatible) client not initialized.")
            raise ValueError("Groq client not initialized. Make sure GROQ_API_KEY is set.")

        groq_model = "meta-llama/llama-4-maverick-17b-128e-instruct" # Or Llama3 if preferred

        # Format messages for OpenAI/Groq API
        formatted_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            # Ensure content is a string, skip if not
            if isinstance(msg.get("content"), str):
                 formatted_messages.append({"role": msg["role"], "content": msg["content"]})
            elif isinstance(msg.get("content"), list): # Handle Claude's block format if necessary
                 text_content = " ".join(block.get("text", "") for block in msg["content"] if block.get("type") == "text")
                 if text_content:
                      formatted_messages.append({"role": msg["role"], "content": text_content})
            else:
                logger.warning(f"Skipping message with non-string content for Groq: {msg}")


        try:
            logger.info(f"Calling Groq API ({groq_model}) with {len(formatted_messages)} messages.")
            # logger.debug(f"Groq messages: {json.dumps(formatted_messages, indent=2)}") # Careful logging messages

            response = self.openai_client.chat.completions.create(
                model=groq_model,
                messages=formatted_messages,
                temperature=0.1, # Low temperature for more deterministic code
                max_tokens=4000 # Set a reasonable max token limit for output
            )

            generated_text = response.choices[0].message.content
            logger.info("Groq API call successful.")
            return {
                "response": generated_text.strip(),
                "thinking": ""  # Groq doesn't support thinking blocks
            }
        except openai.APIConnectionError as e:
             logger.error(f"Groq API connection error: {e}", exc_info=True)
             raise ConnectionError(f"Groq API connection error: {e}") from e
        except openai.RateLimitError as e:
             logger.error(f"Groq API rate limit exceeded: {e}", exc_info=True)
             raise TimeoutError("Groq API rate limit exceeded.") from e
        except openai.APIStatusError as e:
             logger.error(f"Groq API status error ({e.status_code}): {e.response}", exc_info=True)
             raise SystemError(f"Groq API error ({e.status_code}): {e.message}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Groq API call: {str(e)}", exc_info=True)
            raise

    async def generate_code(self, task_description: str) -> GeneratedCode:
        """Generation process with error handling, validation, and model selection"""
        logger.info(f"Starting code generation for task: {task_description[:100]}...")
        self.conversation.add_user_message(task_description) # Add user task to conversation first

        max_attempts = 3  # Reduced attempts for faster UI feedback
        backoff_time = 2  # Starting backoff time

        # Define system prompt (consider tailoring slightly based on model)
        prompt_path = Path(__file__).parent / "prompt.txt"
        try:
            system_prompt_template = prompt_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning("prompt.txt not found, using default system prompt.")
            system_prompt_template = """You are an expert Python code generation assistant.
Generate high-quality, production-ready Python code based on the user's request.

Follow these instructions carefully:
1.  **Understand the Request:** Analyze the user's requirements fully. Ask for clarification via text if needed, before generating code.
2.  **Code Quality:** Produce clean, well-structured, and readable Python code. Adhere strictly to PEP 8 style guidelines.
3.  **Error Handling:** Implement robust error handling using try-except blocks for anticipated issues (e.g., I/O, network, invalid input).
4.  **Documentation:** Include comprehensive docstrings (Google style preferred) for all modules, classes, and functions. Explain the purpose, arguments, and return values.
5.  **Type Hinting:** Use Python type hints for function signatures and important variables.
6.  **Modularity:** Write modular and reusable code where appropriate. Avoid overly long functions.
7.  **Imports:** Only import necessary standard library modules or packages explicitly requested or essential for the core task.
8.  **Security:** Avoid security pitfalls like using `eval()`, hardcoding credentials, or generating insecure code patterns. Alert the user to potential security considerations if applicable.
9.  **Output Format:**
    *   **Primary Output:** Generate the Python code enclosed ONLY within a single triple-backtick Python block (```python ... ```).
    *   **Explanation (Optional):** If necessary, provide a brief explanation or comments *outside* the code block.
    *   **Do NOT:** Do not include explanations, comments, or any other text *inside* the ```python ... ``` block itself, unless it's a standard Python comment (#).
    *   **String Literals:** Ensure all string literals (single, double, triple quotes) are correctly terminated within the code.
10. **Tool Use (if applicable):** If you have tools (like syntax validation), use them to verify the code *before* finalizing your response. If validation fails, attempt to fix the code.

Generate the Python code now based on the conversation history.
"""

        # Add model-specific nuances if needed
        if self.model_provider == ModelProvider.CLAUDE:
            system_prompt = system_prompt_template # + "\nUse available tools to validate syntax and imports before responding."
        else: # Groq
            system_prompt = system_prompt_template # Keep it clear for Groq

        for attempt in range(max_attempts):
            try:
                logger.info(f"Code generation attempt {attempt + 1}/{max_attempts}")

                # Exponential backoff for retries
                if attempt > 0:
                    wait_time = backoff_time * (2 ** (attempt - 1))
                    wait_time = min(wait_time, 30) # Cap wait time
                    logger.info(f"Rate limit or error encountered. Waiting {wait_time:.1f}s before retry.")
                    await asyncio.sleep(wait_time)

                # Get current conversation history
                conversation_history = self.conversation.get_conversation_history()

                # Call the appropriate model API
                api_response = {}
                if self.model_provider == ModelProvider.CLAUDE:
                    if not self.anthropic_client: raise ValueError("Claude provider selected but client not initialized.")
                    api_response = await self._call_anthropic_api(messages=conversation_history, system_prompt=system_prompt)
                # elif self.model_provider == ModelProvider.OLLAMA:
                #     # Simplified call - adapt if re-enabling Ollama
                #     ollama_response = await self._call_ollama_with_rate_limit(messages=conversation_history, temperature=0.1 + (attempt * 0.1))
                #     api_response = {"response": ollama_response.get('response', ''), "thinking": ""}
                elif self.model_provider == ModelProvider.GROQ:
                     if not self.openai_client: raise ValueError("Groq provider selected but client not initialized.")
                     api_response = await self._call_groq_api(messages=conversation_history, system_prompt=system_prompt)
                else:
                     raise ValueError(f"Unsupported model provider: {self.model_provider}")


                raw_llm_output = api_response.get('response', '').strip()
                thinking = api_response.get('thinking', '') # Get thinking if available

                if not raw_llm_output:
                     logger.warning(f"Attempt {attempt + 1} resulted in empty response from LLM.")
                     continue # Retry if response is empty

                logger.debug(f"Raw LLM Output (attempt {attempt+1}):\n{raw_llm_output[:500]}...") # Log snippet

                # --- Code Extraction ---
                # Look for ```python ... ``` blocks
                code_pattern = r"```python\s*([\s\S]*?)\s*```"
                code_matches = re.findall(code_pattern, raw_llm_output)
                extracted_code = ""

                if code_matches:
                    # Prioritize the last block if multiple exist, or the first if only one
                    extracted_code = code_matches[-1].strip()
                    logger.info(f"Extracted code block (length: {len(extracted_code)}).")
                    # Keep the explanation part if needed
                    explanation = re.sub(code_pattern, '', raw_llm_output).strip()
                else:
                    # If no block found, assume the entire response might be code (less ideal)
                    logger.warning("No ```python block found in LLM response. Assuming entire response is code.")
                    extracted_code = raw_llm_output
                    explanation = "" # No separate explanation

                if not extracted_code:
                    logger.warning(f"Attempt {attempt + 1} failed to extract any code content.")
                    # Add assistant message indicating failure to extract
                    self.conversation.add_assistant_message(
                        content=f"I received a response, but couldn't extract valid Python code from it. Response snippet: {raw_llm_output[:200]}...",
                        thinking=thinking
                    )
                    continue # Retry

                # --- Syntax Validation and Fixing ---
                is_valid, syntax_error = self.environment.validate_syntax(extracted_code)
                final_code = extracted_code

                if not is_valid and syntax_error:
                    logger.warning(f"Initial syntax validation failed: {syntax_error}")
                    # Try to fix *once*
                    fixed_code, fix_applied = await self._fix_specific_issues(extracted_code, syntax_error)
                    if fix_applied:
                        logger.info("Attempting to apply automated syntax fix.")
                        is_valid_after_fix, syntax_error_after_fix = self.environment.validate_syntax(fixed_code)
                        if is_valid_after_fix:
                            logger.info("Automated syntax fix successful.")
                            final_code = fixed_code
                            is_valid = True
                        else:
                            logger.warning(f"Automated syntax fix failed. Error after fix: {syntax_error_after_fix}")
                            # Stick with the original code with the error for the assistant message
                    else:
                         logger.warning("No automated fix applied for the syntax error.")


                # --- Process Result ---
                if is_valid:
                    logger.info(f"Successfully generated and validated code on attempt {attempt + 1}.")
                    generated_code = GeneratedCode(
                        code=final_code,
                        description=task_description, # Use the original task desc
                        language="python"
                        # TODO: Add logic to extract test cases and requirements if LLM provides them
                    )

                    # Perform post-processing (versioning, docs, etc.)
                    processing_results = self._process_successful_code(generated_code)
                    # Update generated_code object with results like hash, doc path
                    generated_code.version_hash = processing_results.get("version_hash")
                    generated_code.documentation = processing_results.get("documentation_path")
                    generated_code.metrics["diagram_path"] = processing_results.get("diagram_path")
                    generated_code.dependencies = list(self.dependency_manager.requirements) # Store detected deps


                    # Add successful assistant message to conversation
                    assistant_response_content = explanation if explanation else f"Here's the generated code for '{task_description[:50]}...'."
                    # Include the validated code in the message for Streamlit display
                    assistant_response_content += f"\n```python\n{final_code}\n```"

                    self.conversation.add_assistant_message(
                        content=assistant_response_content,
                        code=generated_code, # Associate the full GeneratedCode object
                        thinking=thinking
                    )
                    return generated_code # Return the successful code object

                else: # Syntax still invalid after potential fix
                    logger.warning(f"Attempt {attempt + 1} failed: Code remains syntactically invalid. Error: {syntax_error}")
                    # Add assistant message indicating the failure and the invalid code
                    self.conversation.add_assistant_message(
                        content=f"I tried to generate the code, but it has a syntax error: {syntax_error}\n```python\n{final_code}\n```", # Show the invalid code
                        thinking=thinking
                    )
                    # Continue to the next attempt

            except (ConnectionError, TimeoutError, SystemError, ValueError) as e: # Catch specific API/config errors
                logger.error(f"API or configuration error during generation attempt {attempt + 1}: {str(e)}", exc_info=True)
                # For UI, it might be better to fail fast on these errors
                if attempt == max_attempts - 1:
                    logger.critical("API/Config error persisted across retries. Aborting generation.")
                    self.conversation.add_assistant_message(f"Failed to generate code due to an API or configuration issue: {e}")
                    return self._handle_generation_failure("API/Configuration Error")
                # Continue retrying for transient issues like rate limits/connection errors
                if not isinstance(e, ValueError): # Don't retry ValueErrors (like missing keys)
                     continue
                else:
                      self.conversation.add_assistant_message(f"Configuration error prevented code generation: {e}")
                      return self._handle_generation_failure("Configuration Error")

            except Exception as e:
                logger.error(f"Unexpected error during generation attempt {attempt + 1}: {str(e)}", exc_info=True)
                # Log traceback for unexpected errors
                # traceback.print_exc() # Already logged via exc_info=True
                if attempt == max_attempts - 1:
                    logger.error("Maximum attempts reached after unexpected error. Returning fallback.")
                    self.conversation.add_assistant_message(f"An unexpected error occurred after {max_attempts} attempts: {e}")
                    return self._handle_generation_failure(f"Unexpected Error: {e}")
                continue # Retry after unexpected error

        # If loop finishes without returning (all attempts failed)
        logger.error(f"Failed to generate valid code after {max_attempts} attempts.")
        # Assistant message already added in the loop for the last failure
        return self._handle_generation_failure(f"Failed after {max_attempts} attempts")


    async def _fix_specific_issues(self, code: str, error: SyntaxError) -> tuple[str, bool]:
        """Attempt to fix common, specific syntax errors."""
        original_code = code
        fixed = False
        error_str = str(error).lower()
        lineno = error.lineno if error.lineno is not None else 0
        offset = error.offset if error.offset is not None else 0
        lines = code.splitlines()

        logger.debug(f"Attempting to fix syntax error: '{error_str}' at line {lineno}, offset {offset}")

        # 1. Unterminated string literal
        if "unterminated string literal" in error_str and 0 < lineno <= len(lines):
            line_index = lineno - 1
            error_line = lines[line_index]
            logger.debug(f"Fixing unterminated string on line: '{error_line}'")

            # Count quotes, handling escaped quotes simply might be tricky
            # Focus on simple cases: line ends mid-string
            open_single = error_line.count("'") % 2 != 0
            open_double = error_line.count('"') % 2 != 0
            # Basic check for triple quotes (might not be perfect)
            open_triple_single = error_line.count("'''") % 2 != 0 and not open_single
            open_triple_double = error_line.count('"""') % 2 != 0 and not open_double

            # Simplistic fix: add the missing quote at the end of the line
            if open_triple_double:
                lines[line_index] += '"""'
                fixed = True
            elif open_triple_single:
                lines[line_index] += "'''"
                fixed = True
            elif open_double:
                # Check if it's likely an f-string issue
                if error_line.lstrip().startswith('f"') and offset >= len(error_line):
                     lines[line_index] += '"'
                     fixed = True
                elif offset >= len(error_line): # Only add if error is at/after line end
                     lines[line_index] += '"'
                     fixed = True
            elif open_single:
                 if error_line.lstrip().startswith("f'") and offset >= len(error_line):
                      lines[line_index] += "'"
                      fixed = True
                 elif offset >= len(error_line):
                      lines[line_index] += "'"
                      fixed = True

            if fixed:
                 logger.info(f"Applied fix: Added closing quote to line {lineno}.")
                 code = '\n'.join(lines)


        # 2. IndentationError / TabError (Less safe to auto-fix, but maybe simple dedent)
        # if not fixed and ("indentationerror" in error_str or "taberror" in error_str):
        #     if "expected an indented block" in error_str and 0 < lineno < len(lines):
        #          # Maybe add pass to the next line? Risky.
        #          pass
        #     elif "unexpected indent" in error_str and 0 < lineno <= len(lines):
        #          line_index = lineno - 1
        #          original_line = lines[line_index]
        #          lines[line_index] = original_line.lstrip()
        #          if lines[line_index] != original_line:
        #              logger.info(f"Applied fix: Removed leading whitespace from line {lineno}.")
        #              code = '\n'.join(lines)
        #              fixed = True

        # 3. Missing colon (EOF while scanning) - Hard to fix reliably

        # Check if code actually changed
        code_changed = (code != original_code)
        if fixed and code_changed:
            logger.info("Code modified by automated fixer.")
            return code, True
        elif fixed and not code_changed:
             logger.warning("Fix was marked successful but code did not change.")
             return original_code, False
        else:
            logger.debug("No automated fix applied.")
            return original_code, False

    def _process_successful_code(self, code: GeneratedCode) -> Dict[str, Any]:
        """Post-generation workflow for valid code. Returns dict of results."""
        logger.info("Processing successful code generation...")
        results = {}

        # 1. Version Control
        try:
            version_hash = self.version_control.commit_version(code)
            if version_hash:
                results["version_hash"] = version_hash
                logger.info(f"Code committed with hash: {version_hash}")
            else:
                 logger.warning("Failed to commit code version.")
        except Exception as e:
             logger.error(f"Error during version control commit: {e}", exc_info=True)

        # 2. Dependency Analysis
        try:
            logger.debug("Extracting dependencies...")
            self.dependency_manager.requirements = set() # Clear previous
            # Simple regex for imports (could be enhanced with AST)
            # Standard library modules to exclude
            std_lib_modules = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()
            # Add common built-ins just in case
            std_lib_modules.update(["os", "sys", "re", "json", "time", "datetime", "typing", "math", "collections", "itertools", "functools", "logging", "subprocess", "tempfile", "pathlib"])

            imports = set()
            try:
                 tree = ast.parse(code.code)
                 for node in ast.walk(tree):
                     if isinstance(node, ast.Import):
                         for alias in node.names:
                              top_level = alias.name.split('.')[0]
                              if top_level and top_level not in std_lib_modules:
                                   imports.add(top_level)
                     elif isinstance(node, ast.ImportFrom):
                          if node.module and node.level == 0: # Absolute imports only
                               top_level = node.module.split('.')[0]
                               if top_level and top_level not in std_lib_modules:
                                    imports.add(top_level)
            except SyntaxError:
                 logger.warning("Could not parse code for dependency extraction due to syntax error (should not happen here).")


            logger.info(f"Potential dependencies found: {imports}")
            for pkg in imports:
                self.dependency_manager.add_requirement(pkg)

            # Generate requirements file
            self.dependency_manager.generate_requirements_file(self.version_control.repo_path)
            results["dependencies"] = sorted(list(self.dependency_manager.requirements))
        except Exception as e:
            logger.error(f"Error during dependency analysis: {e}", exc_info=True)

        # 3. Documentation Generation
        try:
            doc_path = self.documentation_gen.generate_html_docs(code, self.version_control.repo_path / "docs")
            if doc_path:
                results["documentation_path"] = doc_path
                logger.info(f"Documentation generated at: {doc_path}")
            else:
                 logger.warning("Documentation generation failed or produced no output.")
        except Exception as e:
             logger.error(f"Error during documentation generation: {e}", exc_info=True)

        # 4. Architecture Diagram (Optional, potentially slow/error-prone)
        try:
            # Disable diagram generation for speed/stability unless explicitly needed
            # diagram_path = self.generate_architecture_diagram(code)
            # if diagram_path:
            #     results["diagram_path"] = str(diagram_path) # Store as string
            #     logger.info(f"Architecture diagram generated: {diagram_path}")
            pass # Keep disabled for now
        except Exception as e:
            logger.error(f"Error during diagram generation: {e}", exc_info=True)

        return results

    def _handle_generation_failure(self, reason: str = "Unknown") -> GeneratedCode:
        """Handles generation failure, logs, commits fallback, returns fallback."""
        logger.error(f"Code generation failed: {reason}. Returning fallback code.")
        fallback = GeneratedCode.get_fallback_code()
        fallback.description = f"Fallback code generated due to failure: {reason}"
        try:
             self.version_control.commit_version(fallback, message=f"Failed generation snapshot: {reason[:100]}")
        except Exception as e:
             logger.error(f"Failed to commit fallback code: {e}", exc_info=True)
        return fallback

    async def improve_code(
        self,
        original_code_obj: GeneratedCode, # Pass the object
        analysis: Optional[CodeAnalysis] = None, # Make analysis optional
        max_attempts: int = 1 # Only try improvement once for UI speed
    ) -> Optional[GeneratedCode]: # Return Optional, might fail
        """Attempts to improve code based on analysis or general principles using the LLM."""
        logger.info(f"Attempting to improve code (hash: {original_code_obj.version_hash})...")

        # Construct a prompt for improvement
        improvement_prompt = "Please review the following Python code and improve it."
        if analysis and analysis.issues:
            improvement_prompt += "\nConsider these specific issues:\n" + "\n".join(f"- {issue}" for issue in analysis.issues)
        if analysis and analysis.suggestions:
             improvement_prompt += "\nConsider these suggestions:\n" + "\n".join(f"- {sug}" for sug in analysis.suggestions)
        improvement_prompt += "\n\nFocus on improving clarity, robustness, documentation, and adherence to PEP 8."
        improvement_prompt += "\nReturn *only* the improved Python code within a single ```python ... ``` block."
        improvement_prompt += f"\n\n```python\n{original_code_obj.code}\n```"

        # Use the generate_code logic, but with a different initial prompt
        # We need a temporary conversation context for this
        original_conversation = self.conversation.messages.copy() # Backup original convo
        self.conversation.clear() # Start fresh for improvement request
        self.conversation.add_user_message(improvement_prompt)

        try:
            # Use generate_code's retry logic etc.
            improved_code_obj = await self.generate_code(task_description="Improve the provided code.")

            # Check if the returned code is actually different and not fallback
            if improved_code_obj.code != original_code_obj.code and "Fallback code" not in improved_code_obj.description:
                 logger.info("Code improvement successful.")
                 # Restore original conversation *after* improvement attempt
                 self.conversation.messages = original_conversation
                 # Add a message about the improvement to the *original* conversation
                 self.conversation.add_assistant_message(
                      "I've attempted to improve the previous code. Here's the revised version:",
                      code=improved_code_obj
                 )
                 return improved_code_obj
            elif improved_code_obj.code == original_code_obj.code:
                 logger.info("Improvement attempt resulted in identical code.")
                 self.conversation.messages = original_conversation # Restore convo
                 self.conversation.add_assistant_message("I reviewed the code, but didn't find significant areas for improvement based on the request, or the changes were minimal.")
                 return None
            else: # Fallback code returned
                 logger.warning("Code improvement failed, fallback code was generated.")
                 self.conversation.messages = original_conversation # Restore convo
                 self.conversation.add_assistant_message("I encountered an error while trying to improve the code.")
                 return None

        except Exception as e:
             logger.error(f"Error during code improvement process: {e}", exc_info=True)
             self.conversation.messages = original_conversation # Restore convo
             self.conversation.add_assistant_message(f"An error occurred during the improvement attempt: {e}")
             return None


    def run_advanced_tests(self, code: GeneratedCode) -> Dict[str, Any]:
        """Runs comprehensive tests on the generated code - simplified version"""
        logger.info(f"Running advanced tests/analysis for code (hash: {code.version_hash})...")
        results: Dict[str, Any] = {
            "syntax_check": {"passed": False, "error": None},
            "static_analysis": {"issues": [], "metrics": {}},
            # "runtime_tests": [], # Runtime tests need specific test cases from LLM or user
            "security_checks": {"issues": []},
            # "performance_metrics": {} # Performance is complex to measure reliably here
        }

        # 1. Syntax Check (already done mostly, but double-check)
        valid, error = self.environment.validate_syntax(code.code)
        results["syntax_check"]["passed"] = valid
        results["syntax_check"]["error"] = str(error) if error else None
        if not valid:
            logger.warning("Advanced tests skipped: Code has syntax errors.")
            return results # Stop if syntax is wrong

        # 2. Static Analysis (Basic Metrics)
        try:
            tree = ast.parse(code.code)
            # Use pre-calculated metrics if available, otherwise calculate basic ones
            results["static_analysis"]["metrics"]["doc_coverage"] = code.metrics.get("doc_coverage", self._calculate_doc_coverage(tree))
            results["static_analysis"]["metrics"]["type_hint_coverage"] = code.metrics.get("type_hint_coverage", self._calculate_type_hint_coverage(tree))
            # Simplified complexity: count functions/classes
            func_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            results["static_analysis"]["metrics"]["function_count"] = func_count
            results["static_analysis"]["metrics"]["class_count"] = class_count
            # You could add pylint/flake8 integration here for more issues
            # Example: Run flake8 subprocess
            # flake8_issues = self._run_flake8(code.code)
            # results["static_analysis"]["issues"].extend(flake8_issues)

        except Exception as e:
            logger.error(f"Error during static analysis: {e}", exc_info=True)
            results["static_analysis"]["issues"].append(f"Analysis Error: {e}")

        # 3. Security Checks (Using Bandit)
        try:
             security_issues = self._check_security(code.code) # Uses Bandit
             results["security_checks"]["issues"] = security_issues
             if security_issues:
                 logger.warning(f"Bandit found {len(security_issues)} potential security issues.")
             else:
                  logger.info("Bandit security scan found no high/medium severity issues.")
        except Exception as e:
             logger.error(f"Error during security checks: {e}", exc_info=True)
             results["security_checks"]["issues"].append({"error": f"Security Scan Error: {e}"})

        # 4. Runtime tests (Placeholder - requires generated test cases)
        # test_outcomes = self.test_code(code) # Assumes code.test_cases is populated
        # results["runtime_tests"] = [r.model_dump() for r in test_outcomes] # Use model_dump for Pydantic v2

        logger.info("Advanced tests/analysis complete.")
        return results

    def _run_flake8(self, code: str) -> List[str]:
        """Runs flake8 static analyzer and returns issues."""
        issues = []
        try:
             with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding='utf-8') as f:
                 filepath = f.name
                 f.write(code)
                 f.flush() # Ensure written to disk

             logger.info(f"Running flake8 on {filepath}")
             result = subprocess.run(
                 [sys.executable, "-m", "flake8", filepath], # Use sys.executable to ensure correct env
                 capture_output=True,
                 text=True,
                 encoding='utf-8',
                 timeout=30
             )

             if result.stdout:
                  issues = result.stdout.strip().splitlines()
                  # Remove filepath prefix from issues for cleaner output
                  issues = [line.split(':', 1)[-1].strip() for line in issues if ':' in line]
                  logger.info(f"Flake8 found {len(issues)} issues.")

             if result.stderr:
                  logger.warning(f"Flake8 stderr: {result.stderr}")

        except FileNotFoundError:
             logger.warning("flake8 not found. Skipping flake8 analysis. (Install with: pip install flake8)")
             return ["Flake8 not found, analysis skipped."]
        except subprocess.TimeoutExpired:
              logger.warning("flake8 execution timed out.")
              return ["Flake8 execution timed out."]
        except Exception as e:
              logger.error(f"Flake8 execution failed: {e}", exc_info=True)
              return [f"Flake8 execution error: {e}"]
        finally:
             # Clean up temp file
             if 'filepath' in locals() and os.path.exists(filepath):
                  try:
                      os.unlink(filepath)
                  except OSError as e:
                       logger.warning(f"Could not delete temp file {filepath}: {e}")
        return issues

    # Simplified maintainability - complex metrics are hard to get right quickly
    # def _calculate_maintainability_index(self, code: str) -> float: ...

    def _calculate_doc_coverage(self, tree: ast.AST) -> float:
        """Calculate documentation coverage percentage"""
        total_documentable = 0
        documented = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                 total_documentable += 1
                 docstring = ast.get_docstring(node, clean=True) # Use clean=True
                 if docstring and docstring.strip(): # Check if non-empty after cleaning
                      documented += 1
        coverage = (documented / total_documentable * 100) if total_documentable > 0 else 100.0 # Assume 100% if nothing to document
        logger.debug(f"Doc coverage: {documented}/{total_documentable} = {coverage:.2f}%")
        return round(coverage, 2)


    def _calculate_type_hint_coverage(self, tree: ast.AST) -> float:
        """Calculate type hint coverage for function arguments and return types."""
        total_annotatable = 0
        annotated = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function arguments (excluding self/cls)
                args = node.args
                arg_list = args.args + args.posonlyargs + args.kwonlyargs
                for arg in arg_list:
                    if arg.arg not in ('self', 'cls'): # Exclude self/cls
                        total_annotatable += 1
                        if arg.annotation:
                            annotated += 1

                # Check return type
                # Only count if the function isn't explicitly marked as returning None implicitly (e.g. no return statement or only return None)
                # Basic check: does it have a non-None return statement?
                has_return_value = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node))
                if has_return_value or node.returns: # If it has annotation OR returns a value
                     total_annotatable += 1
                     if node.returns:
                         annotated += 1

        coverage = (annotated / total_annotatable * 100) if total_annotatable > 0 else 100.0 # Assume 100% if nothing to annotate
        logger.debug(f"Type hint coverage: {annotated}/{total_annotatable} = {coverage:.2f}%")
        return round(coverage, 2)

    # Removed _run_security_checks (replaced by _check_security using Bandit)
    # def _run_security_checks(self, code: str) -> List[Dict[str, str]]: ...

    # Removed _measure_performance - too unreliable in this context
    # def _measure_performance(self, code: str) -> Dict[str, float]: ...

    def test_code(self, code: GeneratedCode) -> List[CodeTestResult]:
        """Executes test cases provided within the GeneratedCode object."""
        results = []
        if not code.test_cases:
             logger.info("No test cases provided for execution.")
             return results
        if not code.code:
             logger.warning("Cannot run tests: generated code is empty.")
             return results

        logger.info(f"Running {len(code.test_cases)} provided test cases...")
        with self.environment.sandbox() as temp_dir:
            main_code_file = temp_dir / "generated_module.py"
            try:
                 main_code_file.write_text(code.code, encoding='utf-8')
                 logger.debug(f"Written generated code to {main_code_file}")
            except IOError as e:
                 logger.error(f"Failed to write generated code to {main_code_file}: {e}")
                 results.append(CodeTestResult(passed=False, error_message=f"Failed to write code file: {e}", execution_time=0.0))
                 return results # Cannot proceed

            for i, test_case_code in enumerate(code.test_cases):
                if not test_case_code or not test_case_code.strip():
                    logger.warning(f"Skipping empty test case {i+1}.")
                    continue

                test_file = temp_dir / f"test_case_{i+1}.py"
                start_time = time.monotonic()
                error_message: Optional[str] = None
                passed = False
                stdout = ""
                stderr = ""

                try:
                    # Prepend import of the generated code to the test case
                    full_test_code = f"import generated_module\n\n{test_case_code}"
                    test_file.write_text(full_test_code, encoding='utf-8')
                    logger.debug(f"Written test case {i+1} to {test_file}")

                    # Execute the test case file as a script
                    result = subprocess.run(
                        [sys.executable, str(test_file)],
                        capture_output=True,
                        text=True,
                        encoding='utf-8', # Ensure consistent encoding
                        timeout=15, # Increased timeout for potentially complex tests
                        cwd=temp_dir # Run from the temp dir so imports work
                    )
                    stdout = result.stdout
                    stderr = result.stderr
                    passed = result.returncode == 0

                    if not passed:
                        error_message = f"Test exited with code {result.returncode}."
                        if stderr: # Append stderr if present
                            error_message += f"\nStderr:\n{stderr.strip()}"
                        logger.warning(f"Test case {i+1} failed. RC={result.returncode}, Stderr: {stderr.strip()}")
                    else:
                         logger.info(f"Test case {i+1} passed.")


                except subprocess.TimeoutExpired:
                    passed = False
                    error_message = "Test execution timed out (> 15 seconds)."
                    logger.warning(f"Test case {i+1} timed out.")
                except IOError as e:
                     passed = False
                     error_message = f"Failed to write test case file: {e}"
                     logger.error(f"Failed to write test case {i+1} to {test_file}: {e}")
                except Exception as e:
                    passed = False
                    error_message = f"Unexpected error during test execution: {str(e)}"
                    logger.error(f"Unexpected error running test case {i+1}: {e}", exc_info=True)

                execution_time = time.monotonic() - start_time
                results.append(CodeTestResult(
                    passed=passed,
                    error_message=error_message,
                    execution_time=round(execution_time, 3),
                    # memory_usage=None # Memory usage is hard to get reliably from subprocess
                ))
                # Clean up test file immediately? Or let sandbox handle it. Let sandbox handle.

        return results

    # Removed display_code - Streamlit handles display
    # def display_code(self, code: GeneratedCode): ...

    def generate_architecture_diagram(self, code: GeneratedCode) -> Optional[str]:
        """Generate UML diagram using pyreverse"""
        if not code or not code.code.strip():
            logger.warning("Cannot generate diagram for empty code.")
            return None

        output_dir = self.version_control.repo_path / "diagrams"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Attempting to generate architecture diagram in: {output_dir}")

        # Use a temporary file within the sandbox
        code_env = CodeEnvironment()
        try:
            with code_env.sandbox() as temp_sandbox_dir:
                temp_module_path = temp_sandbox_dir / "temp_module_for_diagram.py"
                temp_module_path.write_text(code.code, encoding='utf-8')
                logger.debug(f"Writing code for diagram to: {temp_module_path}")

                # Define expected output path (pyreverse uses project name)
                project_name = "GeneratedCodeDiagram"
                # Common diagram generated by default
                expected_diagram_path = output_dir / f"classes_{project_name}.png"

                # Run pyreverse
                # Use -A to analyze all modules found starting from the path
                cmd = ["pyreverse", "-A", "-o", "png", "-p", project_name, str(temp_module_path)]
                # Running inside the sandbox dir might help resolve imports if needed
                logger.info(f"Running pyreverse command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=60, # Timeout for analysis
                    cwd=temp_sandbox_dir # Run from the directory containing the temp module
                )

                logger.debug(f"Pyreverse stdout:\n{result.stdout}")
                logger.debug(f"Pyreverse stderr:\n{result.stderr}")

                # Move generated diagram (if any) to the desired output dir
                diagram_found = None
                # Look for common diagram types in the *current* directory (where pyreverse ran)
                possible_diagrams = list(temp_sandbox_dir.glob(f"{project_name}.png")) \
                                    + list(temp_sandbox_dir.glob(f"classes_{project_name}.png")) \
                                    + list(temp_sandbox_dir.glob(f"packages_{project_name}.png"))

                if possible_diagrams:
                     # Prefer classes diagram
                     source_path = next((p for p in possible_diagrams if "classes_" in p.name), possible_diagrams[0])
                     target_path = output_dir / source_path.name
                     try:
                         source_path.rename(target_path) # Move the file
                         diagram_found = str(target_path)
                         logger.info(f"Diagram '{source_path.name}' moved to {target_path}")
                     except OSError as e:
                          logger.error(f"Failed to move diagram from {source_path} to {target_path}: {e}")


                if diagram_found:
                    logger.info(f"Diagram generated successfully: {diagram_found}")
                    return diagram_found
                else:
                    logger.warning(f"Pyreverse ran (RC={result.returncode}) but no expected diagram file found in {temp_sandbox_dir}.")
                    if result.returncode != 0:
                         logger.error(f"Pyreverse failed. Stderr: {result.stderr}")
                    return None

        except FileNotFoundError:
            logger.error("pyreverse command not found. Ensure pylint is installed (pip install pylint) and in PATH.")
            return None
        except subprocess.TimeoutExpired:
             logger.error("Pyreverse execution timed out.")
             return None
        except Exception as e:
            logger.error(f"Diagram generation failed: {str(e)}", exc_info=True)
            return None


    def save_feedback(self, feedback: str):
        """Store user feedback for future improvements"""
        if not feedback or not feedback.strip():
            return
        feedback_file = self.version_control.repo_path / "feedback.log"
        timestamp = datetime.now().isoformat()
        log_entry = f"--- {timestamp} ---\n{feedback.strip()}\n\n"
        try:
            with open(feedback_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
            logger.info(f"User feedback saved to {feedback_file}")
        except IOError as e:
            logger.error(f"Failed to save feedback to {feedback_file}: {e}")


    def _check_security(self, code: str) -> List[dict]:
        """Enhanced security checks using bandit"""
        results = []
        if not code.strip():
            return results # Skip empty code

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding='utf-8') as f:
                filepath = f.name
                f.write(code)
                f.flush()

            logger.info(f"Running bandit security scan on {filepath}")
            # Run bandit with JSON output, specifying confidence and severity levels
            cmd = [
                sys.executable, "-m", "bandit", # Use sys.executable
                "-f", "json",
                "-r", filepath, # Scan recursively (though it's one file)
                "-c", "bandit.yaml" # Optional: Use a config file if needed
                #"-lll", # Severity level: Low (LLL), Medium (LL), High (L) - Report all
                #"-iii" # Confidence level: Low (III), Medium (II), High (I) - Report all
            ]
            # For simplicity, run with defaults (usually Medium+ severity/confidence)
            result = subprocess.run(
                 cmd,
                 capture_output=True,
                 text=True,
                 encoding='utf-8',
                 timeout=60
            )

            # Bandit exits with 1 if issues found, 0 if none, >1 on error
            if result.returncode in [0, 1]:
                try:
                    bandit_output = json.loads(result.stdout)
                    raw_results = bandit_output.get("results", [])
                    # Format results slightly
                    for issue in raw_results:
                         results.append({
                              "severity": issue.get("issue_severity", "UNKNOWN"),
                              "confidence": issue.get("issue_confidence", "UNKNOWN"),
                              "text": issue.get("issue_text", "No description"),
                              "file": os.path.basename(issue.get("filename", "")), # Show just filename
                              "line": issue.get("line_number", "?"),
                              "code": issue.get("code", "").strip() # Show relevant code snippet
                         })

                    logger.info(f"Bandit scan complete. Found {len(results)} issues (considering defaults).")
                    if result.stderr: # Log warnings/errors from bandit itself
                         logger.warning(f"Bandit stderr: {result.stderr}")

                except json.JSONDecodeError:
                    logger.error(f"Failed to parse bandit JSON output. Raw output: {result.stdout[:500]}...")
                    results.append({"error": "Failed to parse Bandit JSON output"})
                except Exception as e:
                     logger.error(f"Error processing Bandit results: {e}", exc_info=True)
                     results.append({"error": f"Error processing Bandit results: {e}"})

            else: # Bandit execution error
                 logger.error(f"Bandit execution failed (return code {result.returncode}). Stderr: {result.stderr}")
                 results.append({"error": f"Bandit execution failed (RC={result.returncode})", "details": result.stderr})

        except FileNotFoundError:
            logger.warning("bandit command not found. Skipping security scan. (Install with: pip install bandit)")
            return [{"error": "Bandit not installed, security scan skipped."}]
        except subprocess.TimeoutExpired:
             logger.warning("Bandit execution timed out.")
             return [{"error": "Bandit execution timed out."}]
        except Exception as e:
            logger.error(f"Security scan failed unexpectedly: {str(e)}", exc_info=True)
            return [{"error": f"Unexpected security scan error: {e}"}]
        finally:
             # Clean up temp file
             if 'filepath' in locals() and os.path.exists(filepath):
                  try:
                      os.unlink(filepath)
                  except OSError as e:
                       logger.warning(f"Could not delete temp file {filepath}: {e}")

        return results


# --- Streamlit UI ---

st.set_page_config(page_title="AI Code Assistant", layout="wide")

st.title("🤖 AI Code Assistant")
st.caption("Generate, test, and version control code using AI.")

# Initialize session state
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "messages" not in st.session_state:
    st.session_state.messages = [] # For displaying chat history
if "current_code" not in st.session_state:
    st.session_state.current_code = None # Stores the latest GeneratedCode object
if "test_results" not in st.session_state:
    st.session_state.test_results = None # Stores results from run_advanced_tests
if "model_provider" not in st.session_state:
     # Default model from environment or fallback
     default_provider_str = os.environ.get("MODEL_CHOICE", "claude").lower()
     try:
          st.session_state.model_provider = ModelProvider(default_provider_str)
     except ValueError:
          st.session_state.model_provider = ModelProvider.CLAUDE # Fallback
          logger.warning(f"Invalid MODEL_CHOICE '{default_provider_str}', defaulting to {ModelProvider.CLAUDE.value}")


# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # Model Selection
    model_options = [p.value for p in ModelProvider]
    selected_model_str = st.selectbox(
        "Select AI Model Provider",
        options=model_options,
        index=model_options.index(st.session_state.model_provider.value),
        key="model_select",
        help="Choose the AI model to use for code generation. API keys must be set in .env file."
    )
    selected_model = ModelProvider(selected_model_str)

    # Re-initialize assistant if model changes or not initialized
    if st.session_state.assistant is None or selected_model != st.session_state.model_provider:
        try:
            with st.spinner(f"Initializing assistant with {selected_model.value}..."):
                 st.session_state.assistant = AICodeAssistant(model_provider=selected_model)
            st.session_state.model_provider = selected_model
            st.session_state.messages = [] # Clear messages on model change
            st.session_state.current_code = None
            st.session_state.test_results = None
            st.success(f"Assistant initialized with {selected_model.value}.")
             # Rerun to update the UI state cleanly
        except (ValueError, ConnectionError, Exception) as e:
             st.error(f"Failed to initialize {selected_model.value}: {e}", icon="🚨")
             st.session_state.assistant = None # Ensure assistant is None if init fails
             # Attempt to fallback maybe? Or just show error. Showing error is clearer.

    st.divider()

    # New Conversation Button
    if st.button("🔄️ Start New Conversation"):
        logger.info("Starting new conversation via Streamlit button.")
        if st.session_state.assistant:
            st.session_state.assistant.conversation.clear()
        st.session_state.messages = []
        st.session_state.current_code = None
        st.session_state.test_results = None
        st.success("New conversation started.")
        

    st.divider()

    # Display Info
    if st.session_state.assistant:
        st.info(f"Using: **{st.session_state.model_provider.value.upper()}**")
        try:
             # Display basic repo status
             repo = st.session_state.assistant.version_control.repo
             if repo:
                 commits = list(repo.iter_commits('main', max_count=1))
                 st.write(f"**Repo:** `{st.session_state.assistant.version_control.repo_path.name}`")
                 if commits:
                      st.write(f"**Last Commit:** `{commits[0].hexsha[:7]}`")
                 else:
                      st.write("**Repo Status:** Initialized, no commits yet.")
             else:
                  st.warning("Repo not initialized.")
        except Exception as e:
             st.warning(f"Could not get repo status: {e}")
    else:
        st.warning("Assistant not initialized. Check API keys and model selection.")

# --- Main Chat Area ---

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
         # Display text content
         st.markdown(message["content"])
         # If assistant message has associated code object, display details from it
         if message["role"] == "assistant" and message.get("code_hash"):
             # Find the code object in the assistant's history (less efficient but needed)
             code_obj = next((c for c in st.session_state.assistant.conversation.generated_codes if getattr(c, 'version_hash', None) == message["code_hash"]), None)
             if code_obj:
                 with st.expander("Generated Code Details", expanded=False):
                     st.code(code_obj.code, language="python")
                     if code_obj.version_hash:
                          st.caption(f"Version Hash: {code_obj.version_hash}")


# Handle user input
if prompt := st.chat_input("Enter your code generation task or follow-up request...", disabled=(st.session_state.assistant is None)):
    logger.info(f"Streamlit received user input: {prompt[:100]}...")
    # Add user message to Streamlit state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate code using the assistant
    if st.session_state.assistant:
        try:
            with st.spinner("Generating code... Please wait..."):
                # Run the async function
                generated_code_object = asyncio.run(st.session_state.assistant.generate_code(prompt))
                st.session_state.current_code = generated_code_object # Store the latest code object

            # Display assistant's response (which includes the code block from generate_code)
            # The assistant's message should already be in its internal conversation history
            last_assistant_message = st.session_state.assistant.conversation.get_conversation_history()[-1]

            # Add assistant message to Streamlit state for display
            # Include a reference to the code hash for potentially linking later if needed
            msg_to_display = {
                "role": "assistant",
                "content": last_assistant_message["content"], # Content now includes the formatted code block
                "code_hash": getattr(generated_code_object, 'version_hash', None) # Store hash if available
            }
            st.session_state.messages.append(msg_to_display)

            # Rerun immediately to show the assistant's response text and code
            

        except (ValueError, ConnectionError, TimeoutError, SystemError, Exception) as e:
            logger.error(f"Streamlit code generation failed: {e}", exc_info=True)
            st.error(f"Code generation failed: {e}", icon="🚨")
            # Add error message to chat
            error_msg = {"role": "assistant", "content": f"Sorry, I encountered an error: {e}"}
            st.session_state.messages.append(error_msg)
             # Rerun to show the error message

# --- Display Results Below Chat ---
if st.session_state.current_code and "Fallback code" not in st.session_state.current_code.description:
     st.divider()
     st.subheader("Latest Generation Details")

     code_obj = st.session_state.current_code

     # Use tabs for better organization
     tab_code, tab_analysis, tab_files, tab_feedback = st.tabs([
          "📄 Code", "📊 Analysis", "🗂️ Files & History", "💬 Feedback"
      ])

     with tab_code:
          st.markdown(f"**Description:** {code_obj.description}")
          st.code(code_obj.code, language=code_obj.language)

          # Display test cases if any
          if code_obj.test_cases:
                with st.expander("Provided Test Cases"):
                    for i, test in enumerate(code_obj.test_cases):
                         st.code(test, language="python", key=f"test_{i}")

          # Option to run provided tests
          # if code_obj.test_cases:
          #     if st.button("Run Provided Tests"):
          #         with st.spinner("Running tests..."):
          #             test_run_results = st.session_state.assistant.test_code(code_obj)
          #             st.session_state.runtime_test_results = test_run_results # Store separately maybe
          #             # Display results immediately or store and display in analysis tab? Store and display.
          #             # This requires modifying run_advanced_tests or having a separate display logic
          #          # Rerun to potentially show results in analysis tab


     with tab_analysis:
         if st.button("🔬 Run Analysis & Security Scan"):
              with st.spinner("Running analysis..."):
                   st.session_state.test_results = st.session_state.assistant.run_advanced_tests(code_obj)
               # Rerun to display results

         if st.session_state.test_results:
              results = st.session_state.test_results
              st.markdown("**Syntax Check:**")
              if results["syntax_check"]["passed"]:
                  st.success("✅ Syntax Valid")
              else:
                  st.error(f"❌ Syntax Invalid: {results['syntax_check']['error']}")

              st.markdown("**Static Analysis Metrics:**")
              metrics = results["static_analysis"]["metrics"]
              cols = st.columns(3)
              cols[0].metric("Doc Coverage", f"{metrics.get('doc_coverage', 0):.1f}%")
              cols[1].metric("Type Hint Coverage", f"{metrics.get('type_hint_coverage', 0):.1f}%")
              cols[2].metric("Functions/Classes", f"{metrics.get('function_count', 0)}/{metrics.get('class_count', 0)}")

              # Display flake8/pylint issues if analyzer was run
              static_issues = results["static_analysis"].get("issues", [])
              if static_issues:
                   with st.expander("Static Analysis Issues"):
                       for issue in static_issues:
                           st.warning(issue)

              st.markdown("**Security Scan (Bandit):**")
              security_issues = results["security_checks"]["issues"]
              if not security_issues:
                   st.success("✅ No high/medium issues found.")
              else:
                   is_error = any("error" in issue for issue in security_issues)
                   if is_error:
                        st.error("Error during security scan.")
                        st.json(security_issues) # Show raw error
                   else:
                        st.warning(f"⚠️ Found {len(security_issues)} potential issue(s).")
                        with st.expander("Security Issues Details"):
                             # Provide a more structured view
                             for issue in security_issues:
                                  sev = issue.get('severity', 'LOW')
                                  conf = issue.get('confidence', 'LOW')
                                  color = "red" if sev == "HIGH" else ("orange" if sev == "MEDIUM" else "blue")
                                  st.markdown(f"<span style='color:{color};'>**[{sev}/{conf}]** {issue.get('text', 'No description')} (Line: {issue.get('line', '?')})</span>", unsafe_allow_html=True)
                                  # Optionally show code snippet
                                  if issue.get('code'):
                                       st.code(issue.get('code'), language='python', line_numbers=False)
         else:
              st.info("Click 'Run Analysis & Security Scan' to view results.")


     with tab_files:
            st.markdown("**Versioning:**")
            if code_obj.version_hash:
                st.success(f"✅ Committed as: `{code_obj.version_hash}`")
                # Show recent history
                try:
                     repo = st.session_state.assistant.version_control.repo
                     commits = list(repo.iter_commits('main', max_count=5))
                     with st.expander("Recent Commit History (main branch)"):
                          for commit in commits:
                               st.markdown(f"- `{commit.hexsha[:7]}`: {commit.summary} (*{commit.authored_datetime.strftime('%Y-%m-%d %H:%M')}*)")
                except Exception as e:
                     st.warning(f"Could not retrieve commit history: {e}")
            else:
                st.warning("Code not committed (or commit failed).")

            st.markdown("**Dependencies:**")
            deps = code_obj.dependencies
            if deps:
                 st.code("\n".join(deps), language="text")
                 req_path = st.session_state.assistant.version_control.repo_path / "requirements.txt"
                 st.caption(f"Saved to: `{req_path}`")
            else:
                 st.info("No external dependencies detected.")

            st.markdown("**Documentation:**")
            doc_path_str = code_obj.documentation
            if doc_path_str and Path(doc_path_str).exists():
                 st.success(f"✅ Generated: `{doc_path_str}`")
                 # Provide link/button to open? (Might be tricky depending on deployment)
            elif doc_path_str:
                 st.warning(f"Documentation path recorded, but file not found: {doc_path_str}")
            else:
                 st.info("Documentation not generated or failed.")

            # Diagram (if generated)
            # diag_path = code_obj.metrics.get("diagram_path")
            # if diag_path and Path(diag_path).exists():
            #     st.markdown("**Architecture Diagram:**")
            #     st.success(f"✅ Generated: `{diag_path}`")
            #     # st.image(str(diag_path)) # Display diagram if possible
            # elif diag_path:
            #      st.warning(f"Diagram path recorded, but file not found: {diag_path}")


     with tab_feedback:
            st.markdown("Provide feedback on the generated code:")
            feedback_text = st.text_area("Your feedback helps improve future results.", key=f"feedback_{code_obj.version_hash}")
            if st.button("Submit Feedback", key=f"submit_feedback_{code_obj.version_hash}"):
                if feedback_text.strip():
                    st.session_state.assistant.save_feedback(feedback_text)
                    st.success("Feedback submitted! Thank you.")
                    # Clear the text area after submission? Need to handle state carefully.
                else:
                    st.warning("Please enter some feedback before submitting.")

            # Option to Improve Code (Experimental)
            st.divider()
            st.markdown("**Code Improvement (Experimental):**")
            if st.button("✨ Attempt to Improve Code"):
                analysis_for_improvement = None
                if st.session_state.test_results:
                    # Basic analysis based on results
                    analysis_for_improvement = CodeAnalysis(
                        quality=CodeQuality.NEEDS_IMPROVEMENT, # Assume improvement needed if button clicked
                        issues=[f"Syntax Error: {st.session_state.test_results['syntax_check']['error']}" ] if not st.session_state.test_results['syntax_check']['passed'] else [],
                        suggestions=["Improve documentation coverage.", "Check for potential security issues.", "Simplify complex logic if possible."],
                        documentation_coverage=st.session_state.test_results['static_analysis']['metrics'].get('doc_coverage', 0),
                        security_issues=st.session_state.test_results['security_checks']['issues']
                    )

                with st.spinner("Attempting to improve the code..."):
                    improved_code = asyncio.run(st.session_state.assistant.improve_code(code_obj, analysis=analysis_for_improvement))

                if improved_code:
                    st.success("Code improvement generated!")
                    st.session_state.current_code = improved_code # Update current code
                    st.session_state.test_results = None # Clear old results
                    # Add message to chat - handled inside improve_code now
                    
                else:
                    st.info("Improvement attempt did not produce a new version or failed.")

elif st.session_state.current_code and "Fallback code" in st.session_state.current_code.description:
     st.divider()
     st.error("🚨 Code generation failed. Displaying fallback code.")
     st.code(st.session_state.current_code.code, language="python")
     st.markdown(f"**Reason:** {st.session_state.current_code.description}")

# Initial message if conversation is empty
if not st.session_state.messages and st.session_state.assistant:
    st.info("Enter a task description above to start generating code.")
elif not st.session_state.assistant:
     st.warning("Assistant failed to initialize. Please check configuration and API keys in the sidebar and refresh.")