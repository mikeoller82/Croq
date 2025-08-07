"""
Advanced Codebase Search and Context Extraction System
Provides intelligent code analysis, search, and context gathering
"""
import asyncio
import ast
import hashlib
import json
import logging
import os
import re
import subprocess
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, NamedTuple
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import gitignore_parser
from pygments import highlight
from pygments.lexers import get_lexer_for_filename, TextLexer
from pygments.formatters import NullFormatter
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class FileInfo:
    """Information about a file in the codebase"""
    path: Path
    size: int
    modified: datetime
    language: str
    encoding: str = "utf-8"
    lines: int = 0
    hash: str = ""
    
@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, variable, etc.)"""
    name: str
    type: str  # function, class, variable, constant, etc.
    file_path: Path
    line_number: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    complexity: int = 0
    references: List[Tuple[Path, int]] = field(default_factory=list)

@dataclass
class SearchResult:
    """Result from codebase search"""
    file_path: Path
    line_number: int
    line_content: str
    match_text: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    symbol: Optional[CodeSymbol] = None
    relevance_score: float = 1.0

@dataclass
class CodebaseStats:
    """Statistics about the codebase"""
    total_files: int = 0
    total_lines: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    file_types: Dict[str, int] = field(default_factory=dict)
    symbols: Dict[str, int] = field(default_factory=dict)
    complexity_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class LanguageAnalyzer:
    """Language-specific code analysis"""
    
    EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.sql': 'sql',
        '.sh': 'bash',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.md': 'markdown',
        '.txt': 'text'
    }
    
    @classmethod
    def get_language(cls, file_path: Path) -> str:
        """Determine language from file extension"""
        return cls.EXTENSIONS.get(file_path.suffix.lower(), 'unknown')
    
    @classmethod
    def analyze_python_file(cls, content: str, file_path: Path) -> List[CodeSymbol]:
        """Analyze Python file for symbols"""
        symbols = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.append(CodeSymbol(
                        name=node.name,
                        type='function',
                        file_path=file_path,
                        line_number=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        signature=cls._get_function_signature(node),
                        docstring=ast.get_docstring(node) or "",
                        complexity=cls._calculate_complexity(node)
                    ))
                elif isinstance(node, ast.ClassDef):
                    symbols.append(CodeSymbol(
                        name=node.name,
                        type='class',
                        file_path=file_path,
                        line_number=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node) or "",
                        complexity=cls._calculate_complexity(node)
                    ))
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            symbols.append(CodeSymbol(
                                name=target.id,
                                type='variable',
                                file_path=file_path,
                                line_number=node.lineno,
                                end_line=node.lineno
                            ))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing Python file {file_path}: {e}")
        
        return symbols
    
    @classmethod
    def _get_function_signature(cls, node: ast.FunctionDef) -> str:
        """Extract function signature from AST node"""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        signature = f"{node.name}({', '.join(args)})"
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"
        
        return signature
    
    @classmethod
    def _calculate_complexity(cls, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

class CodebaseIndex:
    """Maintains an index of codebase symbols and content"""
    
    def __init__(self, db_path: str = ".croq_index.db"):
        self.db_path = db_path
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for indexing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                size INTEGER,
                modified TIMESTAMP,
                language TEXT,
                hash TEXT,
                lines INTEGER,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Symbols table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                file_id INTEGER,
                line_number INTEGER,
                end_line INTEGER,
                signature TEXT,
                docstring TEXT,
                complexity INTEGER DEFAULT 0,
                FOREIGN KEY (file_id) REFERENCES files(id)
            )
        """)
        
        # Search index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_index (
                id INTEGER PRIMARY KEY,
                file_id INTEGER,
                line_number INTEGER,
                content TEXT,
                tokens TEXT,
                FOREIGN KEY (file_id) REFERENCES files(id)
            )
        """)
        
        # Create indexes for better search performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_tokens ON search_index(tokens)")
        
        conn.commit()
        conn.close()
    
    def should_index_file(self, file_path: Path) -> bool:
        """Check if file should be indexed"""
        # Skip binary files, hidden files, and common ignore patterns
        if file_path.name.startswith('.'):
            return False
        
        # Skip common binary extensions
        binary_extensions = {
            '.exe', '.bin', '.so', '.dll', '.dylib', '.a', '.o', '.obj',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'
        }
        
        if file_path.suffix.lower() in binary_extensions:
            return False
        
        # Skip files that are too large (>1MB)
        try:
            if file_path.stat().st_size > 1024 * 1024:
                return False
        except (OSError, IOError):
            return False
        
        return True
    
    async def index_file(self, file_path: Path) -> bool:
        """Index a single file"""
        if not self.should_index_file(file_path):
            return False
        
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Calculate file hash
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check if file is already indexed and up to date
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, hash FROM files WHERE path = ?
            """, (str(file_path),))
            
            result = cursor.fetchone()
            if result and result[1] == file_hash:
                conn.close()
                return True  # Already up to date
            
            # Get file info
            stat = file_path.stat()
            file_info = FileInfo(
                path=file_path,
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime),
                language=LanguageAnalyzer.get_language(file_path),
                hash=file_hash,
                lines=len(content.splitlines())
            )
            
            # Insert or update file record
            if result:
                cursor.execute("""
                    UPDATE files SET size=?, modified=?, language=?, hash=?, lines=?, indexed_at=CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (file_info.size, file_info.modified, file_info.language, file_info.hash, file_info.lines, result[0]))
                file_id = result[0]
                # Clear old symbols and search index
                cursor.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
                cursor.execute("DELETE FROM search_index WHERE file_id = ?", (file_id,))
            else:
                cursor.execute("""
                    INSERT INTO files (path, size, modified, language, hash, lines)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (str(file_path), file_info.size, file_info.modified, file_info.language, file_info.hash, file_info.lines))
                file_id = cursor.lastrowid
            
            # Analyze symbols based on language
            symbols = []
            if file_info.language == 'python':
                symbols = LanguageAnalyzer.analyze_python_file(content, file_path)
            
            # Insert symbols
            for symbol in symbols:
                cursor.execute("""
                    INSERT INTO symbols (name, type, file_id, line_number, end_line, signature, docstring, complexity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol.name, symbol.type, file_id, symbol.line_number, symbol.end_line,
                      symbol.signature, symbol.docstring, symbol.complexity))
            
            # Index content for search
            lines = content.splitlines()
            for line_num, line in enumerate(lines, 1):
                if line.strip():  # Skip empty lines
                    tokens = self._tokenize_for_search(line)
                    cursor.execute("""
                        INSERT INTO search_index (file_id, line_number, content, tokens)
                        VALUES (?, ?, ?, ?)
                    """, (file_id, line_num, line, ' '.join(tokens)))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Indexed {file_path} with {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return False
    
    def _tokenize_for_search(self, text: str) -> List[str]:
        """Tokenize text for search indexing"""
        # Simple tokenization - split on non-alphanumeric chars and lowercase
        tokens = re.findall(r'\w+', text.lower())
        return [token for token in tokens if len(token) > 2]  # Filter short tokens
    
    async def index_directory(self, directory: Path, ignore_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Index all files in a directory"""
        if not directory.exists() or not directory.is_dir():
            return {"error": "Directory does not exist"}
        
        # Parse .gitignore if present
        gitignore_path = directory / '.gitignore'
        gitignore_func = None
        if gitignore_path.exists():
            try:
                gitignore_func = gitignore_parser.parse_gitignore(gitignore_path)
            except Exception as e:
                logger.warning(f"Error parsing .gitignore: {e}")
        
        # Collect files to index
        files_to_index = []
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Check gitignore
            if gitignore_func and gitignore_func(str(file_path.relative_to(directory))):
                continue
            
            # Check custom ignore patterns
            if ignore_patterns:
                relative_path = str(file_path.relative_to(directory))
                if any(re.search(pattern, relative_path) for pattern in ignore_patterns):
                    continue
            
            if self.should_index_file(file_path):
                files_to_index.append(file_path)
        
        # Index files concurrently
        start_time = datetime.now()
        indexed_count = 0
        failed_count = 0
        
        # Process in batches to avoid overwhelming the system
        batch_size = 10
        for i in range(0, len(files_to_index), batch_size):
            batch = files_to_index[i:i + batch_size]
            tasks = [self.index_file(file_path) for file_path in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                elif result:
                    indexed_count += 1
                else:
                    failed_count += 1
        
        duration = datetime.now() - start_time
        
        # Update statistics
        stats = self.get_codebase_stats()
        
        return {
            "indexed_files": indexed_count,
            "failed_files": failed_count,
            "total_files": len(files_to_index),
            "duration_seconds": duration.total_seconds(),
            "stats": stats
        }
    
    def search(self, query: str, max_results: int = 50, file_pattern: str = None) -> List[SearchResult]:
        """Search the codebase index"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tokenize query
        query_tokens = self._tokenize_for_search(query)
        
        # Build search query
        search_conditions = []
        params = []
        
        for token in query_tokens:
            search_conditions.append("search_index.tokens LIKE ?")
            params.append(f"%{token}%")
        
        where_clause = " AND ".join(search_conditions) if search_conditions else "1=1"
        
        # Add file pattern filter if specified
        if file_pattern:
            where_clause += " AND files.path LIKE ?"
            params.append(f"%{file_pattern}%")
        
        sql = f"""
            SELECT files.path, search_index.line_number, search_index.content,
                   files.language, files.id
            FROM search_index
            JOIN files ON search_index.file_id = files.id
            WHERE {where_clause}
            ORDER BY files.path, search_index.line_number
            LIMIT ?
        """
        params.append(max_results)
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            file_path, line_number, content, language, file_id = row
            
            # Calculate relevance score based on query match
            relevance = self._calculate_relevance(query, content, query_tokens)
            
            # Get context lines
            context_before, context_after = self._get_context_lines(cursor, file_id, line_number)
            
            # Find matching symbol if applicable
            symbol = self._find_symbol_at_line(cursor, file_id, line_number)
            
            result = SearchResult(
                file_path=Path(file_path),
                line_number=line_number,
                line_content=content,
                match_text=query,
                context_before=context_before,
                context_after=context_after,
                symbol=symbol,
                relevance_score=relevance
            )
            results.append(result)
        
        conn.close()
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    def search_symbols(self, query: str, symbol_type: str = None) -> List[CodeSymbol]:
        """Search for symbols (functions, classes, etc.)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        conditions = ["symbols.name LIKE ?"]
        params = [f"%{query}%"]
        
        if symbol_type:
            conditions.append("symbols.type = ?")
            params.append(symbol_type)
        
        where_clause = " AND ".join(conditions)
        
        cursor.execute(f"""
            SELECT symbols.name, symbols.type, files.path, symbols.line_number,
                   symbols.end_line, symbols.signature, symbols.docstring, symbols.complexity
            FROM symbols
            JOIN files ON symbols.file_id = files.id
            WHERE {where_clause}
            ORDER BY symbols.name
        """, params)
        
        symbols = []
        for row in cursor.fetchall():
            symbol = CodeSymbol(
                name=row[0],
                type=row[1],
                file_path=Path(row[2]),
                line_number=row[3],
                end_line=row[4],
                signature=row[5] or "",
                docstring=row[6] or "",
                complexity=row[7]
            )
            symbols.append(symbol)
        
        conn.close()
        return symbols
    
    def _calculate_relevance(self, query: str, content: str, query_tokens: List[str]) -> float:
        """Calculate relevance score for search result"""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Exact match gets highest score
        if query_lower in content_lower:
            return 1.0
        
        # Token match scoring
        token_matches = sum(1 for token in query_tokens if token in content_lower)
        token_score = token_matches / len(query_tokens) if query_tokens else 0
        
        # Word boundary matches get higher scores
        word_matches = len(re.findall(rf'\b{re.escape(query_lower)}\b', content_lower))
        word_score = min(word_matches * 0.5, 0.8)
        
        return max(token_score, word_score)
    
    def _get_context_lines(self, cursor, file_id: int, line_number: int, context_size: int = 3) -> Tuple[List[str], List[str]]:
        """Get context lines around a search result"""
        # Get lines before
        cursor.execute("""
            SELECT content FROM search_index
            WHERE file_id = ? AND line_number BETWEEN ? AND ?
            ORDER BY line_number
        """, (file_id, max(1, line_number - context_size), line_number - 1))
        context_before = [row[0] for row in cursor.fetchall()]
        
        # Get lines after
        cursor.execute("""
            SELECT content FROM search_index
            WHERE file_id = ? AND line_number BETWEEN ? AND ?
            ORDER BY line_number
        """, (file_id, line_number + 1, line_number + context_size))
        context_after = [row[0] for row in cursor.fetchall()]
        
        return context_before, context_after
    
    def _find_symbol_at_line(self, cursor, file_id: int, line_number: int) -> Optional[CodeSymbol]:
        """Find symbol that contains the given line"""
        cursor.execute("""
            SELECT name, type, line_number, end_line, signature, docstring, complexity
            FROM symbols
            WHERE file_id = ? AND line_number <= ? AND end_line >= ?
            ORDER BY (end_line - line_number)
            LIMIT 1
        """, (file_id, line_number, line_number))
        
        row = cursor.fetchone()
        if row:
            return CodeSymbol(
                name=row[0],
                type=row[1],
                file_path=Path(""),  # Will be filled by caller
                line_number=row[2],
                end_line=row[3],
                signature=row[4] or "",
                docstring=row[5] or "",
                complexity=row[6]
            )
        return None
    
    def get_codebase_stats(self) -> CodebaseStats:
        """Get statistics about the indexed codebase"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic file stats
        cursor.execute("SELECT COUNT(*), SUM(lines) FROM files")
        total_files, total_lines = cursor.fetchone()
        
        # Language distribution
        cursor.execute("SELECT language, COUNT(*) FROM files GROUP BY language")
        languages = dict(cursor.fetchall())
        
        # Symbol type distribution
        cursor.execute("SELECT type, COUNT(*) FROM symbols GROUP BY type")
        symbols = dict(cursor.fetchall())
        
        # Average complexity
        cursor.execute("SELECT AVG(complexity) FROM symbols WHERE complexity > 0")
        result = cursor.fetchone()
        avg_complexity = result[0] if result[0] else 0.0
        
        conn.close()
        
        return CodebaseStats(
            total_files=total_files or 0,
            total_lines=total_lines or 0,
            languages=languages,
            symbols=symbols,
            complexity_score=avg_complexity,
            last_updated=datetime.now()
        )
    
    def get_file_content(self, file_path: Path, start_line: int = None, end_line: int = None) -> str:
        """Get content of a specific file or range of lines"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            if start_line is not None or end_line is not None:
                lines = content.splitlines()
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)
                return '\n'.join(lines[start_idx:end_idx])
            
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def get_symbol_references(self, symbol_name: str) -> List[SearchResult]:
        """Find all references to a symbol"""
        # This is a simplified implementation - a more sophisticated version
        # would use actual AST analysis to find true references
        return self.search(symbol_name, max_results=100)

# Global codebase index instance
codebase_index = CodebaseIndex()

class ContextExtractor:
    """Extracts relevant context for AI generation"""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except:
            self.tokenizer = None
            logger.warning("tiktoken not available, using simple token estimation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: 1 token ≈ 0.75 words
            return int(len(text.split()) * 1.33)
    
    async def extract_context_for_query(self, query: str, codebase_path: Path) -> str:
        """Extract relevant context from codebase for a query"""
        # Search for relevant code
        search_results = codebase_index.search(query, max_results=20)
        
        # Also search for symbols that might be relevant
        symbol_results = codebase_index.search_symbols(query)
        
        context_parts = []
        current_tokens = 0
        
        # Add file paths and basic info
        relevant_files = set()
        for result in search_results:
            relevant_files.add(result.file_path)
        
        if relevant_files:
            file_list = "Relevant files found:\n" + "\n".join(f"- {f}" for f in sorted(relevant_files))
            context_parts.append(file_list)
            current_tokens += self.count_tokens(file_list)
        
        # Add symbol information
        if symbol_results:
            symbol_info = self._format_symbols(symbol_results)
            if symbol_info and current_tokens + self.count_tokens(symbol_info) < self.max_tokens:
                context_parts.append(symbol_info)
                current_tokens += self.count_tokens(symbol_info)
        
        # Add search results with context
        for result in search_results:
            if current_tokens >= self.max_tokens * 0.8:  # Leave room for more content
                break
            
            result_context = self._format_search_result(result)
            result_tokens = self.count_tokens(result_context)
            
            if current_tokens + result_tokens < self.max_tokens:
                context_parts.append(result_context)
                current_tokens += result_tokens
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)
    
    def _format_symbols(self, symbols: List[CodeSymbol]) -> str:
        """Format symbol information for context"""
        if not symbols:
            return ""
        
        parts = ["Symbol Definitions:"]
        for symbol in symbols[:10]:  # Limit to top 10 symbols
            symbol_str = f"- {symbol.type} `{symbol.name}` in {symbol.file_path}:{symbol.line_number}"
            if symbol.signature:
                symbol_str += f"\n  Signature: {symbol.signature}"
            if symbol.docstring:
                # Truncate long docstrings
                docstring = symbol.docstring[:200] + "..." if len(symbol.docstring) > 200 else symbol.docstring
                symbol_str += f"\n  Documentation: {docstring}"
            parts.append(symbol_str)
        
        return "\n".join(parts)
    
    def _format_search_result(self, result: SearchResult) -> str:
        """Format a search result with context for inclusion"""
        parts = [f"File: {result.file_path}:{result.line_number}"]
        
        # Add context
        if result.context_before:
            for i, line in enumerate(result.context_before[-3:]):  # Last 3 lines before
                parts.append(f"{result.line_number - len(result.context_before) + i:4d}: {line}")
        
        # Add the actual matching line (highlighted)
        parts.append(f"{result.line_number:4d}:*{result.line_content}")
        
        if result.context_after:
            for i, line in enumerate(result.context_after[:3]):  # First 3 lines after
                parts.append(f"{result.line_number + i + 1:4d}: {line}")
        
        if result.symbol:
            parts.append(f"Symbol: {result.symbol.type} `{result.symbol.name}`")
        
        return "\n".join(parts)

# Global context extractor
context_extractor = ContextExtractor()

# CLI interface for codebase indexing
async def main():
    """CLI interface for testing codebase search"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Croq Codebase Search")
    parser.add_argument("command", choices=["index", "search", "stats"], help="Command to run")
    parser.add_argument("--path", default=".", help="Path to index or search in")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--type", help="Symbol type to search for")
    
    args = parser.parse_args()
    
    if args.command == "index":
        print(f"Indexing directory: {args.path}")
        result = await codebase_index.index_directory(Path(args.path))
        print(f"Indexed {result['indexed_files']} files in {result['duration_seconds']:.2f} seconds")
        
    elif args.command == "search":
        if not args.query:
            print("Please provide a search query with --query")
            sys.exit(1)
        
        print(f"Searching for: {args.query}")
        results = codebase_index.search(args.query)
        
        for result in results[:10]:  # Show top 10 results
            print(f"\n{result.file_path}:{result.line_number} (score: {result.relevance_score:.2f})")
            print(f"  {result.line_content}")
    
    elif args.command == "stats":
        stats = codebase_index.get_codebase_stats()
        print(f"Codebase Statistics:")
        print(f"  Total files: {stats.total_files}")
        print(f"  Total lines: {stats.total_lines}")
        print(f"  Languages: {dict(stats.languages)}")
        print(f"  Symbols: {dict(stats.symbols)}")
        print(f"  Average complexity: {stats.complexity_score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
