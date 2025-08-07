"""
Advanced Code Analysis Utilities
"""
import ast
import asyncio
import logging
from typing import Dict, Any, List, Optional
import subprocess
import tempfile
import os
from pathlib import Path


logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Advanced code quality analyzer"""
    
    def __init__(self):
        self.analyzers = [
            self._analyze_complexity,
            self._analyze_maintainability,
            self._analyze_documentation,
            self._analyze_style,
            self._analyze_performance,
        ]
    
    async def analyze_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Comprehensive code analysis"""
        
        if not code.strip():
            return {
                "quality_score": 0,
                "complexity_score": 0,
                "maintainability_score": 0,
                "documentation_score": 0,
                "style_score": 0,
                "performance_score": 0,
                "issues": ["Empty code provided"],
                "suggestions": []
            }
        
        # Parse code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "quality_score": 0,
                "complexity_score": 0,
                "maintainability_score": 0,
                "documentation_score": 0,
                "style_score": 0,
                "performance_score": 0,
                "issues": [f"Syntax error: {str(e)}"],
                "suggestions": ["Fix syntax errors"]
            }
        
        # Run all analyzers
        results = {}
        for analyzer in self.analyzers:
            try:
                result = await analyzer(code, tree)
                results.update(result)
            except Exception as e:
                logger.error(f"Analyzer failed: {e}")
        
        # Calculate overall quality score
        scores = [
            results.get("complexity_score", 5),
            results.get("maintainability_score", 5),
            results.get("documentation_score", 5),
            results.get("style_score", 5),
            results.get("performance_score", 5)
        ]
        
        overall_score = sum(scores) / len(scores)
        results["quality_score"] = round(overall_score, 1)
        
        return results
    
    async def quick_analysis(self, code: str) -> Dict[str, Any]:
        """Quick analysis for streaming responses"""
        
        if not code.strip():
            return {"quality_score": 0, "issues": ["Empty code"]}
        
        try:
            tree = ast.parse(code)
            complexity = self._calculate_complexity(tree)
            
            return {
                "quality_score": max(1, 10 - complexity),
                "complexity_score": min(complexity, 10),
                "line_count": len(code.splitlines()),
                "function_count": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                "class_count": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            }
        except SyntaxError:
            return {"quality_score": 0, "issues": ["Syntax error"]}
    
    async def _analyze_complexity(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze cyclomatic complexity"""
        
        complexity = self._calculate_complexity(tree)
        
        score = max(1, 10 - complexity)  # Invert: lower complexity = higher score
        
        issues = []
        if complexity > 15:
            issues.append("Very high complexity detected")
        elif complexity > 10:
            issues.append("High complexity detected")
        elif complexity > 7:
            issues.append("Moderate complexity detected")
        
        return {
            "complexity_score": score,
            "raw_complexity": complexity,
            "complexity_issues": issues
        }
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate McCabe cyclomatic complexity"""
        
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Decision points increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)  # Each except handler
                if node.orelse:
                    complexity += 1  # else clause
                if node.finalbody:
                    complexity += 1  # finally clause
            elif isinstance(node, (ast.BoolOp,)):
                # And/Or operations
                complexity += len(node.values) - 1
            elif isinstance(node, ast.comprehension):
                # List/dict/set comprehensions
                complexity += 1
                for if_clause in node.ifs:
                    complexity += 1
        
        return complexity
    
    async def _analyze_maintainability(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code maintainability"""
        
        lines = code.splitlines()
        loc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        # Function/method metrics
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        avg_function_length = (loc / len(functions)) if functions else 0
        
        # Class metrics  
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        
        score = 10
        issues = []
        
        # Penalize long functions
        if avg_function_length > 50:
            score -= 3
            issues.append("Functions are too long (average > 50 lines)")
        elif avg_function_length > 30:
            score -= 2
            issues.append("Functions are moderately long")
        
        # Penalize too many nested levels
        max_nesting = self._calculate_max_nesting(tree)
        if max_nesting > 4:
            score -= 2
            issues.append("Deep nesting detected (> 4 levels)")
        elif max_nesting > 3:
            score -= 1
            issues.append("Moderate nesting detected")
        
        # Reward good structure
        if len(classes) > 0 and len(functions) > 0:
            score += 1  # Good OOP structure
        
        return {
            "maintainability_score": max(1, score),
            "lines_of_code": loc,
            "avg_function_length": round(avg_function_length, 1),
            "max_nesting_depth": max_nesting,
            "function_count": len(functions),
            "class_count": len(classes),
            "maintainability_issues": issues
        }
    
    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                                    ast.With, ast.AsyncWith, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_depth(tree)
    
    async def _analyze_documentation(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze documentation coverage and quality"""
        
        # Count documentable items
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        
        total_items = len(functions) + len(classes)
        if total_items == 0:
            return {
                "documentation_score": 10,
                "doc_coverage": 100,
                "documentation_issues": []
            }
        
        # Count documented items
        documented = 0
        
        for func in functions:
            if ast.get_docstring(func):
                documented += 1
        
        for cls in classes:
            if ast.get_docstring(cls):
                documented += 1
        
        # Check module docstring
        module_doc = ast.get_docstring(tree)
        has_module_doc = module_doc is not None
        
        coverage = (documented / total_items) * 100 if total_items > 0 else 100
        
        score = min(10, coverage / 10)
        if has_module_doc:
            score += 1
        
        issues = []
        if coverage < 50:
            issues.append("Low documentation coverage (< 50%)")
        elif coverage < 80:
            issues.append("Moderate documentation coverage")
        
        if not has_module_doc and total_items > 3:
            issues.append("Missing module-level documentation")
        
        return {
            "documentation_score": max(1, min(10, score)),
            "doc_coverage": round(coverage, 1),
            "documented_items": documented,
            "total_documentable": total_items,
            "has_module_doc": has_module_doc,
            "documentation_issues": issues
        }
    
    async def _analyze_style(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code style and formatting"""
        
        score = 10
        issues = []
        
        lines = code.splitlines()
        
        # Check line length
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 88]
        if long_lines:
            score -= min(3, len(long_lines) * 0.1)
            issues.append(f"Long lines detected: {len(long_lines)} lines > 88 chars")
        
        # Check naming conventions
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        
        bad_function_names = [f.name for f in functions if not f.name.islower() or '__' in f.name[1:-1]]
        bad_class_names = [c.name for c in classes if not c.name[0].isupper()]
        
        if bad_function_names:
            score -= 1
            issues.append(f"Non-PEP8 function names: {bad_function_names[:3]}")
        
        if bad_class_names:
            score -= 1
            issues.append(f"Non-PEP8 class names: {bad_class_names[:3]}")
        
        # Check imports organization
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        if len(imports) > 5:
            # Simple check: imports should be at the top
            first_import_line = min(n.lineno for n in imports) if imports else 0
            if first_import_line > 5:  # Imports not at top
                score -= 1
                issues.append("Imports not at top of file")
        
        return {
            "style_score": max(1, score),
            "long_lines": len(long_lines),
            "naming_issues": len(bad_function_names) + len(bad_class_names),
            "style_issues": issues
        }
    
    async def _analyze_performance(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze potential performance issues"""
        
        score = 8  # Start with good score
        issues = []
        
        # Check for potential performance anti-patterns
        for node in ast.walk(tree):
            # Nested loops
            if isinstance(node, (ast.For, ast.While)):
                nested_loops = [n for n in ast.walk(node) 
                               if isinstance(n, (ast.For, ast.While)) and n != node]
                if nested_loops:
                    score -= 1
                    issues.append("Nested loops detected (potential O(n²) complexity)")
            
            # String concatenation in loops
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if (isinstance(child, ast.AugAssign) and 
                        isinstance(child.op, ast.Add) and 
                        isinstance(child.target, ast.Name)):
                        score -= 1
                        issues.append("String concatenation in loop (use join() instead)")
                        break
            
            # Global variable access in functions
            if isinstance(node, ast.FunctionDef):
                globals_used = [n for n in ast.walk(node) 
                               if isinstance(n, ast.Global)]
                if globals_used:
                    score -= 0.5
                    issues.append("Global variable access in functions")
        
        return {
            "performance_score": max(1, score),
            "performance_issues": issues
        }
