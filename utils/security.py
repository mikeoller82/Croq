"""
Security Scanner Utility
"""
import ast
import asyncio
import logging
import re
from typing import Dict, List, Any
import tempfile
import subprocess


logger = logging.getLogger(__name__)


class SecurityScanner:
    """Advanced security vulnerability scanner"""
    
    def __init__(self):
        self.vulnerability_patterns = {
            "code_injection": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"compile\s*\(",
                r"__import__\s*\(",
            ],
            "command_injection": [
                r"os\.system\s*\(",
                r"subprocess\.call\s*\(",
                r"subprocess\.run\s*\(",
                r"subprocess\.Popen\s*\(",
                r"os\.popen\s*\(",
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"os\.path\.join\s*\([^)]*\.\.",
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
                r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
                r"token\s*=\s*['\"][^'\"]+['\"]",
            ],
            "sql_injection": [
                r"\.execute\s*\(\s*['\"][^'\"]*%[sd][^'\"]*['\"]",
                r"\.execute\s*\(\s*['\"][^'\"]*\+[^'\"]*['\"]",
                r"\.execute\s*\(\s*f['\"]",
            ],
            "unsafe_deserialization": [
                r"pickle\.load",
                r"pickle\.loads",
                r"cPickle\.load",
                r"yaml\.load\s*\(",
            ]
        }
    
    async def scan_code(self, code: str, language: str = "python") -> List[Dict[str, Any]]:
        """Comprehensive security scan"""
        
        if not code.strip():
            return []
        
        issues = []
        
        # Static pattern matching
        pattern_issues = await self._scan_patterns(code)
        issues.extend(pattern_issues)
        
        # AST-based analysis
        try:
            tree = ast.parse(code)
            ast_issues = await self._scan_ast(tree, code)
            issues.extend(ast_issues)
        except SyntaxError:
            # Can't analyze malformed code
            pass
        
        # External tool scanning
        bandit_issues = await self._scan_with_bandit(code)
        issues.extend(bandit_issues)
        
        return self._deduplicate_issues(issues)
    
    async def _scan_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Scan for security anti-patterns using regex"""
        
        issues = []
        lines = code.splitlines()
        
        for category, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append({
                            "type": "pattern_match",
                            "category": category,
                            "severity": self._get_severity(category),
                            "line": line_num,
                            "description": f"Potential {category.replace('_', ' ')} vulnerability",
                            "code_snippet": line.strip(),
                            "pattern": pattern
                        })
        
        return issues
    
    async def _scan_ast(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """AST-based security analysis"""
        
        issues = []
        lines = code.splitlines()
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    # Dangerous builtins
                    if func_name in ["eval", "exec", "compile", "__import__"]:
                        issues.append({
                            "type": "dangerous_call",
                            "category": "code_injection",
                            "severity": "high",
                            "line": node.lineno,
                            "description": f"Use of dangerous function '{func_name}'",
                            "code_snippet": lines[node.lineno-1].strip() if node.lineno <= len(lines) else "",
                            "function": func_name
                        })
                
                elif isinstance(node.func, ast.Attribute):
                    # Check for dangerous attribute calls
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == "os" and 
                        node.func.attr in ["system", "popen"]):
                        
                        issues.append({
                            "type": "dangerous_call",
                            "category": "command_injection",
                            "severity": "high",
                            "line": node.lineno,
                            "description": f"Use of dangerous function 'os.{node.func.attr}'",
                            "code_snippet": lines[node.lineno-1].strip() if node.lineno <= len(lines) else "",
                            "function": f"os.{node.func.attr}"
                        })
            
            # Check for hardcoded secrets in assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id.lower()
                        if any(secret in var_name for secret in ["password", "secret", "key", "token"]):
                            if isinstance(node.value, ast.Constant):
                                issues.append({
                                    "type": "hardcoded_secret",
                                    "category": "hardcoded_secrets",
                                    "severity": "medium",
                                    "line": node.lineno,
                                    "description": f"Potential hardcoded secret in variable '{target.id}'",
                                    "code_snippet": lines[node.lineno-1].strip() if node.lineno <= len(lines) else "",
                                    "variable": target.id
                                })
            
            # Check for unsafe file operations
            elif isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Name) and 
                    node.func.id == "open" and 
                    node.args):
                    
                    # Check if file path comes from user input
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.BinOp):  # String concatenation
                        issues.append({
                            "type": "unsafe_file_access",
                            "category": "path_traversal",
                            "severity": "medium",
                            "line": node.lineno,
                            "description": "File path constructed from concatenation - potential path traversal",
                            "code_snippet": lines[node.lineno-1].strip() if node.lineno <= len(lines) else ""
                        })
        
        return issues
    
    async def _scan_with_bandit(self, code: str) -> List[Dict[str, Any]]:
        """Use bandit for additional security scanning"""
        
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            # Run bandit
            result = await asyncio.to_thread(
                subprocess.run,
                ["bandit", "-f", "json", temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                try:
                    import json
                    data = json.loads(result.stdout)
                    
                    issues = []
                    for issue in data.get("results", []):
                        issues.append({
                            "type": "bandit",
                            "category": issue.get("test_name", "unknown"),
                            "severity": issue.get("issue_severity", "medium").lower(),
                            "line": issue.get("line_number", 0),
                            "description": issue.get("issue_text", "Security issue detected"),
                            "code_snippet": issue.get("code", "").strip(),
                            "confidence": issue.get("issue_confidence", "medium").lower()
                        })
                    
                    return issues
                
                except json.JSONDecodeError:
                    logger.error("Failed to parse bandit output")
            
            # Clean up temp file
            import os
            os.unlink(temp_path)
            
        except FileNotFoundError:
            logger.debug("Bandit not available")
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
        
        return []
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for vulnerability category"""
        
        severity_map = {
            "code_injection": "high",
            "command_injection": "high",
            "sql_injection": "high",
            "unsafe_deserialization": "high",
            "path_traversal": "medium",
            "hardcoded_secrets": "medium",
        }
        
        return severity_map.get(category, "medium")
    
    def _deduplicate_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate security issues"""
        
        seen = set()
        deduplicated = []
        
        for issue in issues:
            # Create key based on line, category, and type
            key = (issue.get("line", 0), issue.get("category", ""), issue.get("type", ""))
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(issue)
        
        # Sort by severity and line number
        severity_order = {"high": 0, "medium": 1, "low": 2}
        
        deduplicated.sort(
            key=lambda x: (
                severity_order.get(x.get("severity", "medium"), 1),
                x.get("line", 0)
            )
        )
        
        return deduplicated
