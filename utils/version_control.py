"""
Version Control Utility
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from git import Repo, GitError


logger = logging.getLogger(__name__)


class VersionControl:
    """Simple version control wrapper"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self._repo: Optional[Repo] = None
        self._init_repo()
    
    def _init_repo(self):
        """Initialize or open existing repository"""
        try:
            self._repo = Repo(self.repo_path)
            logger.info(f"Opened existing repository at {self.repo_path}")
        except GitError:
            try:
                self._repo = Repo.init(self.repo_path)
                logger.info(f"Initialized new repository at {self.repo_path}")
                
                # Create initial commit
                readme_path = self.repo_path / "README.md"
                readme_path.write_text("# Croq Generated Code Repository\n")
                
                self._repo.index.add([str(readme_path)])
                self._repo.index.commit("Initial commit")
                
            except Exception as e:
                logger.error(f"Failed to initialize repository: {e}")
                self._repo = None
    
    def is_healthy(self) -> bool:
        """Check if version control is working"""
        return self._repo is not None
    
    def commit_code(self, code: str, message: str, filename: str = "generated_code.py") -> Optional[str]:
        """Commit generated code"""
        
        if not self._repo:
            return None
        
        try:
            code_file = self.repo_path / filename
            code_file.write_text(code)
            
            self._repo.index.add([str(code_file)])
            commit = self._repo.index.commit(message)
            
            logger.info(f"Committed code: {commit.hexsha[:8]}")
            return commit.hexsha
            
        except Exception as e:
            logger.error(f"Failed to commit code: {e}")
            return None
    
    def get_history(self, max_count: int = 10) -> List[Dict[str, Any]]:
        """Get commit history"""
        
        if not self._repo:
            return []
        
        try:
            commits = []
            for commit in self._repo.iter_commits(max_count=max_count):
                commits.append({
                    "hash": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": str(commit.author),
                    "date": commit.committed_datetime.isoformat(),
                })
            
            return commits
            
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []
