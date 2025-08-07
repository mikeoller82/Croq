"""
Advanced Hooks System for Croq AI Assistant
Provides extensible functionality through pre/post execution hooks
"""
import asyncio
import logging
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar('T')

class HookPriority(Enum):
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100

class HookType(Enum):
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"
    PRE_REQUEST = "pre_request"
    POST_REQUEST = "post_request"
    ON_ERROR = "on_error"
    ON_CACHE_HIT = "on_cache_hit"
    ON_CACHE_MISS = "on_cache_miss"
    PRE_ANALYSIS = "pre_analysis"
    POST_ANALYSIS = "post_analysis"
    ON_FILE_CHANGE = "on_file_change"
    ON_MODEL_SWITCH = "on_model_switch"

@dataclass
class HookContext:
    """Context passed to hook functions"""
    hook_type: HookType
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

@dataclass
class Hook:
    """Hook registration details"""
    name: str
    function: Callable
    hook_type: HookType
    priority: HookPriority = HookPriority.NORMAL
    enabled: bool = True
    description: str = ""
    conditions: List[Callable[[HookContext], bool]] = field(default_factory=list)
    
    def should_execute(self, context: HookContext) -> bool:
        """Check if hook should execute based on conditions"""
        if not self.enabled:
            return False
        return all(condition(context) for condition in self.conditions)

class HookResult:
    """Result of hook execution"""
    def __init__(self, success: bool = True, data: Any = None, error: Optional[Exception] = None):
        self.success = success
        self.data = data
        self.error = error
        self.modified = data is not None

class HookManager:
    """Manages and executes hooks throughout the application"""
    
    def __init__(self):
        self._hooks: Dict[HookType, List[Hook]] = defaultdict(list)
        self._hook_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"executed": 0, "errors": 0})
        self._middleware_stack: List[Callable] = []
    
    def register_hook(
        self,
        hook_type: HookType,
        function: Callable,
        name: Optional[str] = None,
        priority: HookPriority = HookPriority.NORMAL,
        description: str = "",
        conditions: Optional[List[Callable[[HookContext], bool]]] = None
    ) -> str:
        """Register a new hook"""
        hook_name = name or f"{function.__module__}.{function.__name__}"
        
        hook = Hook(
            name=hook_name,
            function=function,
            hook_type=hook_type,
            priority=priority,
            description=description,
            conditions=conditions or []
        )
        
        self._hooks[hook_type].append(hook)
        # Sort by priority (lowest number = highest priority)
        self._hooks[hook_type].sort(key=lambda h: h.priority.value)
        
        logger.info(f"Registered hook '{hook_name}' for {hook_type.value}")
        return hook_name
    
    def hook(
        self,
        hook_type: HookType,
        priority: HookPriority = HookPriority.NORMAL,
        name: Optional[str] = None,
        description: str = "",
        conditions: Optional[List[Callable[[HookContext], bool]]] = None
    ):
        """Decorator for registering hooks"""
        def decorator(func: Callable) -> Callable:
            self.register_hook(
                hook_type=hook_type,
                function=func,
                name=name,
                priority=priority,
                description=description,
                conditions=conditions
            )
            return func
        return decorator
    
    async def execute_hooks(
        self,
        hook_type: HookType,
        context: HookContext,
        stop_on_error: bool = False
    ) -> List[HookResult]:
        """Execute all hooks of a specific type"""
        results = []
        hooks = self._hooks.get(hook_type, [])
        
        if not hooks:
            return results
        
        logger.debug(f"Executing {len(hooks)} hooks for {hook_type.value}")
        
        for hook in hooks:
            if not hook.should_execute(context):
                continue
                
            try:
                self._hook_stats[hook.name]["executed"] += 1
                
                # Execute hook (async or sync)
                if inspect.iscoroutinefunction(hook.function):
                    result_data = await hook.function(context)
                else:
                    result_data = hook.function(context)
                
                results.append(HookResult(success=True, data=result_data))
                
            except Exception as e:
                self._hook_stats[hook.name]["errors"] += 1
                logger.error(f"Hook '{hook.name}' failed: {e}")
                
                results.append(HookResult(success=False, error=e))
                
                if stop_on_error:
                    break
        
        return results
    
    def unregister_hook(self, hook_type: HookType, name: str) -> bool:
        """Unregister a hook by name"""
        hooks = self._hooks.get(hook_type, [])
        for i, hook in enumerate(hooks):
            if hook.name == name:
                del hooks[i]
                logger.info(f"Unregistered hook '{name}' from {hook_type.value}")
                return True
        return False
    
    def enable_hook(self, name: str) -> bool:
        """Enable a hook by name"""
        for hooks in self._hooks.values():
            for hook in hooks:
                if hook.name == name:
                    hook.enabled = True
                    return True
        return False
    
    def disable_hook(self, name: str) -> bool:
        """Disable a hook by name"""
        for hooks in self._hooks.values():
            for hook in hooks:
                if hook.name == name:
                    hook.enabled = False
                    return True
        return False
    
    def get_hook_stats(self) -> Dict[str, Dict[str, int]]:
        """Get execution statistics for all hooks"""
        return dict(self._hook_stats)
    
    def list_hooks(self, hook_type: Optional[HookType] = None) -> Dict[HookType, List[Hook]]:
        """List all registered hooks"""
        if hook_type:
            return {hook_type: self._hooks.get(hook_type, [])}
        return dict(self._hooks)
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to the execution stack"""
        self._middleware_stack.append(middleware)
    
    async def execute_with_hooks(
        self,
        main_function: Callable,
        pre_hook: HookType,
        post_hook: HookType,
        context_data: Dict[str, Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute function with pre/post hooks"""
        # Create context
        context = HookContext(
            hook_type=pre_hook,
            data=context_data.copy()
        )
        
        # Execute pre-hooks
        await self.execute_hooks(pre_hook, context)
        
        try:
            # Execute main function
            if inspect.iscoroutinefunction(main_function):
                result = await main_function(*args, **kwargs)
            else:
                result = main_function(*args, **kwargs)
            
            # Update context for post-hooks
            context.hook_type = post_hook
            context.set("result", result)
            
            # Execute post-hooks
            await self.execute_hooks(post_hook, context)
            
            return result
            
        except Exception as e:
            # Execute error hooks
            error_context = HookContext(
                hook_type=HookType.ON_ERROR,
                data={"error": e, "function": main_function.__name__}
            )
            await self.execute_hooks(HookType.ON_ERROR, error_context)
            raise

# Global hook manager instance
hook_manager = HookManager()

# Convenience decorators
def pre_generation(priority: HookPriority = HookPriority.NORMAL, **kwargs):
    return hook_manager.hook(HookType.PRE_GENERATION, priority, **kwargs)

def post_generation(priority: HookPriority = HookPriority.NORMAL, **kwargs):
    return hook_manager.hook(HookType.POST_GENERATION, priority, **kwargs)

def on_error(priority: HookPriority = HookPriority.NORMAL, **kwargs):
    return hook_manager.hook(HookType.ON_ERROR, priority, **kwargs)

def pre_analysis(priority: HookPriority = HookPriority.NORMAL, **kwargs):
    return hook_manager.hook(HookType.PRE_ANALYSIS, priority, **kwargs)

def post_analysis(priority: HookPriority = HookPriority.NORMAL, **kwargs):
    return hook_manager.hook(HookType.POST_ANALYSIS, priority, **kwargs)

# Built-in hooks
@pre_generation(priority=HookPriority.HIGH, description="Log generation requests")
async def log_generation_request(context: HookContext):
    """Log details about generation requests"""
    logger.info(f"Starting generation: {context.get('model', 'unknown')} - {context.get('prompt_length', 0)} chars")

@post_generation(priority=HookPriority.HIGH, description="Log generation results")
async def log_generation_result(context: HookContext):
    """Log generation results"""
    result = context.get("result")
    if result:
        logger.info(f"Generation completed: {len(result.content if hasattr(result, 'content') else str(result))} chars")

@on_error(priority=HookPriority.HIGHEST, description="Log errors")
async def log_errors(context: HookContext):
    """Log errors with context"""
    error = context.get("error")
    function = context.get("function", "unknown")
    logger.error(f"Error in {function}: {error}")
