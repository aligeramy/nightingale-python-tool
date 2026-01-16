"""Sandboxed Python code execution using RestrictedPython."""

import io
import re
import sys
import signal
from contextlib import contextmanager
from typing import Any

from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
)
from RestrictedPython.PrintCollector import PrintCollector as RPPrintCollector
import numpy as np
import pandas as pd

from app.config import settings


class ExecutionError(Exception):
    """Custom exception for code execution errors."""
    pass


class TimeoutError(Exception):
    """Custom exception for execution timeout."""
    pass


@contextmanager
def timeout_handler(seconds: int):
    """Context manager for execution timeout."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def get_safe_globals() -> dict[str, Any]:
    """Get a dictionary of safe globals for code execution."""
    # Start with safe builtins
    safe_globals = {
        "__builtins__": safe_builtins,
        "_getiter_": default_guarded_getiter,
        "_getitem_": default_guarded_getitem,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        # Use RestrictedPython's PrintCollector - it's a class that creates instances
        "_print_": RPPrintCollector,
    }

    # Add safe math functions
    import math
    safe_math = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "frozenset": frozenset,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "type": type,
        "isinstance": isinstance,
    }
    safe_globals.update(safe_math)

    # Add numpy with restricted access
    safe_globals["np"] = np
    safe_globals["numpy"] = np

    # Add pandas with restricted access
    safe_globals["pd"] = pd
    safe_globals["pandas"] = pd

    # Add math module
    safe_globals["math"] = math

    # Add scipy.stats for statistical functions
    from scipy import stats
    safe_globals["stats"] = stats

    return safe_globals


def strip_imports(code: str) -> str:
    """
    Remove all import statements from Python code.
    
    Handles:
    - import x
    - import x as y
    - from x import y
    - from x import y as z
    - Multi-line imports (with parentheses)
    - Comments after imports
    
    Args:
        code: Python code string
        
    Returns:
        Code string with all import statements removed
    """
    lines = code.split('\n')
    cleaned_lines = []
    in_multiline_import = False
    paren_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines that were part of imports
        if not stripped:
            if not in_multiline_import:
                cleaned_lines.append(line)
            continue
        
        # Check if this is an import statement
        is_import = stripped.startswith('import ') or stripped.startswith('from ')
        
        if is_import:
            # Check for multi-line import (has opening parenthesis)
            if '(' in stripped:
                in_multiline_import = True
                paren_count = stripped.count('(') - stripped.count(')')
                # Skip this line
                continue
            else:
                # Single-line import, skip it
                continue
        
        # Handle multi-line import continuation
        if in_multiline_import:
            paren_count += stripped.count('(') - stripped.count(')')
            # Check if this line ends the multi-line import
            if paren_count <= 0 and ')' in stripped:
                in_multiline_import = False
                paren_count = 0
            # Skip this line
            continue
        
        # Regular code line, keep it
        cleaned_lines.append(line)
    
    # Join lines back together
    cleaned_code = '\n'.join(cleaned_lines)
    
    # Clean up multiple consecutive blank lines (max 2 consecutive)
    cleaned_code = re.sub(r'\n{3,}', '\n\n', cleaned_code)
    
    # Remove leading blank lines
    cleaned_code = cleaned_code.lstrip('\n')
    
    return cleaned_code


def validate_code(code: str) -> tuple[bool, str | None]:
    """Validate code for forbidden patterns."""
    # Check for forbidden imports
    for forbidden in settings.forbidden_imports:
        if f"import {forbidden}" in code or f"from {forbidden}" in code:
            return False, f"Import of '{forbidden}' is not allowed"

    # Check for forbidden builtins usage
    for forbidden in settings.forbidden_builtins:
        if f"{forbidden}(" in code:
            return False, f"Use of '{forbidden}' is not allowed"

    # Check for file operations
    if "open(" in code and "open(" not in str(settings.forbidden_builtins):
        return False, "File operations are not allowed"

    return True, None


def execute_code(code: str, timeout: int | None = None) -> dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (default from settings)

    Returns:
        Dictionary with 'output', 'result', and 'error' keys
    """
    timeout = timeout or settings.execution_timeout

    # Strip import statements before validation/execution
    # This allows LLM-generated code with imports to work automatically
    code = strip_imports(code)

    # Validate code after stripping imports
    is_valid, error_msg = validate_code(code)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
            "output": None,
            "result": None,
        }

    # Compile with RestrictedPython
    try:
        byte_code = compile_restricted(code, "<inline>", "exec")
        if byte_code is None:
            return {
                "success": False,
                "error": "Code compilation failed - possibly contains forbidden constructs",
                "output": None,
                "result": None,
            }
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Syntax error: {e}",
            "output": None,
            "result": None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Compilation error: {e}",
            "output": None,
            "result": None,
        }

    # Prepare execution environment
    safe_globals = get_safe_globals()
    local_vars: dict[str, Any] = {}

    try:
        with timeout_handler(timeout):
            exec(byte_code, safe_globals, local_vars)

        # Get output from RestrictedPython's PrintCollector
        # The collector stores output in '_print' local variable as 'printed' attribute
        output = ""
        if "_print" in local_vars:
            printed = getattr(local_vars["_print"], "printed", None)
            if printed:
                output = printed

        # Try to get a result value (last expression or 'result' variable)
        result = local_vars.get("result", None)

        # Convert numpy/pandas types to Python natives for JSON serialization
        if isinstance(result, (np.ndarray, pd.Series)):
            result = result.tolist()
        elif isinstance(result, pd.DataFrame):
            result = result.to_dict(orient="records")
        elif isinstance(result, np.generic):
            result = result.item()

        return {
            "success": True,
            "output": output if output else None,
            "result": result,
            "error": None,
        }

    except TimeoutError as e:
        output = ""
        if "_print" in local_vars:
            printed = getattr(local_vars["_print"], "printed", None)
            if printed:
                output = printed
        return {
            "success": False,
            "error": str(e),
            "output": output or None,
            "result": None,
        }
    except Exception as e:
        output = ""
        if "_print" in local_vars:
            printed = getattr(local_vars["_print"], "printed", None)
            if printed:
                output = printed
        return {
            "success": False,
            "error": f"Execution error: {type(e).__name__}: {e}",
            "output": output or None,
            "result": None,
        }
