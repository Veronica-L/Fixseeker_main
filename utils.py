import numpy as np
import re
from io import StringIO
import keyword 
import builtins

def cost_effort_at_l(y_true, y_pred_proba, l):
    n = len(y_true)
    k = int(n * l)
    sort_indices = np.argsort(y_pred_proba)[::-1]
    top_k_true = y_true[sort_indices[:k]]
    return 1 - np.sum(top_k_true) / np.sum(y_true)

def post_at_l(y_true, y_pred_proba, l):
    n = len(y_true)
    k = int(n * l)
    sort_indices = np.argsort(y_pred_proba)[::-1]
    top_k_true = y_true[sort_indices[:k]]
    return np.sum(top_k_true) / k


def normalize_code(code_str):
    """
    normalize code：remove redundant spaces, tabs, and comments.
    
    Args:
        code_str (str): input code string
        
    Returns:
        str: formatted code string
    """
    code_str = re.sub(r'#.*$', '', code_str, flags=re.MULTILINE)
    
    code_str = re.sub(r'\s+$', '', code_str, flags=re.MULTILINE)
    code_str = re.sub(r' +', ' ', code_str)
    code_str = re.sub(r'^\t+', '    ', code_str, flags=re.MULTILINE)
    
    return code_str.strip()

def abstract_code(code_str):
    """
    Abstract code: replace variable names and function arguments with placeholders
    """

    keywords = set(keyword.kwlist)
    common_functions = {'print', 'len', 'range', 'int', 'str', 'float', 'list', 'dict', 'set', 'return'}
    reserved = keywords.union(common_functions)

    def_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\):'
    function_defs = re.findall(def_pattern, code_str, re.DOTALL)

    var_map = {}
    arg_map = {}
    var_counter = 1
    arg_counter = 1

    for func_name, args in function_defs:
        if func_name not in reserved:
            var_map[func_name] = f"FUNC{var_counter}"
            var_counter += 1
            
        args = args.split(',')
        for arg in args:
            arg = arg.strip()
            if arg and arg not in reserved:
                arg_name = arg.split('=')[0].strip()  
                if arg_name not in arg_map:
                    arg_map[arg_name] = f"ARG{arg_counter}"
                    arg_counter += 1

    var_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
    variables = re.findall(var_pattern, code_str)
    
    for var in variables:
        if var not in reserved and var not in arg_map and var not in var_map:
            var_map[var] = f"VAR{var_counter}"
            var_counter += 1
    
    result = code_str

    all_replacements = []
    for name, placeholder in arg_map.items():
        all_replacements.append((name, placeholder))
    for name, placeholder in var_map.items():
        all_replacements.append((name, placeholder))
    
    all_replacements.sort(key=lambda x: len(x[0]), reverse=True)
    
    for name, placeholder in all_replacements:
        pattern = r'\b' + re.escape(name) + r'\b'
        result = re.sub(pattern, placeholder, result)
    
    return result