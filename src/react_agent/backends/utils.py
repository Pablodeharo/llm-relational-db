"""
backends/utils.py

Utility functions for model backends.
Includes quantization helpers, prompt formatters, and tool parsing utilities.
"""

import re
import json
import torch
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


# ==========================================
# QUANTIZATION UTILITIES
# ==========================================

# Decide si un modelo cabe en GPU
# Seleccionar backend (transformers vs llamacpp)
# Automatizar eleccion

def estimate_model_memory(
    model_size_b: float,
    quantization: Optional[str] = None,
    overhead_mb: float = 500
) -> float:
    """
    Estimate memory requirements for a model.
    
    Args:
        model_size_b: Model size in billions of parameters
        quantization: Type of quantization (4bit, 8bit, Q4, Q5, Q8, None)
        overhead_mb: Additional memory overhead
        
    Returns:
        Estimated memory in MB
    """
    # Base calculation: ~4 bytes per parameter for fp32
    base_mb = model_size_b * 1000 * 4
    
    # Apply quantization factor
    if quantization:
        quant_lower = quantization.lower()
        
        if "4bit" in quant_lower or "q4" in quant_lower:
            base_mb *= 0.25  # 4-bit = 25% of fp32
        elif "8bit" in quant_lower or "q8" in quant_lower:
            base_mb *= 0.5   # 8-bit = 50% of fp32
        elif "q5" in quant_lower:
            base_mb *= 0.3125  # Q5 = ~31% of fp32
        elif "fp16" in quant_lower or "float16" in quant_lower:
            base_mb *= 0.5
    
    return base_mb + overhead_mb




def get_quantization_config(quantization: str, device: str = "cuda") -> Optional[Dict]:
    """
    Get BitsAndBytes quantization config for transformers.
    
    Args:
        quantization: Type of quantization (4bit, 8bit)
        device: Target device (cuda, cpu)
        
    Returns:
        Config dict for BitsAndBytesConfig or None
    """
    if device == "cpu" or not quantization:
        return None
    
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return None
    
    if quantization == "4bit":
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    elif quantization == "8bit":
        return {
            "load_in_8bit": True
        }
    
    return None


def detect_quantization_from_filename(filename: str) -> Optional[str]:
    """
    Detect quantization type from GGUF filename.
    
    Args:
        filename: Name of the GGUF file
        
    Returns:
        Quantization type (Q4_K_M, Q5_K_M, etc.) or None
    """
    patterns = [
        r'Q([2-8])_K_M',
        r'Q([2-8])_K_S',
        r'Q([2-8])_K_L',
        r'Q([2-8])_0',
        r'Q([2-8])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(0).upper()
    
    return None


# ==========================================
# PROMPT FORMATTING
# ==========================================
# Construye prompt estructurado

def format_sql_prompt(
    question: str,
    schema: Dict[str, List[Dict]],
    examples: Optional[List[Dict]] = None
) -> str:
    """
    Format a prompt for SQL generation.
    
    Args:
        question: User's question
        schema: Database schema dict {table: [columns]}
        examples: Optional few-shot examples
        
    Returns:
        Formatted prompt string
    """
    prompt = "### Database Schema\n\n"
    
    # Add schema
    for table, columns in schema.items():
        prompt += f"Table: {table}\n"
        for col in columns:
            col_name = col.get('column', col.get('name', ''))
            col_type = col.get('type', col.get('data_type', ''))
            prompt += f"  - {col_name} ({col_type})\n"
        prompt += "\n"
    
    # Add examples if provided
    if examples:
        prompt += "### Examples\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"SQL: {ex['sql']}\n\n"
    
    # Add question
    prompt += "### Task\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Generate a SQL query to answer this question.\n"
    prompt += "Return ONLY the SQL query, no explanations.\n\n"
    prompt += "SQL:\n"
    
    return prompt

# Genera un prompt React clásico
def format_react_prompt(
    question: str,
    tools: List[Dict],
    history: Optional[List[Dict]] = None
) -> str:
    """
    Format a ReAct-style prompt with tool descriptions.
    
    Args:
        question: User's question
        tools: Available tools
        history: Previous actions/observations
        
    Returns:
        Formatted ReAct prompt
    """
    prompt = "You are an AI assistant with access to tools.\n\n"
    
    # Add tools
    prompt += "### Available Tools\n\n"
    for tool in tools:
        name = tool.get('name', '')
        desc = tool.get('description', '')
        params = tool.get('parameters', {})
        
        prompt += f"**{name}**: {desc}\n"
        if params:
            prompt += f"  Parameters: {json.dumps(params)}\n"
        prompt += "\n"
    
    # Add history if present
    if history:
        prompt += "### Previous Actions\n\n"
        for item in history:
            if 'action' in item:
                prompt += f"Action: {item['action']}\n"
            if 'observation' in item:
                prompt += f"Observation: {item['observation']}\n"
            prompt += "\n"
    
    # Add question
    prompt += "### Question\n"
    prompt += f"{question}\n\n"
    
    # Add format instructions
    prompt += "### Response Format\n"
    prompt += "Respond with:\n"
    prompt += "Thought: [your reasoning]\n"
    prompt += "Action: [tool_name]\n"
    prompt += "Action Input: {\"param\": \"value\"}\n"
    prompt += "\nYour response:\n"
    
    return prompt


# ==========================================
# TOOL CALLING PARSERS
# ==========================================

def parse_json_tool_call(text: str) -> List[Dict]:
    """
    Parse JSON-formatted tool calls from text.
    
    Looks for patterns like:
    ```json
    {"tool": "query_postgres", "args": {...}}
    ```
    
    Args:
        text: Generated text
        
    Returns:
        List of parsed tool calls
    """
    tool_calls = []
    
    # Pattern 1: JSON code blocks
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.finditer(json_pattern, text, re.DOTALL)
    
    for i, match in enumerate(matches):
        try:
            data = json.loads(match.group(1))
            
            # Support different formats
            tool_name = data.get('tool') or data.get('function') or data.get('name')
            args = data.get('args') or data.get('arguments') or data.get('parameters', {})
            
            if tool_name:
                tool_calls.append({
                    'id': f'call_{i}',
                    'name': tool_name,
                    'arguments': args
                })
        except json.JSONDecodeError:
            continue
    
    # Pattern 2: Plain JSON (no code blocks)
    if not tool_calls:
        try:
            # Try to find any JSON object
            json_objs = re.findall(r'\{[^{}]*"(?:tool|function|name)"[^{}]*\}', text)
            for i, obj_str in enumerate(json_objs):
                data = json.loads(obj_str)
                tool_name = data.get('tool') or data.get('function') or data.get('name')
                args = data.get('args') or data.get('arguments') or {}
                
                if tool_name:
                    tool_calls.append({
                        'id': f'call_{i}',
                        'name': tool_name,
                        'arguments': args
                    })
        except:
            pass
    
    return tool_calls


def parse_react_tool_call(text: str) -> List[Dict]:
    """
    Parse ReAct-formatted tool calls from text.
    
    Looks for patterns like:
    Action: query_postgres
    Action Input: {"query": "SELECT * FROM users"}
    
    Args:
        text: Generated text
        
    Returns:
        List of parsed tool calls
    """
    tool_calls = []
    
    # Pattern: Action: X\nAction Input: {...}
    pattern = r'Action:\s*(\w+)\s*(?:Action Input|Input):\s*(\{.*?\})'
    matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
    
    for i, match in enumerate(matches):
        tool_name = match.group(1).strip()
        try:
            args = json.loads(match.group(2))
            tool_calls.append({
                'id': f'call_{i}',
                'name': tool_name,
                'arguments': args
            })
        except json.JSONDecodeError:
            # Try to parse as Python dict
            try:
                args = eval(match.group(2))
                tool_calls.append({
                    'id': f'call_{i}',
                    'name': tool_name,
                    'arguments': args
                })
            except:
                continue
    
    return tool_calls


def parse_tool_calls(text: str, format: str = "auto") -> List[Dict]:
    """
    Parse tool calls from text using specified format.
    
    Args:
        text: Generated text
        format: Format to use (auto, json, react)
        
    Returns:
        List of parsed tool calls
    """
    if format == "json":
        return parse_json_tool_call(text)
    elif format == "react":
        return parse_react_tool_call(text)
    else:  # auto
        # Try JSON first, then ReAct
        calls = parse_json_tool_call(text)
        if not calls:
            calls = parse_react_tool_call(text)
        return calls


# ==========================================
# MODEL INFO UTILITIES
# ==========================================

# def get_model_size_from_config(config_path: Path) -> Optional[float]:
#    """
#    Extract model size (in billions) from config.json.
#    
#    Args:
#        config_path: Path to config.json
#        
#    Returns:
#        Model size in billions or None
#    """
#    try:
#        with open(config_path, 'r') as f:
#            config = json.load(f)
#        
#        # Try different fields
#        if 'num_parameters' in config:
#            return config['num_parameters'] / 1e9
#        elif 'n_params' in config:
#            return config['n_params'] / 1e9
#        
#        # Estimate from layers and hidden size
#        hidden_size = config.get('hidden_size', 0)
#        num_layers = config.get('num_hidden_layers', 0)
#       
#        if hidden_size and num_layers:
#            # Rough estimate: hidden_size^2 * layers * 12 / 1e9
#            return (hidden_size ** 2 * num_layers * 12) / 1e9
#        
#    except Exception:
#        pass
#    
#    return None

def get_model_size_from_config_dict(config: Dict) -> Optional[float]:
    """
    Extract model size (in billions) from a HuggingFace config dict.
    """
    if "num_parameters" in config:
        return config["num_parameters"] / 1e9
    if "n_params" in config:
        return config["n_params"] / 1e9

    hidden_size = config.get("hidden_size")
    num_layers = config.get("num_hidden_layers")

    if hidden_size and num_layers:
        return (hidden_size ** 2 * num_layers * 12) / 1e9

    return None

def infer_context_length(model_config: Dict) -> int:
    """
    Infer context length from model configuration.
    
    Args:
        model_config: Model config dict
        
    Returns:
        Context length in tokens
    """
    # Common field names
    fields = [
        'max_position_embeddings',
        'n_positions',
        'seq_length',
        'max_seq_length',
        'context_length',
        'n_ctx'
    ]
    
    for field in fields:
        if field in model_config:
            return model_config[field]
    
    # Default fallback
    return 4096


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_sql_output(text: str) -> str:
    """
    Clean SQL output from model generation.
    
    Args:
        text: Raw generated text
        
    Returns:
        Cleaned SQL query
    """
    # Remove markdown code blocks
    text = re.sub(r'```sql\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Remove common prefixes
    prefixes = ['SQL:', 'Query:', 'Answer:', 'SELECT']
    for prefix in prefixes:
        if text.strip().startswith(prefix):
            if prefix == 'SELECT':
                text = 'SELECT' + text.split('SELECT', 1)[-1]
            else:
                text = text.split(prefix, 1)[-1]
            break
    
    # Take only first query (before semicolon)
    if ';' in text:
        text = text.split(';')[0] + ';'
    else:
        text = text.strip() + ';'
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def truncate_text(text: str, max_tokens: int, tokenizer=None) -> str:
    """
    Truncate text to maximum tokens.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        tokenizer: Optional tokenizer for accurate counting
        
    Returns:
        Truncated text
    """
    if tokenizer:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    else:
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]


# ==========================================
# VALIDATION
# ==========================================

def validate_sql_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    Basic SQL query validation.
    
    Args:
        query: SQL query string
        
    Returns:
        (is_valid, error_message)
    """
    query = query.strip().upper()
    
    # Check if empty
    if not query:
        return False, "Empty query"
    
    # Check for dangerous operations in simple way
    dangerous = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']
    for op in dangerous:
        if op in query and 'SELECT' not in query:
            return False, f"Potentially dangerous operation: {op}"
    
    # Check basic syntax
    if not any(kw in query for kw in ['SELECT', 'INSERT', 'UPDATE', 'WITH']):
        return False, "No valid SQL keyword found"
    
    # Check balanced parentheses
    if query.count('(') != query.count(')'):
        return False, "Unbalanced parentheses"
    
    return True, None


def validate_model_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate model configuration dictionary.
    
    Args:
        config: Model config dict
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required = ['name', 'backends']
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate backends
    if 'backends' in config:
        valid_backends = ['transformers', 'llamacpp', 'vllm']
        for backend in config['backends']:
            if backend not in valid_backends:
                errors.append(f"Invalid backend: {backend}")
    
    # Check for either repo or gguf_path
    if 'transformers' in config.get('backends', []):
        if 'repo' not in config:
            errors.append("Backend 'transformers' requires 'repo' field")
    
    if 'llamacpp' in config.get('backends', []):
        if 'gguf_path' not in config:
            errors.append("Backend 'llamacpp' requires 'gguf_path' field")
    
    return len(errors) == 0, errors