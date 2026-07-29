import re
from typing import Any


def fix_slash(s: str) -> str:
    """Fix escape character issues"""
    s = re.sub(r'\x08oxed{([A-Z\d.]+)}', r'\\boxed{\1}', s)
    return s


def extract_answer_from_boxed(text: str) -> str | None:
    """Extract answer from \boxed{A} or \boxed{123}"""
    # Fix potential escape issues
    text = fix_slash(text)
    
    # Match letter answers
    match = re.search(r'\\boxed\{([A-Z])\}', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Match numeric answers
    match = re.search(r'\\boxed\{([\d.]+)\}', text)
    if match:
        return match.group(1)
    
    return None


def is_number_close(pred: str, gt: str, tolerance: float = 0.3) -> bool:
    """Check if two numbers are within tolerance range"""
    try:
        pred_num = float(pred)
        gt_num = float(gt)
        
        if gt_num == 0:
            return abs(pred_num - gt_num) <= tolerance
        
        relative_error = abs(pred_num - gt_num) / abs(gt_num)
        return relative_error <= tolerance
    except (ValueError, TypeError):
        return False


def compare_answers(pred: str | None, gt: str) -> bool:
    """Compare predicted answer with ground truth"""
    if pred is None:
        return False
    
    # Direct comparison for letter answers (case-insensitive)
    if pred.isalpha() and gt.isalpha():
        return pred.upper() == gt.upper()
    
    # Try numeric comparison
    try:
        float(gt)
        # Ground truth is a number
        if pred.replace('.', '').isdigit():
            # Prediction is also a number
            return is_number_close(pred, gt)
        else:
            return False
    except ValueError:
        # Ground truth is not a number, fall back to string comparison
        return pred.upper() == gt.upper()


def format_reward(response: str) -> float:
    """Check if response format contains \boxed{...}"""
    # Check for boxed format after fixing escape issues
    fixed_response = fix_slash(response)
    pattern = re.compile(r'\\boxed\{[A-Za-z\d.]+\}')
    format_match = pattern.search(fixed_response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Extract answer and compare with ground truth"""
    answer = extract_answer_from_boxed(response)
    
    if answer is None:
        return 0.0
    
    # Normalize ground truth (remove spaces, uppercase letters)
    gt = ground_truth.strip()
    if gt.isalpha():
        gt = gt.upper()
    
    return 1.0 if compare_answers(answer, gt) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    """
    Calculate reward scores for video QA tasks
    
    Args:
        reward_inputs: List of dictionaries containing 'response' and 'ground_truth'
        format_weight: Weight for format score (default 0.1)
    
    Returns:
        List of dictionaries containing 'overall', 'format', 'accuracy' scores
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for video QA reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        
        # Calculate format score
        format_score = format_reward(response)
        
        # Calculate accuracy score
        accuracy_score = accuracy_reward(response, ground_truth)
        
        # Calculate overall score
        overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
        })

    return scores