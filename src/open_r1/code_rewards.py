"""Reward functions for code verification."""
import re
from typing import Dict, List, Callable, Any
import ast
import sys
from io import StringIO
import pytest

def verify_output_correctness(
    output: str,
    expected_output: str,
    custom_validator: Callable[[str, str], bool] = None
) -> bool:
    """通用的输出正确性验证函数。
    
    Args:
        output: 代码实际输出
        expected_output: 期望输出
        custom_validator: 自定义验证函数，用于特定问题的验证逻辑
    
    Returns:
        bool: 输出是否正确
    """
    if custom_validator:
        return custom_validator(output, expected_output)
    
    # 标准化输出（移除多余空白字符，统一换行符）
    output = output.strip().replace('\r\n', '\n')
    expected_output = expected_output.strip().replace('\r\n', '\n')
    
    return output == expected_output

def run_code_safely(code: str, test_input: str, timeout: int = 5) -> Dict[str, Any]:
    """安全地运行代码并捕获输出。
    
    Args:
        code: 要执行的代码
        test_input: 测试输入
        timeout: 执行超时时间（秒）
    
    Returns:
        Dict with:
            'output': 代码输出
            'error': 错误信息（如果有）
            'execution_time': 执行时间
    """
    import time
    
    # 保存原始的 stdin 和 stdout
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    
    # 创建新的 stdin 和 stdout
    sys.stdin = StringIO(test_input)
    sys.stdout = StringIO()
    
    result = {
        'output': '',
        'error': None,
        'execution_time': 0
    }
    
    try:
        start_time = time.time()
        
        # 设置代码执行环境
        namespace = {}
        exec(code, namespace)
        
        result['execution_time'] = time.time() - start_time
        result['output'] = sys.stdout.getvalue()
        
    except Exception as e:
        result['error'] = str(e)
    finally:
        # 恢复原始的 stdin 和 stdout
        sys.stdin = old_stdin
        sys.stdout = old_stdout
    
    return result

def analyze_code_quality(code: str) -> Dict[str, float]:
    """分析代码质量的各个方面。
    
    Returns:
        Dict with:
            'naming_score': 变量命名得分
            'complexity_score': 代码复杂度得分
            'documentation_score': 文档和注释得分
            'error_handling_score': 错误处理得分
    """
    try:
        tree = ast.parse(code)
        scores = {
            'naming_score': 1.0,
            'complexity_score': 1.0,
            'documentation_score': 1.0,
            'error_handling_score': 1.0
        }
        
        # 1. 变量命名评分
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                # 允许常见的单字母变量
                if len(node.id) == 1 and node.id not in ['i', 'j', 'k', 'n', 'm', 'x', 'y']:
                    scores['naming_score'] -= 0.1
        
        # 2. 代码复杂度评分
        def get_complexity(node, level=0):
            if isinstance(node, (ast.For, ast.While, ast.If)):
                level += 1
            return max([level] + [get_complexity(child, level) 
                                for child in ast.iter_child_nodes(node)])
        
        complexity = get_complexity(tree)
        if complexity > 3:
            scores['complexity_score'] -= 0.2 * (complexity - 3)
        
        # 3. 文档和注释评分
        has_docstring = any(isinstance(node, ast.Expr) and 
                          isinstance(node.value, ast.Str)
                          for node in ast.walk(tree))
        if not has_docstring:
            scores['documentation_score'] -= 0.5
        
        # 4. 错误处理评分
        has_try_except = any(isinstance(node, ast.Try) for node in ast.walk(tree))
        if not has_try_except:
            scores['error_handling_score'] -= 0.3
        
        # 确保所有分数在 0-1 之间
        return {k: max(0.0, min(1.0, v)) for k, v in scores.items()}
    except:
        return {k: 0.0 for k in ['naming_score', 'complexity_score', 
                                'documentation_score', 'error_handling_score']}

def code_reward(
    completions: List[Dict],
    test_cases: List[Dict],
    custom_validator: Callable = None,
    weights: Dict[str, float] = None
) -> List[float]:
    """通用代码奖励函数。
    
    Args:
        completions: 模型生成的代码列表
        test_cases: 测试用例列表，每个包含 input 和 expected_output
        custom_validator: 自定义验证函数
        weights: 各项评分的权重，默认值：
                {'correctness': 0.7,
                 'quality': 0.3}
    
    Returns:
        List[float]: 奖励值列表
    """
    if weights is None:
        weights = {
            'correctness': 0.7,  # 答案正确性权重
            'quality': 0.3       # 代码质量权重
        }
    
    rewards = []
    
    for completion in completions:
        code = completion[0]["content"]
        
        # 1. 答案正确性评分 (70%)
        correctness_scores = []
        for test_case in test_cases:
            result = run_code_safely(code, test_case["input"])
            if result['error']:
                correctness_scores.append(0.0)
                continue
                
            is_correct = verify_output_correctness(
                result['output'],
                test_case["expected_output"],
                custom_validator
            )
            correctness_scores.append(1.0 if is_correct else 0.0)
        
        correctness_score = sum(correctness_scores) / len(correctness_scores)
        
        # 2. 代码质量评分 (30%)
        quality_scores = analyze_code_quality(code)
        quality_score = sum(quality_scores.values()) / len(quality_scores)
        
        # 3. 计算总分
        total_score = (
            weights['correctness'] * correctness_score +
            weights['quality'] * quality_score
        )
        
        rewards.append(total_score)
    
    return rewards

# 示例：特定问题的验证器
def validate_boys_girls_arrangement(output: str, expected_output: str) -> bool:
    """Boys and Girls 问题的特定验证器"""
    output = output.strip()
    n = output.count('B')
    m = output.count('G')
    
    # 1. 验证字符
    if not all(c in 'BG' for c in output):
        return False
    
    # 2. 验证交替最大化
    alternations = sum(1 for i in range(len(output)-1) 
                      if output[i] != output[i+1])
    
    # 计算理论最大交替次数
    max_possible_alternations = 2 * min(n, m) - 1 if n != m else 2 * min(n, m)
    
    return alternations >= max_possible_alternations

# 示例测试用例
SAMPLE_TEST_CASES = [
    {
        "input": "3 3\n",
        "expected_output": "GBGBGB\n"
    },
    {
        "input": "4 2\n",
        "expected_output": "BGBGBB\n"
    }
]
