"""
算法模式检测器
包含各种算法模式的具体检测逻辑
"""

import ast
from typing import Any

def has_binary_search_pattern(code: str) -> bool:
    """检测是否包含二分查找的典型模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_mid_calc = False
    has_comparison = False
    
    for child in ast.walk(node):
        # 检测 mid = (left + right) // 2 模式
        if isinstance(child, ast.Assign):
            if isinstance(child.value, ast.BinOp):
                if isinstance(child.value.op, ast.FloorDiv):
                    has_mid_calc = True
        
        # 检测比较操作
        if isinstance(child, ast.Compare):
            has_comparison = True
    
    return has_mid_calc and has_comparison

def has_dp_pattern(code: str) -> bool:
    """检测是否包含动态规划的典型模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_array_init = False
    has_nested_loops = False
    has_array_update = False
    
    for child in ast.walk(node):
        # 检测数组初始化
        if isinstance(child, ast.Assign):
            if isinstance(child.value, (ast.List, ast.ListComp)):
                has_array_init = True
        
        # 检测嵌套循环
        if isinstance(child, ast.For):
            for subchild in ast.walk(child):
                if isinstance(subchild, ast.For):
                    has_nested_loops = True
        
        # 检测数组更新
        if isinstance(child, ast.Subscript):
            has_array_update = True
    
    return has_array_init and has_nested_loops and has_array_update

def has_sorting_pattern(code: str) -> bool:
    """检测排序算法模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_comparison = False
    has_swap = False
    has_partition = False
    
    for child in ast.walk(node):
        # 检测比较操作
        if isinstance(child, ast.Compare):
            has_comparison = True
        
        # 检测交换操作 (a, b = b, a 或类似模式)
        if isinstance(child, ast.Assign):
            if isinstance(child.value, ast.Tuple):
                has_swap = True
        
        # 检测分区操作 (快速排序特征)
        if isinstance(child, ast.Assign):
            target = child.targets[0] if child.targets else None
            if isinstance(target, ast.Subscript):
                has_partition = True
    
    return has_comparison and (has_swap or has_partition)

def has_graph_pattern(code: str) -> bool:
    """检测图算法模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_adjacency = False
    has_visited = False
    has_graph_ops = False
    
    for child in ast.walk(node):
        # 检测邻接表/矩阵
        if isinstance(child, ast.Name):
            if 'graph' in child.id.lower() or 'adj' in child.id.lower():
                has_adjacency = True
        
        # 检测访问标记
        if isinstance(child, ast.Name):
            if 'visit' in child.id.lower():
                has_visited = True
        
        # 检测图操作
        if isinstance(child, ast.Call):
            if hasattr(child.func, 'id'):
                if any(op in child.func.id.lower() for op in ['dfs', 'bfs', 'path', 'connect']):
                    has_graph_ops = True
    
    return has_adjacency and (has_visited or has_graph_ops)

def has_tree_pattern(code: str) -> bool:
    """检测树算法模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_node_class = False
    has_tree_ops = False
    
    for child in ast.walk(node):
        # 检测节点类定义
        if isinstance(child, ast.ClassDef):
            if 'node' in child.name.lower():
                has_node_class = True
        
        # 检测树操作
        if isinstance(child, ast.Call):
            if hasattr(child.func, 'id'):
                if any(op in child.func.id.lower() for op in ['insert', 'delete', 'traverse', 'balance']):
                    has_tree_ops = True
    
    return has_node_class or has_tree_ops

def has_backtracking_pattern(code: str) -> bool:
    """检测回溯算法模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_state_tracking = False
    has_recursive_call = False
    has_state_restore = False
    
    function_name = None
    
    for child in ast.walk(node):
        # 获取函数名
        if isinstance(child, ast.FunctionDef):
            function_name = child.name
            
        # 检测状态跟踪
        if isinstance(child, ast.Assign):
            if isinstance(child.targets[0], ast.Subscript):
                has_state_tracking = True
        
        # 检测递归调用
        if isinstance(child, ast.Call):
            if hasattr(child.func, 'id'):
                if function_name and child.func.id == function_name:
                    has_recursive_call = True
        
        # 检测状态恢复
        if isinstance(child, ast.Assign):
            if isinstance(child.value, ast.Subscript):
                has_state_restore = True
    
    return has_recursive_call and (has_state_tracking or has_state_restore)

def has_greedy_pattern(code: str) -> bool:
    """检测贪心算法模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_sorting = False
    has_local_optimal = False
    
    for child in ast.walk(node):
        # 检测排序操作
        if isinstance(child, ast.Call):
            if hasattr(child.func, 'id') and 'sort' in child.func.id.lower():
                has_sorting = True
        
        # 检测局部最优选择
        if isinstance(child, ast.Compare):
            has_local_optimal = True
    
    return has_sorting or has_local_optimal

def has_divide_conquer_pattern(code: str) -> bool:
    """检测分治算法模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_recursive_call = False
    has_merge_operation = False
    
    for child in ast.walk(node):
        # 检测递归调用
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                # 如果是函数调用，检查是否是递归调用
                for func_def in ast.walk(node):
                    if isinstance(func_def, ast.FunctionDef) and func_def.name == child.func.id:
                        has_recursive_call = True
                        break
        
        # 检测合并操作（通常涉及列表或数组的操作）
        if isinstance(child, (ast.BinOp, ast.List, ast.ListComp)):
            has_merge_operation = True
    
    return has_recursive_call and has_merge_operation

def has_sliding_window_pattern(code: str) -> bool:
    """检测滑动窗口模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_window_vars = False
    has_window_update = False
    
    for child in ast.walk(node):
        # 检测窗口变量
        if isinstance(child, ast.Name):
            if any(var in child.id.lower() for var in ['left', 'right', 'start', 'end', 'window']):
                has_window_vars = True
        
        # 检测窗口更新
        if isinstance(child, ast.AugAssign) or isinstance(child, ast.Assign):
            has_window_update = True
    
    return has_window_vars and has_window_update

def has_trie_pattern(code: str) -> bool:
    """检测字典树模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_node_structure = False
    has_char_processing = False
    
    for child in ast.walk(node):
        # 检测节点结构
        if isinstance(child, ast.Dict):
            has_node_structure = True
        
        # 检测字符处理
        if isinstance(child, ast.Subscript):
            if isinstance(child.value, ast.Name):
                if 'children' in child.value.id.lower():
                    has_char_processing = True
    
    return has_node_structure and has_char_processing

def has_union_find_pattern(code: str) -> bool:
    """检测并查集模式"""
    try:
        node = ast.parse(code)
    except:
        return False
        
    has_parent_array = False
    has_union_find_ops = False
    
    for child in ast.walk(node):
        # 检测父节点数组
        if isinstance(child, ast.Name):
            if any(name in child.id.lower() for name in ['parent', 'root', 'disjoint']):
                has_parent_array = True
        
        # 检测并查集操作
        if isinstance(child, ast.FunctionDef):
            if any(op in child.name.lower() for op in ['find', 'union', 'connect']):
                has_union_find_ops = True
    
    return has_parent_array or has_union_find_ops

def has_math_pattern(code_str):
    """检测数学相关问题的模式
    - 数学运算和函数
    - 数学常量
    - 数学公式计算
    """
    try:
        tree = ast.parse(code_str)
        math_ops = {'pow', 'sqrt', 'abs', 'min', 'max', 'sum', 'factorial', 'gcd', 'lcm'}
        math_constants = {'pi', 'e', 'inf', 'nan'}
        math_imports = {'math', 'numpy', 'scipy'}
        
        class MathVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_math = False
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in math_ops:
                    self.has_math = True
                self.generic_visit(node)
                
            def visit_Name(self, node):
                if node.id in math_constants:
                    self.has_math = True
                self.generic_visit(node)
                
            def visit_Import(self, node):
                for name in node.names:
                    if name.name in math_imports:
                        self.has_math = True
                self.generic_visit(node)
                
            def visit_ImportFrom(self, node):
                if node.module in math_imports:
                    self.has_math = True
                self.generic_visit(node)
        
        visitor = MathVisitor()
        visitor.visit(tree)
        return visitor.has_math
    except:
        return False

def has_string_pattern(code_str):
    """检测字符串处理相关的模式
    - 字符串操作
    - 正则表达式
    - 字符串算法
    """
    try:
        tree = ast.parse(code_str)
        string_methods = {'split', 'join', 'strip', 'replace', 'find', 'count', 'startswith', 'endswith'}
        string_imports = {'re', 'string'}
        
        class StringVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_string_ops = False
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Str):
                    if node.func.attr in string_methods:
                        self.has_string_ops = True
                self.generic_visit(node)
                
            def visit_Import(self, node):
                for name in node.names:
                    if name.name in string_imports:
                        self.has_string_ops = True
                self.generic_visit(node)
        
        visitor = StringVisitor()
        visitor.visit(tree)
        return visitor.has_string_ops
    except:
        return False

def has_constructive_pattern(code_str):
    """检测构造性算法的模式
    - 构造数据结构
    - 构造解决方案
    """
    try:
        tree = ast.parse(code_str)
        
        class ConstructiveVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_constructive = False
                self.list_constructions = 0
                self.assignments = 0
                
            def visit_List(self, node):
                self.list_constructions += 1
                if self.list_constructions > 2:  # 多次构造列表可能是构造性算法的特征
                    self.has_constructive = True
                self.generic_visit(node)
                
            def visit_Assign(self, node):
                self.assignments += 1
                if self.assignments > 5:  # 多次赋值可能是在构造解决方案
                    self.has_constructive = True
                self.generic_visit(node)
        
        visitor = ConstructiveVisitor()
        visitor.visit(tree)
        return visitor.has_constructive
    except:
        return False

def has_number_theory_pattern(code_str):
    """检测数论相关的模式
    - 素数
    - GCD/LCM
    - 模运算
    """
    try:
        tree = ast.parse(code_str)
        number_theory_funcs = {'gcd', 'lcm', 'is_prime', 'prime_factors'}
        
        class NumberTheoryVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_number_theory = False
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in number_theory_funcs:
                    self.has_number_theory = True
                self.generic_visit(node)
                
            def visit_BinOp(self, node):
                if isinstance(node.op, ast.Mod):  # 检测模运算
                    self.has_number_theory = True
                self.generic_visit(node)
        
        visitor = NumberTheoryVisitor()
        visitor.visit(tree)
        return visitor.has_number_theory
    except:
        return False

def has_geometry_pattern(code_str):
    """检测几何相关的模式
    - 点、线、面的计算
    - 几何函数
    """
    try:
        tree = ast.parse(code_str)
        geometry_funcs = {'cos', 'sin', 'tan', 'atan2', 'hypot', 'degrees', 'radians'}
        geometry_imports = {'math', 'numpy'}
        
        class GeometryVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_geometry = False
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in geometry_funcs:
                    self.has_geometry = True
                self.generic_visit(node)
                
            def visit_Import(self, node):
                for name in node.names:
                    if name.name in geometry_imports:
                        self.has_geometry = True
                self.generic_visit(node)
        
        visitor = GeometryVisitor()
        visitor.visit(tree)
        return visitor.has_geometry
    except:
        return False

def has_two_pointers_pattern(code_str):
    """检测双指针模式
    - 两个变量同时遍历
    - 滑动窗口
    """
    try:
        tree = ast.parse(code_str)
        
        class TwoPointersVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_two_pointers = False
                self.pointer_vars = set()
                
            def visit_For(self, node):
                if isinstance(node.target, ast.Name):
                    self.pointer_vars.add(node.target.id)
                    if len(self.pointer_vars) >= 2:
                        self.has_two_pointers = True
                self.generic_visit(node)
                
            def visit_While(self, node):
                # 检查while循环中是否有两个指针的移动
                if isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name) and isinstance(node.test.comparators[0], ast.Name):
                        self.has_two_pointers = True
                self.generic_visit(node)
        
        visitor = TwoPointersVisitor()
        visitor.visit(tree)
        return visitor.has_two_pointers
    except:
        return False

def has_bitmask_pattern(code_str):
    """检测位运算模式
    - 位运算操作
    - 位掩码
    """
    try:
        tree = ast.parse(code_str)
        
        class BitmaskVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_bitmask = False
                
            def visit_BinOp(self, node):
                if isinstance(node.op, (ast.BitOr, ast.BitAnd, ast.BitXor, ast.LShift, ast.RShift)):
                    self.has_bitmask = True
                self.generic_visit(node)
                
            def visit_UnaryOp(self, node):
                if isinstance(node.op, ast.Invert):
                    self.has_bitmask = True
                self.generic_visit(node)
        
        visitor = BitmaskVisitor()
        visitor.visit(tree)
        return visitor.has_bitmask
    except:
        return False

def has_dsu_pattern(code_str):
    """检测并查集模式
    - Union-Find数据结构
    - 集合合并操作
    """
    try:
        tree = ast.parse(code_str)
        dsu_methods = {'find', 'union', 'find_set', 'union_sets', 'make_set'}
        
        class DSUVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_dsu = False
                
            def visit_FunctionDef(self, node):
                if node.name in dsu_methods:
                    self.has_dsu = True
                self.generic_visit(node)
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in dsu_methods:
                    self.has_dsu = True
                self.generic_visit(node)
        
        visitor = DSUVisitor()
        visitor.visit(tree)
        return visitor.has_dsu
    except:
        return False

def has_hash_pattern(code_str):
    """检测哈希相关的模式
    - 哈希函数
    - 哈希表操作
    """
    try:
        tree = ast.parse(code_str)
        hash_funcs = {'hash', 'hashlib', '__hash__'}
        
        class HashVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_hash = False
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in hash_funcs:
                    self.has_hash = True
                self.generic_visit(node)
                
            def visit_Dict(self, node):
                if len(node.keys) > 3:  # 使用字典可能表示在使用哈希表
                    self.has_hash = True
                self.generic_visit(node)
        
        visitor = HashVisitor()
        visitor.visit(tree)
        return visitor.has_hash
    except:
        return False

def has_matrix_pattern(code_str):
    """检测矩阵相关的模式
    - 二维数组操作
    - 矩阵运算
    """
    try:
        tree = ast.parse(code_str)
        matrix_imports = {'numpy', 'scipy.linalg'}
        
        class MatrixVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_matrix = False
                self.nested_lists = 0
                
            def visit_List(self, node):
                if any(isinstance(elt, ast.List) for elt in node.elts):
                    self.nested_lists += 1
                    if self.nested_lists > 1:
                        self.has_matrix = True
                self.generic_visit(node)
                
            def visit_Import(self, node):
                for name in node.names:
                    if name.name in matrix_imports:
                        self.has_matrix = True
                self.generic_visit(node)
        
        visitor = MatrixVisitor()
        visitor.visit(tree)
        return visitor.has_matrix
    except:
        return False

def has_fft_pattern(code_str):
    """检测FFT变换相关的模式
    - FFT函数调用
    - 频域变换
    """
    try:
        tree = ast.parse(code_str)
        fft_funcs = {'fft', 'ifft', 'rfft', 'irfft'}
        fft_imports = {'numpy.fft', 'scipy.fft'}
        
        class FFTVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_fft = False
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in fft_funcs:
                    self.has_fft = True
                self.generic_visit(node)
                
            def visit_Import(self, node):
                for name in node.names:
                    if name.name in fft_imports:
                        self.has_fft = True
                self.generic_visit(node)
                
            def visit_ImportFrom(self, node):
                if node.module in fft_imports:
                    self.has_fft = True
                self.generic_visit(node)
        
        visitor = FFTVisitor()
        visitor.visit(tree)
        return visitor.has_fft
    except:
        return False

feature_detectors = {
    'binary_search': has_binary_search_pattern,
    'dp': has_dp_pattern,
    'sorting': has_sorting_pattern,
    'graph': has_graph_pattern,
    'tree': has_tree_pattern,
    'backtracking': has_backtracking_pattern,
    'greedy': has_greedy_pattern,
    'divide_conquer': has_divide_conquer_pattern,
    'sliding_window': has_sliding_window_pattern,
    'trie': has_trie_pattern,
    'union_find': has_union_find_pattern
}

feature_detectors.update({
    'math': has_math_pattern,
    'string': has_string_pattern,
    'constructive': has_constructive_pattern,
    'number_theory': has_number_theory_pattern,
    'geometry': has_geometry_pattern,
    'two_pointers': has_two_pointers_pattern,
    'bitmask': has_bitmask_pattern,
    'dsu': has_dsu_pattern,
    'hash': has_hash_pattern,
    'matrix': has_matrix_pattern,
    'fft': has_fft_pattern
})
