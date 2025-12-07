"""
Advanced AST-based security pattern detection.
This provides a static analysis layer to catch malicious patterns that might bypass LLM review.
"""
import ast
from typing import List, Tuple


class SecurityPatternDetector(ast.NodeVisitor):
    """
    Detects suspicious patterns in Python code using AST analysis.
    This acts as a backup to LLM-based security review.
    
    The following patterns are detected:
    - Dangerous builtins (exec, eval, compile, __import__)
    - Potential obfuscation (base64/hex decoding)
    - System command execution (subprocess, os.system)
    - Environment variable access
    - Infinite loops
    - Path traversal
    """
    
    def __init__(self):
        self.violations: List[Tuple[str, str]] = []
        self.risk_score = 0
        
    def visit_Call(self, node: ast.Call) -> None:
        """Detect suspicious function calls."""
        func_name = self._get_func_name(node.func)
        
        # Detect encoding/decoding that could hide malicious code
        if func_name in ['exec', 'eval', 'compile', '__import__']:
            self.violations.append(('CRITICAL', f'Dangerous builtin: {func_name}'))
            self.risk_score += 10
            
        # Detect base64/hex decoding (common obfuscation)
        if func_name in ['b64decode', 'fromhex', 'unhexlify', 'decode']:
            self.violations.append(('HIGH', f'Potential obfuscation: {func_name}'))
            self.risk_score += 5
            
        # Detect subprocess/os.system calls
        if func_name in ['system', 'popen', 'spawn', 'execv', 'execl']:
            self.violations.append(('CRITICAL', f'System command execution: {func_name}'))
            self.risk_score += 10
            
        # Detect environment variable access
        if func_name == 'getenv' or (hasattr(node.func, 'attr') and 
                                     node.func.attr == 'environ'):
            self.violations.append(('HIGH', 'Environment variable access detected'))
            self.risk_score += 7
            
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Detect suspicious attribute access."""
        # Detect os.environ, os.system, etc.
        if isinstance(node.value, ast.Name):
            if node.value.id == 'os':
                if node.attr in [
                    'system', 'popen', 'environ', 'execv', 'execl',
                    'spawn', 'remove', 'rmdir', 'unlink'
                ]:
                    self.violations.append(
                        ('CRITICAL', f'Dangerous os.{node.attr} access')
                    )
                    self.risk_score += 10
                    
        # Detect __dict__, __class__, __bases__ (introspection for exploits)
        if node.attr in [
            '__dict__', '__class__', '__bases__', '__subclasses__',
            '__globals__', '__code__', '__builtins__'
        ]:
            self.violations.append(('MEDIUM', f'Introspection detected: {node.attr}'))
            self.risk_score += 3
            
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Detect suspicious imports."""
        for alias in node.names:
            # Subprocess module
            if alias.name in ['subprocess', 'multiprocessing', 'threading']:
                self.violations.append(('HIGH', f'Process/thread module: {alias.name}'))
                self.risk_score += 5
                
            # Network modules
            if alias.name in ['socket', 'urllib', 'http']:
                self.violations.append(('MEDIUM', f'Network module: {alias.name}'))
                self.risk_score += 2
                
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Detect suspicious from-imports."""
        module = node.module or ''
        
        # Subprocess module
        if module in ['subprocess', 'multiprocessing', 'threading']:
            self.violations.append(('HIGH', f'Process/thread module: from {module}'))
            self.risk_score += 5
            
        # Network modules
        if module in ['socket', 'urllib', 'http']:
            self.violations.append(('MEDIUM', f'Network module: from {module}'))
            self.risk_score += 2
            
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While) -> None:
        """Detect potential infinite loops."""
        # Check for while True without break
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            has_break = self._has_break(node.body)
            if not has_break:
                self.violations.append(
                    ('HIGH', 'Potential infinite loop: while True without break')
                )
                self.risk_score += 5
                
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For) -> None:
        """Detect suspicious loops."""
        # Detect very large ranges
        if isinstance(node.iter, ast.Call):
            func_name = self._get_func_name(node.iter.func)
            if func_name == 'range' and node.iter.args:
                if isinstance(node.iter.args[0], ast.Constant):
                    if isinstance(
                        node.iter.args[0].value, int
                    ) and node.iter.args[0].value > 1000000:
                        self.violations.append(
                            ('MEDIUM', f'Large loop range: {node.iter.args[0].value}')
                        )
                        self.risk_score += 3
                        
        self.generic_visit(node)
    
    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Detect suspicious operations."""
        # Detect large memory allocations
        if isinstance(node.op, ast.Mult):
            if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                left_val = node.left.value
                right_val = node.right.value
                
                # String multiplication: 'A' * 1000000 or 1000000 * 'A'
                if isinstance(left_val, str) and isinstance(right_val, int):
                    if right_val > 100_000_000:  # 100MB
                        self.violations.append(
                            ('HIGH', f'Large memory allocation: string * {right_val}')
                        )
                        self.risk_score += 5
                elif isinstance(left_val, int) and isinstance(right_val, str):
                    if left_val > 100_000_000:  # 100MB
                        self.violations.append(
                            ('HIGH', f'Large memory allocation: {left_val} * string')
                        )
                        self.risk_score += 5
                    
        self.generic_visit(node)
    
    def visit_Str(self, node: ast.Str) -> None:
        """Detect suspicious strings (Python < 3.8)."""
        self._check_string_content(node.s)
        self.generic_visit(node)
    
    def visit_Constant(self, node: ast.Constant) -> None:
        """Detect suspicious constants."""
        if isinstance(node.value, str):
            self._check_string_content(node.value)
        self.generic_visit(node)
    
    def _check_string_content(self, s: str) -> None:
        """Check string for suspicious patterns."""
        # Detect shell commands - be more specific to avoid false positives
        # Only flag if it looks like an actual shell command, not just the word
        dangerous_patterns = [
            ('rm -rf /', 'Dangerous rm command'),
            ('rm -rf ~', 'Dangerous rm command'),
            ('dd if=/dev', 'Dangerous dd command'),
            ('mkfs.', 'Dangerous mkfs command'),
            ('del /f /s /q', 'Dangerous Windows delete command'),
            ('rmdir /s /q', 'Dangerous Windows rmdir command'),
        ]
        
        s_lower = s.lower()
        for pattern, description in dangerous_patterns:
            if pattern in s_lower:
                self.violations.append(
                    ('CRITICAL', f'{description}: {pattern}')
                )
                self.risk_score += 10
                
        # Detect path traversal - but only if it looks suspicious
        # Allow relative paths in URLs and normal file operations
        if ('../../../' in s or '..\\..\\..\\' in s):
            # Only flag if it's trying to traverse multiple levels
            self.violations.append(('HIGH', 'Deep path traversal pattern detected'))
            self.risk_score += 5
    
    def _get_func_name(self, node: ast.AST) -> str:
        """Extract function name from various node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ''
    
    def _has_break(self, body: List[ast.stmt]) -> bool:
        """Check if a code block contains a break statement."""
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Break):
                return True
        return False


def analyze_code_patterns(code: str) -> Tuple[bool, str, int]:
    """
    Perform AST-based pattern analysis on code.
    
    Args:
        code: Python source code to analyze
        
    Returns:
        Tuple of (is_safe, reason, risk_score)
        - is_safe: False if critical violations found
        - reason: Description of violations
        - risk_score: Numeric risk score (0-100)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f'Syntax error: {str(e)}', 100
    
    detector = SecurityPatternDetector()
    detector.visit(tree)
    
    # Determine if code is safe based on violations
    critical_violations = [v for v in detector.violations if v[0] == 'CRITICAL']
    
    if critical_violations:
        reasons = '; '.join([v[1] for v in critical_violations])
        return False, f'Critical security violations: {reasons}', detector.risk_score
    
    if detector.risk_score > 15:  # Threshold for multiple high-risk patterns
        reasons = '; '.join([v[1] for v in detector.violations])
        return False, f'High risk score ({detector.risk_score}): {reasons}', detector.risk_score
    
    if detector.violations:
        reasons = '; '.join([v[1] for v in detector.violations])
        return True, f'Minor concerns detected: {reasons}', detector.risk_score
    
    return True, 'No suspicious patterns detected', 0
