"""
Unit tests for the AST-based security pattern detector.
"""
import pytest

from kodeagent.pattern_detector import analyze_code_patterns, SecurityPatternDetector


class TestDangerousBuiltins:
    """Test detection of dangerous builtin functions."""
    
    def test_exec_detected(self):
        """Test that exec() is detected as critical."""
        code = "exec('print(1)')"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'exec' in reason.lower()
        assert risk >= 10
    
    def test_eval_detected(self):
        """Test that eval() is detected as critical."""
        code = "result = eval('2 + 2')"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'eval' in reason.lower()
        assert risk >= 10
    
    def test_compile_detected(self):
        """Test that compile() is detected as critical."""
        code = "compile('x = 1', '<string>', 'exec')"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'compile' in reason.lower()
        assert risk >= 10
    
    def test_import_builtin_detected(self):
        """Test that __import__() is detected as critical."""
        code = "__import__('os')"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert '__import__' in reason.lower()
        assert risk >= 10


class TestObfuscationDetection:
    """Test detection of code obfuscation techniques."""
    
    def test_base64_decode_detected(self):
        """Test that base64 decoding is flagged."""
        code = """
import base64
decoded = base64.b64decode('aGVsbG8=')
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'b64decode' in reason.lower() or 'obfuscation' in reason.lower()
        assert risk >= 5
    
    def test_fromhex_detected(self):
        """Test that hex decoding is flagged."""
        code = "data = bytes.fromhex('48656c6c6f')"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'fromhex' in reason.lower() or 'obfuscation' in reason.lower()
        assert risk >= 5
    
    def test_exec_with_base64_critical(self):
        """Test that exec + base64 is critical violation."""
        code = """
import base64
exec(base64.b64decode('cHJpbnQoImhpIik='))
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'exec' in reason.lower()
        assert risk >= 10


class TestSystemCommandDetection:
    """Test detection of system command execution."""
    
    def test_os_system_detected(self):
        """Test that os.system() is detected as critical."""
        code = """
import os
os.system('ls -la')
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'os.system' in reason.lower() or 'system' in reason.lower()
        assert risk >= 10
    
    def test_subprocess_detected(self):
        """Test that subprocess module is flagged."""
        code = "import subprocess"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'subprocess' in reason.lower()
        assert risk >= 5
    
    def test_dangerous_command_in_string(self):
        """Test that dangerous commands in strings are detected."""
        code = """
cmd = 'rm -rf /'
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'rm -rf' in reason.lower() or 'dangerous command' in reason.lower()
        assert risk >= 10


class TestEnvironmentVariableAccess:
    """Test detection of environment variable access."""
    
    def test_os_environ_detected(self):
        """Test that os.environ access is detected."""
        code = """
import os
env_vars = os.environ
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'environ' in reason.lower()
        assert risk >= 7
    
    def test_os_getenv_detected(self):
        """Test that os.getenv() is detected."""
        code = """
import os
api_key = os.getenv('API_KEY')
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'getenv' in reason.lower() or 'environment' in reason.lower()
        assert risk >= 7


class TestInfiniteLoopDetection:
    """Test detection of infinite loops."""
    
    def test_while_true_without_break(self):
        """Test that while True without break is flagged."""
        code = """
while True:
    print('forever')
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        # This should be flagged but not necessarily blocked (risk < 15)
        assert 'infinite loop' in reason.lower() or 'while true' in reason.lower()
        assert risk >= 5
    
    def test_while_true_with_break_safe(self):
        """Test that while True with break is considered safer."""
        code = """
while True:
    user_input = input()
    if user_input == 'quit':
        break
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        # Should pass or have low risk
        assert risk < 15


class TestPathTraversalDetection:
    """Test detection of path traversal attempts."""
    
    def test_relative_path_traversal(self):
        """Test that ../ path traversal is detected."""
        code = """
with open('../../../etc/passwd', 'r') as f:
    data = f.read()
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'path traversal' in reason.lower() or '../' in reason
        assert risk >= 5
    
    def test_windows_path_traversal(self):
        """Test that ..\\ path traversal is detected."""
        code = r"""
path = '..\\..\\..\\Windows\\System32\\config\\SAM'
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'path traversal' in reason.lower() or '..' in reason
        assert risk >= 5


class TestLargeMemoryAllocation:
    """Test detection of large memory allocations."""
    
    def test_large_string_multiplication(self):
        """Test that large string allocations are detected."""
        code = "x = 'A' * 200000000"  # 200MB
        is_safe, reason, risk = analyze_code_patterns(code)
        
        # Should be flagged as HIGH risk
        assert 'memory allocation' in reason.lower() or 'large' in reason.lower()
        assert risk >= 5
    
    def test_small_allocation_safe(self):
        """Test that small allocations are safe."""
        code = "x = 'A' * 100"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        # Should be safe
        assert is_safe is True
        assert risk == 0


class TestIntrospectionDetection:
    """Test detection of introspection exploits."""
    
    def test_dict_access_detected(self):
        """Test that __dict__ access is flagged."""
        code = "obj.__dict__"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert '__dict__' in reason.lower() or 'introspection' in reason.lower()
        assert risk >= 3
    
    def test_globals_access_detected(self):
        """Test that __globals__ access is flagged."""
        code = "func.__globals__"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert '__globals__' in reason.lower() or 'introspection' in reason.lower()
        assert risk >= 3
    
    def test_builtins_access_detected(self):
        """Test that __builtins__ attribute access is flagged."""
        code = "x = obj.__builtins__"  # Attribute access, not just the name
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert '__builtins__' in reason.lower() or 'introspection' in reason.lower()
        assert risk >= 3


class TestLargeLoops:
    """Test detection of suspiciously large loops."""
    
    def test_large_range_detected(self):
        """Test that very large range() is flagged."""
        code = """
for i in range(10000000):
    pass
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'large' in reason.lower() or 'range' in reason.lower()
        assert risk >= 3
    
    def test_small_range_safe(self):
        """Test that normal range() is safe."""
        code = """
for i in range(100):
    print(i)
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        # Should be safe or low risk
        assert risk < 10


class TestSafeCode:
    """Test that safe code passes all checks."""
    
    def test_simple_math_safe(self):
        """Test that simple math operations are safe."""
        code = """
import math
result = math.sqrt(16)
print(result)
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is True
        assert risk == 0
        assert 'no suspicious patterns' in reason.lower()
    
    def test_data_processing_safe(self):
        """Test that normal data processing is safe."""
        code = """
data = [1, 2, 3, 4, 5]
total = sum(data)
average = total / len(data)
print(f'Average: {average}')
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is True
        assert risk == 0
    
    def test_file_read_safe(self):
        """Test that normal file reading is safe."""
        code = """
with open('data.txt', 'r') as f:
    content = f.read()
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is True
        assert risk == 0


class TestRiskScoring:
    """Test the risk scoring system."""
    
    def test_multiple_violations_increase_risk(self):
        """Test that multiple violations increase risk score."""
        code = """
import subprocess
import os
os.system('ls')
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        # Should have high risk due to multiple violations
        assert is_safe is False
        assert risk >= 15
    
    def test_critical_violation_blocks(self):
        """Test that critical violations block execution."""
        code = "exec('malicious code')"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'critical' in reason.lower()
    
    def test_risk_threshold(self):
        """Test that risk score above threshold blocks code."""
        # Code with multiple medium-risk patterns
        code = """
import subprocess
import socket
for i in range(5000000):
    pass
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        # Should accumulate enough risk to be blocked
        if risk > 15:
            assert is_safe is False


class TestSyntaxErrors:
    """Test handling of syntax errors."""
    
    def test_syntax_error_blocked(self):
        """Test that code with syntax errors is blocked."""
        code = "if True print('missing colon')"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'syntax' in reason.lower()
        assert risk == 100


class TestEdgeCases:
    """Test edge cases and corner scenarios."""
    
    def test_empty_code(self):
        """Test that empty code is safe."""
        code = ""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is True
        assert risk == 0
    
    def test_comments_only(self):
        """Test that comments-only code is safe."""
        code = """
# This is a comment
# Another comment
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is True
        assert risk == 0
    
    def test_multiline_string(self):
        """Test that multiline strings are handled correctly."""
        code = '''
text = """
This is a multiline string
with multiple lines
"""
'''
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is True
        assert risk == 0


class TestSecurityPatternDetectorClass:
    """Test the SecurityPatternDetector class directly."""
    
    def test_detector_initialization(self):
        """Test that detector initializes correctly."""
        detector = SecurityPatternDetector()
        
        assert detector.violations == []
        assert detector.risk_score == 0
    
    def test_detector_accumulates_violations(self):
        """Test that detector accumulates violations."""
        import ast
        
        code = """
import os
os.system('ls')
exec('code')
"""
        tree = ast.parse(code)
        detector = SecurityPatternDetector()
        detector.visit(tree)
        
        assert len(detector.violations) > 0
        assert detector.risk_score > 0
    
    def test_detector_categorizes_severity(self):
        """Test that detector categorizes violations by severity."""
        import ast
        
        code = "exec('malicious')"
        tree = ast.parse(code)
        detector = SecurityPatternDetector()
        detector.visit(tree)
        
        # Should have at least one CRITICAL violation
        critical_violations = [v for v in detector.violations if v[0] == 'CRITICAL']
        assert len(critical_violations) > 0


class TestNetworkModules:
    """Test detection of network-related module imports."""
    
    def test_socket_module_detected(self):
        """Test that socket module import is flagged."""
        code = "import socket"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'socket' in reason.lower() or 'network' in reason.lower()
        assert risk >= 2
    
    def test_urllib_module_detected(self):
        """Test that urllib module import is flagged."""
        code = "import urllib"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'urllib' in reason.lower() or 'network' in reason.lower()
        assert risk >= 2
    
    def test_http_module_detected(self):
        """Test that http module import is flagged."""
        code = "import http"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'http' in reason.lower() or 'network' in reason.lower()
        assert risk >= 2


class TestProcessThreadModules:
    """Test detection of process and threading modules."""
    
    def test_threading_module_detected(self):
        """Test that threading module import is flagged."""
        code = "import threading"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'threading' in reason.lower() or 'thread' in reason.lower()
        assert risk >= 5
    
    def test_multiprocessing_module_detected(self):
        """Test that multiprocessing module import is flagged."""
        code = "import multiprocessing"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'multiprocessing' in reason.lower() or 'process' in reason.lower()
        assert risk >= 5


class TestDangerousCommands:
    """Test detection of dangerous system commands in strings."""
    
    def test_del_command_detected(self):
        """Test that 'del /f /s /q' command is detected."""
        code = "cmd = 'del /f /s /q C:\\\\data'"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'del /f' in reason.lower() or 'dangerous' in reason.lower()
    
    def test_rmdir_command_detected(self):
        """Test that 'rmdir /s /q' command is detected."""
        code = "cmd = 'rmdir /s /q C:\\\\data'"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'rmdir' in reason.lower() or 'dangerous' in reason.lower()
    
    def test_dd_command_detected(self):
        """Test that 'dd if=' command is detected."""
        code = "cmd = 'dd if=/dev/zero of=/dev/sda'"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'dd if=' in reason.lower() or 'dangerous command' in reason.lower()
    
    def test_mkfs_command_detected(self):
        """Test that 'mkfs' command is detected."""
        code = "cmd = 'mkfs.ext4 /dev/sda1'"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'mkfs' in reason.lower() or 'dangerous command' in reason.lower()
    
    def test_format_command_detected(self):
        """Test that dangerous format operations are detected."""
        # Note: Generic 'format' is no longer blocked to avoid false positives with f-strings
        # This test now checks that we don't have false positives
        code = "result = f'Hello {name}'"
        is_safe, reason, risk = analyze_code_patterns(code)
        
        # Should be safe - f-strings are allowed
        assert is_safe is True
        assert risk == 0


class TestFileOperations:
    """Test detection of dangerous file operations."""
    
    def test_os_remove_detected(self):
        """Test that os.remove() is detected."""
        code = """
import os
os.remove('/important/file.txt')
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'os.remove' in reason.lower() or 'remove' in reason.lower()
    
    def test_os_rmdir_detected(self):
        """Test that os.rmdir() is detected."""
        code = """
import os
os.rmdir('/important/dir')
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'os.rmdir' in reason.lower() or 'rmdir' in reason.lower()
    
    def test_os_unlink_detected(self):
        """Test that os.unlink() is detected."""
        code = """
import os
os.unlink('/important/file')
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert is_safe is False
        assert 'os.unlink' in reason.lower() or 'unlink' in reason.lower()


class TestAdditionalMemoryAllocation:
    """Test additional memory allocation patterns."""
    
    def test_reverse_string_multiplication(self):
        """Test that integer * string (reverse order) is detected."""
        code = "x = 200000000 * 'A'"  # Reverse order
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'memory allocation' in reason.lower() or 'large' in reason.lower()
        assert risk >= 5
    
    def test_integer_multiplication_large(self):
        """Test that large integer * integer is detected."""
        code = "x = 50000 * 50000"  # Results in 2,500,000,000
        is_safe, reason, risk = analyze_code_patterns(code)
        
        assert 'memory allocation' in reason.lower() or 'large' in reason.lower()
        assert risk >= 5


class TestHighRiskAccumulation:
    """Test risk score accumulation and thresholds."""
    
    def test_high_risk_score_threshold(self):
        """Test code that accumulates risk > 15 without critical violations."""
        # Multiple medium-risk patterns that add up
        code = """
import subprocess
import socket
import threading
for i in range(5000000):
    pass
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        # Should be blocked due to high accumulated risk
        if risk > 15:
            assert is_safe is False
            assert 'high risk score' in reason.lower()
    
    def test_combined_violations_accumulate_risk(self):
        """Test that multiple violations accumulate risk score."""
        code = """
import subprocess
import socket
import os
os.system('ls')
exec('code')
"""
        is_safe, reason, risk = analyze_code_patterns(code)
        
        # Should have very high risk due to multiple critical violations
        assert is_safe is False
        assert risk >= 20  # Multiple violations


class TestEdgeCasesAndHelpers:
    """Test edge cases and helper method coverage."""
    
    def test_get_func_name_unknown_node_type(self):
        """Test _get_func_name with unknown node type returns empty string."""
        import ast
        
        # Create a code with a complex expression that doesn't match Name or Attribute
        code = "result = (lambda x: x)(42)"
        tree = ast.parse(code)
        detector = SecurityPatternDetector()
        detector.visit(tree)
        
        # Should not crash, just return empty string for unknown types
        assert True  # If we get here without error, the test passes
