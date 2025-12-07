"""
Example of how using AST pattern detection with the existing security review.
"""
from kodeagent.pattern_detector import analyze_code_patterns

# Example 1: Safe code
safe_code = """
import math
result = math.sqrt(16)
print(result)
"""

is_safe, reason, risk_score = analyze_code_patterns(safe_code)
print(f"Safe code: is_safe={is_safe}, reason={reason}, risk={risk_score}")

# Example 2: Obfuscated code (would fool LLM but AST catches it)
obfuscated_code = """
import base64
exec(base64.b64decode('aW1wb3J0IG9zOyBwcmludChvcy5lbnZpcm9uKQ=='))
"""

is_safe, reason, risk_score = analyze_code_patterns(obfuscated_code)
print(f"Obfuscated: is_safe={is_safe}, reason={reason}, risk={risk_score}")

# Example 3: Environment variable access
env_code = """
import os
print(os.environ)
"""

is_safe, reason, risk_score = analyze_code_patterns(env_code)
print(f"Env access: is_safe={is_safe}, reason={reason}, risk={risk_score}")

# Example 4: Infinite loop
loop_code = """
while True:
    print("forever")
"""

is_safe, reason, risk_score = analyze_code_patterns(loop_code)
print(f"Infinite loop: is_safe={is_safe}, reason={reason}, risk={risk_score}")

# Example 5: Path traversal
traversal_code = """
with open('../../../etc/passwd', 'r') as f:
    print(f.read())
"""

is_safe, reason, risk_score = analyze_code_patterns(traversal_code)
print(f"Path traversal: is_safe={is_safe}, reason={reason}, risk={risk_score}")
