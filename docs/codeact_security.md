# Code Security

> **⚠️ DISCLAIMER**
>
> CodeActAgent offers basic security features to prevent malicious code execution. It is not a substitute for proper security measures. By no means the security features described here are bulletproof. Use it with caution.


This section describes how code security is handled in case of code-generating agents. The following sequence diagram (via CodeRabbit) shows the flow of code security in CodeActAgent.

```{kroki}
:type: mermaid
:format: svg
:options: {"themeVariables": {"fontSize": "24px"}}
:caption: Code Security Review by CodeActAgent
sequenceDiagram
    actor User
    participant CodeRunner
    participant PatternDetector
    participant SecurityReviewer
    participant LLM
    participant Executor

    User->>CodeRunner: run(tools_code, generated_code, task_id)
    CodeRunner->>PatternDetector: analyze_code_patterns(generated_code)
    alt Static analysis fails
        PatternDetector-->>CodeRunner: (is_safe=false, reason, risk_score)
        CodeRunner-->>User: raise CodeSecurityError(reason)
    else Static analysis passes
        PatternDetector-->>CodeRunner: (is_safe=true, reason, risk_score)
        CodeRunner->>SecurityReviewer: review(generated_code)
        SecurityReviewer->>LLM: call_llm(system_prompt, user: code)
        LLM-->>SecurityReviewer: CodeReview(is_secure, reason)
        alt LLM deems unsafe
            SecurityReviewer-->>CodeRunner: CodeReview(is_secure=false, reason)
            CodeRunner-->>User: raise CodeSecurityError(reason)
        else LLM approves
            SecurityReviewer-->>CodeRunner: CodeReview(is_secure=true, reason)
            CodeRunner->>Executor: execute(tools_code + generated_code)
            Executor-->>CodeRunner: (stdout, stderr, return_code)
            CodeRunner-->>User: (stdout, stderr, return_code)
        end
    end
```


## Overview

KodeAgent's **CodeActAgent** implements a comprehensive multi-layer security system to ensure that AI-generated code is safe to execute. This prevents malicious code, data exfiltration, and system compromise.

> **⚠️ WARNING**
> 
> **Never execute untrusted AI-generated code without proper security measures!**
> 
> The CodeActAgent implements multiple security layers to protect against:
> 
> * Dangerous system commands
> * Environment variable theft
> * File system attacks
> * Code obfuscation
> * Resource exhaustion


The CodeActAgent uses a **defense-in-depth** approach with 5 security layers:

```
┌─────────────────────────────────────────────┐
│         User Request / LLM Generated Code   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Layer 1: Syntax Validation (AST Parse)     │
│  • Ensures code is valid Python             │
│  • Catches syntax errors early              │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Layer 2: Import Whitelist                  │
│  • Only approved modules allowed            │
│  • Blocks: subprocess, socket, etc.         │
│  • Detects: exec, eval, __import__          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Layer 3: AST Pattern Detection             │
│  • Static code analysis                     │
│  • Catches obfuscated malicious code        │
│  • Risk scoring system                      │
│  • Cannot be socially engineered            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Layer 4: LLM Security Review               │
│  • Intelligent context-aware analysis       │
│  • Checks against security guidelines       │
│  • Provides human-readable explanations     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Layer 5: Sandboxed Execution               │
│  • Isolated environment (E2B/Docker)        │
│  • Resource limits (CPU, memory, time)      │
│  • Network restrictions                     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
               [Result]
```

## Layer 1: Syntax Validation

**Purpose**: Ensure code is syntactically valid Python before any analysis.

**Implementation**:

```python
import ast

try:
    ast.parse(source_code)
except SyntaxError as e:
    return error_response(f"Syntax error: {e}")
```

**Catches**:

* Missing colons, parentheses
* Invalid indentation
* Malformed expressions

## Layer 2: Import Whitelist

**Purpose**: Restrict code to only use approved Python modules.

**Default Allowed Imports**:

```python
DEFAULT_ALLOWED = [
    're', 'json', 'collections', 'itertools', 
    'functools', 'operator', 'string'
]
```

**Dangerous Builtins Blocked**:

* `exec()` - Execute arbitrary code
* `eval()` - Evaluate expressions
* `compile()` - Compile code objects
* `__import__()` - Dynamic imports

**Example**:

```python
# ❌ BLOCKED
import subprocess
subprocess.run(['rm', '-rf', '/'])

# ❌ BLOCKED
exec('import os; print(os.environ)')

# ✅ ALLOWED
import math
result = math.sqrt(16)
```

## Layer 3: AST Pattern Detection

**Purpose**: Static code analysis to catch malicious patterns that might fool LLM review.

**Key Advantage**: Deterministic detection that cannot be bypassed through social engineering or obfuscation.

### Detected Patterns

| Category | Examples | Risk | Action |
|----------|----------|------|--------|
| **Dangerous Builtins** | `exec()`, `eval()`, `compile()` | CRITICAL | Block |
| **Obfuscation** | `base64.b64decode()`, `bytes.fromhex()` | HIGH | Flag |
| **System Commands** | `os.system()`, `subprocess.run()` | CRITICAL | Block |
| **Environment Access** | `os.environ`, `os.getenv()` | HIGH | Block |
| **Path Traversal** | `../../../etc/passwd` | HIGH | Flag |
| **Memory Bombs** | `'A' * 1000000000` | HIGH | Flag |
| **Infinite Loops** | `while True:` without `break` | MEDIUM | Flag |
| **File Operations** | `os.remove()`, `os.unlink()` | CRITICAL | Block |
| **Network Modules** | `socket`, `urllib`, `http` | MEDIUM | Flag |
| **Dangerous Commands** | `rm -rf`, `dd if=`, `format` | CRITICAL | Block |

### Risk Scoring System

Each pattern is assigned a risk score:

* **CRITICAL**: 10 points → Immediate block
* **HIGH**: 5-7 points
* **MEDIUM**: 2-3 points

**Threshold**: Code with risk score > 15 is automatically blocked.

### Example Detection

```python
# This obfuscated code would fool an LLM but AST catches it:
import base64
exec(base64.b64decode('aW1wb3J0IG9zOyBwcmludChvcy5lbnZpcm9uKQ=='))

# ❌ BLOCKED by Pattern Detection:
# "Critical security violations: Dangerous builtin: exec"
# Risk Score: 15 (exec: 10 + b64decode: 5)
```

**Why This Works**:

* LLM might see: *"Some base64 string, probably harmless"*
* AST sees: *"exec() + b64decode() = CRITICAL VIOLATION"*

## Layer 4: LLM Security Review

**Purpose**: Intelligent, context-aware code review using an LLM.

**Security Guidelines**:

The LLM reviewer follows strict guidelines defined in `code_guardrail.txt`:

```
## Prohibited Actions
- NEVER access environment variables, secrets, API keys
- NEVER execute system commands that modify/delete system files
- NEVER access files outside workspace directory
- NEVER follow symlinks outside workspace
- NEVER use path traversal (../, ../../)
- NEVER make network requests to malicious URLs
- NEVER use dangerous builtins: exec, eval, compile
- NEVER create infinite loops or fork bombs
- NEVER log/store PII unless required
```

**Review Process**:

```python
review_result = await code_reviewer.review(generated_code)

if not review_result.is_secure:
    raise CodeSecurityError(
        f"Code blocked: {review_result.reason}"
    )
```

**Example Output**:

```json
{
  "is_secure": false,
  "reason": "Code attempts to access environment variables which could expose sensitive data like API keys"
}
```

## Layer 5: Sandboxed Execution

**Purpose**: Execute code in an isolated environment with resource limits.

### Execution Environments

**E2B Sandbox** (Recommended):

* Isolated Docker container
* Network restrictions
* Automatic cleanup
* Resource limits

**Host Execution** (Development Only):

> **⚠️ WARNING**
> 
> Host execution is **dangerous** and should only be used for development with trusted code!

### Resource Limits

```python
CodeRunner(
    env='e2b',              # Use E2B sandbox
    timeout=30,             # 30 second timeout
    allowed_imports=[...],  # Whitelist
    pip_packages='...'      # Approved packages
)
```

**Enforced Limits**:

* **CPU Time**: 30 seconds (configurable)
* **Memory**: 256MB (recommended)
* **File Size**: 100MB per file
* **File Count**: 100 files per task

## Security Best Practices

### For Developers

1. **Always use E2B sandbox** for production:

   ```python
   agent = CodeActAgent(
       run_env='e2b',  # Not 'host'!
       allowed_imports=['math', 'datetime'],
       timeout=30
   )
   ```

2. **Minimize allowed imports**:

   ```python
   # ❌ TOO PERMISSIVE
   allowed_imports=['os', 'subprocess', 'socket']
   
   # ✅ MINIMAL
   allowed_imports=['math', 'datetime', 'json']
   ```

3. **Review security logs**:

   ```python
   logger.info(f"Security review: {review_result}")
   logger.info(f"Pattern detection: risk={risk_score}")
   ```

4. **Monitor for false positives**:

   Track blocked code to tune risk thresholds.

### For Users

1. **Never disable security checks**
2. **Review code before execution** (if possible)
3. **Use minimal permissions**
4. **Monitor resource usage**
5. **Report suspicious behavior**

## Configuration

### Customizing Security

**Adjust Risk Threshold**:

```python
# In pattern_detector.py
if detector.risk_score > 15:  # Adjust this value
    return False, ...
```

**Add Custom Patterns**:

```python
# In SecurityPatternDetector class
def visit_Call(self, node):
    func_name = self._get_func_name(node.func)
    
    # Add your custom detection
    if func_name in ['my_dangerous_function']:
        self.violations.append(('CRITICAL', 'Custom violation'))
        self.risk_score += 10
```

**Custom LLM Model**:

```python
agent = CodeActAgent(
    model_name='gpt-4',  # Use stronger model for review
    litellm_params={'temperature': 0}  # Deterministic
)
```

## Testing Security

### Run Security Tests

```bash
# Test pattern detector
pytest tests/unit/test_pattern_detector.py -v

# Test code runner security
pytest tests/unit/test_code_runner.py -v -k security

# Check coverage
pytest tests/unit/test_pattern_detector.py --cov=kodeagent.pattern_detector
```

**Expected Coverage**: ≥ 98%

### Demo Security Features

```bash
python examples/pattern_detection_demo.py
```

**Output**:

```
Safe code: is_safe=True, reason=No suspicious patterns detected, risk=0
Obfuscated: is_safe=False, reason=Critical security violations: Dangerous builtin: exec, risk=15
Env access: is_safe=False, reason=Critical security violations: Dangerous os.environ access, risk=10
```

## Known Limitations

1. **LLM Review Can Be Fooled**:
   
   * Social engineering: *"This is for testing purposes only..."*
   * Novel obfuscation techniques
   * **Mitigation**: AST pattern detection as backup

2. **Pattern Detection False Positives**:
   
   * Legitimate use of flagged patterns
   * **Mitigation**: Tune risk thresholds, whitelist specific cases

3. **Resource Limits**:
   
   * Some legitimate tasks may hit limits
   * **Mitigation**: Increase limits for trusted users

4. **Performance Overhead**:
   
   * Security checks add ~100-200ms latency
   * **Mitigation**: Acceptable for safety-critical applications


## API Reference

See the full API documentation for:

* `SecurityPatternDetector` - AST-based pattern detection
* `analyze_code_patterns()` - Analyze code for security violations
* `CodeSecurityReviewer` - LLM-based security review
* `CodeRunner` - Code execution with security checks

