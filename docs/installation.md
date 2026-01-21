# Installation

We recommend installing **KodeAgent** into a dedicated virtual environment.

## Stable Release

To install (or upgrade to) the latest stable version of KodeAgent, run this command:

```bash
pip install -U kodeagent
```

You can verify the installation by checking the version of KodeAgent:

```python
import kodeagent

print(kodeagent.__version__)
```

## Development Version

If you want to use the latest features or contribute, clone the repository and install it in editable mode:

```bash
git clone https://github.com/barun-saha/kodeagent.git
cd kodeagent
pip install -e .
```

## Optional Dependencies

### Tracing

While `langfuse` is included by default, `langsmith` must be installed separately:

```bash
pip install langsmith
```
