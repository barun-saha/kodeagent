# KodeAgent with Ollama

This directory contains examples of running KodeAgent offline with Small Language Models (SLMs) via [Ollama](https://ollama.com/).
You should have Ollama installed and the desired model pulled locally to run the code in this directory.

`local_offline_agent.py`: demonstrates the use of KodeAgent with LFM 2.5 8B-A1B, a Mixture of Experts (MoE) model.
The agent uses web search-related tools to solve this particular task:
```text
Find the current stock price of NVIDIA & calculate how many shares I can buy with $1000.
```

You can also run other tasks with optional file inputs, e.g.,
```python
('What is this page about?', ['https://en.wikipedia.org/wiki/Artificial_intelligence']),
```

or
```python
(
    'Get the transcript of this YouTube video: https://www.youtube.com/watch?v=aircAruvnKk'
    '\nIdentify the main topic, then search Wikipedia for that topic and give me'
    ' a brief summary of what Wikipedia says about it (give Wikipedia page link).',
    None,
),
```

For more examples, refer to this Colab notebook: https://colab.research.google.com/drive/1c7RMTCcSYrO7wZgB25bLX9QenDgVDmAP?usp=sharing

Note: Running even small models in an agentic loop can be slow on the CPU. For better performance, consider using a GPU or a more powerful machine.
