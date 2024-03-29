# 🛡️Guardrail ML
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI - Python Version](https://img.shields.io/pypi/v/llm-guard)](https://pypi.org/project/guardrail-ml)
[![Downloads](https://static.pepy.tech/badge/guardrail-ml)](https://pepy.tech/project/guardrail-ml)

![plot](./static/images/guardrail_v5.png)

Guardrail ML is an alignment toolkit to use LLMs safely and securely. Our firewall scans prompts and LLM behaviors for risks to bring your AI app from prototype to production with confidence.

## Benefits
- 🚀mitigate LLM security and safety risks 
- 📝customize and ensure LLM behaviors are safe and secure
- 💸monitor incidents, costs, and responsible AI metrics 

## Features 
- 🛠️ firewall that safeguards against CVEs and improves with each attack
- 🤖 reduce and measure ungrounded additions (hallucinations) with tools
- 🛡️ multi-layered defense with heuristic detectors, LLM-based, vector DB

## Quickstart 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eUm6tkEl9YvzgObwWDvt6pOnWgnReIug?usp=sharing)

## Installation 💻
1. [Guardrail API Key](app.useguardrail.com) and set `env` variable as `GUARDRAIL_API_KEY` 

2. To install guardrail, use the Python Package Index (PyPI) as follows:
```
pip install guardrail-ml
```

![plot](./static/images/quickstart.png)

## Roadmap

**Firewall**
- [x] Prompt Injections
- [x] Factual Consistency
- [x] Factuality Tool
- [x] Toxicity Detector
- [x] Regex Detector 
- [x] Stop Patterns Detector 
- [x] Malware URL Detector 
- [x] PII Anonymize
- [x] Secrets
- [x] DoS Tokens
- [x] Harmful Detector 
- [x] Relevance
- [x] Contradictions
- [x] Text Quality
- [x] Language
- [x] Bias
- [ ] Adversarial Prompt Generation
- [ ] Attack Signature

**Integrations**
- [x] OpenAI Completion
- [x] LangChain
- [ ] LlamaIndex
- [ ] Cohere
- [ ] HuggingFace

## More Colab Notebooks

Old Quickstart v0.0.1 (08/03/23) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KCn1HIeD3fQy8ecT74yHa3xgJZvdNvqL?usp=sharing)

4-bit QLoRA of `llama-v2-7b` with `dolly-15k` (07/21/23): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/134o_cXcMe_lsvl15ZE_4Y75Kstepsntu?usp=sharing)

Fine-Tuning Dolly 2.0 with LoRA: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n5U13L0Bzhs32QO_bls5jwuZR62GPSwE?usp=sharing)

Inferencing Dolly 2.0: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A8Prplbjr16hy9eGfWd3-r34FOuccB2c?usp=sharing)
