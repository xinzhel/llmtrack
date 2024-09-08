<p align="center">
  <img src="assets/logo.webp" alt="logo" width="200"/>
</p>


**LLMTrack** is a Python package designed to streamline the usage of language models, especially during batch generation. It offers features for easy loading of models, generation caching to optimize performance, detailed logging, and continuous token usage recording on a per-model basis. Whether you're working on research or deploying models in production, EfficientLLM helps you manage and monitor your models efficiently.

## Installation
```
pip install llmtrack
```
## LLM Loading
```python
from llmtrack import get_llm
llm = get_llm(model_name="openai/gpt-4o-mini")
print(llm.generate("Generate ONLY a random word"))
```

Public LLM APIs are specified by simply specifying `model_name` consisting of API providers and model names. The supported APIs include :  
* OpenAI, e.g., "openai/xxxx"  (xxxx should be replaced by specific model names)
    * The environment variable has to be setup: `OPENAI_API_KEY` 
    * Specific model names: See [the document](https://platform.openai.com/docs/models) 
* Azure OpenAI, e.g., "azure_openai/chatgpt-4k" 
    * The three environment variables have to be setup: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`
    * Ask providers for specific model names 
* MoonShot, e.g., "moonshot/moonshot-v1-8k" 
    *  The environment variable has to be setup: `MOONSHOT_API_KEY`

## Unified Parameters
| Parameter              | Description                                                                                 |
|------------------------|---------------------------------------------------------------------------------------------|
| `num_return_sequences`  | Number of sequences to return, defaults to 1. Same as `n` in OpenAI API                     |
| `temperature`           | More random if < 1.0; more deterministic if > 1.0                                           |
| `max_tokens`            | Maximum number of tokens to generate                                                        |
| `top_p`                 | Top p for sampling, refer to the paper: [https://arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751) |
| `stop`                  | Stop sequence for generation                                                                |

An example:
```python
params = {"temperature": 0.2, "num_return_sequences": 1}
print(llm.generate("Generate ONLY a random word", **params))
```

## Caching
```python
from llmtrack import get_llm
llm = get_llm(model_name="openai/gpt-4o-mini", cache=True)
print(llm.generate("Generate ONLY a random word "))
```
After running the code above, the generation cache will be stored in `cahe_llmtrack/openai/gpt-4o-mini`, following the naming rule `cahe_llmtrack/{API provider}/{model name}`.

If you invoke the same model with the same prompt, the cache will be used. 
> Note: You can verify this by checking whether token usage increases with the next function: Token Usage Tracking.

## Token Usage Tracking
```python
from llmtrack import get_llm
llm = get_llm("openai/gpt-4o-mini", cache=True, token_usage=True)
print(llm.generate("Generate ONLY a random word "))
print(llm.generate("Generate ONLY a random word "))
```
Let's track the token usage at the `./gpt-4o-mini_token_usage.json`. Only one record exists, although we invoke the LLM twice.
```json
{"prompt": 17, "completion": 4, "total": 21, "time": "2024-08-25 14:02:36"}
```

## Logging (TBA)
    
