<p align="center">
  <img src="assets/logo.webp" alt="logo" width="200"/>
</p>


**LLMTrack** is a Python package for enabling caching (avoiding repeated API calls) and recording token usage on a per-model basis. 

> Why do we call it "per-model"? Because we want to track the token usage and cache for each model separately under the root directory, following the rule `{root_dir}/{client name}/{model name}`.

## Installation
```
pip install llmtrack
```

## Root Directory for Saving Cache and Token Usage
By default, the root directory for saving cache and token usage is the current working directory (`os.getcwd()`). You can change it, as follows:
```python
from llmtrack import set_root_dir, get_root_dir
set_root_dir("~/my_project/llmtrack")
print(get_root_dir())
```

## Caching and Recording Token Usage
You can use `get_llm` to get a language model instance, as follows:

```python
from llmtrack import get_llm
client_name = "openai"
model_name = "gpt-4o-mini"
llm = get_llm(f"{client_name}/{model_name}", cache=True, token_usage=True)
usr_message = "ONLY generate a positve word"
client_response = llm.respond(usr_message, verbal=True)
```

After running the code above, the cache and token-usage files will be stored in `~/my_project/llmtrack/openai/gpt-4o-mini`, following the rule `{root_dir}/{client name}/{model name}`. Now, if you invoke the same model with the same prompt, the cache will be used. 

You can check the token usage and cache by:
```python
# check token usage
print('Token Usage')
usage = llm.token_usage
print(usage)

# check cache
print('\n\nCache')
cache_key = llm.get_cache_key(usr_message)
print(llm.cache[cache_key])
```

## Supported Clients and Model Names
Public LLM APIs are specified by simply specifying `model_name` consisting of API providers and model names. The supported APIs include :  
* OpenAI, e.g., "openai/xxxx"  (xxxx should be replaced by specific model names)
    * The environment variable has to be setup: `OPENAI_API_KEY` 
    <!-- * Popular `model_name`: `gpt-4o-mini`, `gpt-3.5-turbo` -->
    * All Available `model_name`: See [the document](https://platform.openai.com/docs/models) 
* Azure OpenAI, e.g., "azure_openai/chatgpt-4k" 
    * The three environment variables have to be setup: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`
    * Ask providers for specific model names 
* MoonShot, e.g., "moonshot/moonshot-v1-8k" 
    * The environment variable has to be setup: `MOONSHOT_API_KEY`
* Groq
    <!-- * Popular `model_name`: `llama3-8b-8192`, `llama3-70b-8192` -->
    * All Available `model_name`: See [the document](https://console.groq.com/docs/models)

<!-- ## Unified Parameters
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
print(llm.respond_txt("Generate ONLY a random word", **params))
``` -->

    
