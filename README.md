<p align="center">
  <img src="assets/logo.webp" alt="logo" width="200"/>
</p>


**LLMTrack** is a Python package designed to streamline the usage of language models, especially during batch generation. It offers features for easy loading of models, generation caching to optimize performance, detailed logging, and continuous token usage recording on a per-model basis. Whether you're working on research or deploying models in production, EfficientLLM helps you manage and monitor your models efficiently.

## Installation
```
pip install llmtrack
```
## Supported LLMs
Public LLM APIs are specified by simply specifying `model_name` consisting of API providers and model names. The supported APIs include :  
* OpenAI, e.g., "openai/xxxx"  (xxxx should be replaced by specific model names)
    * The environment variable has to be setup: `OPENAI_API_KEY` 
    * Specific model names: See [the document](https://platform.openai.com/docs/models) 
* Azure OpenAI, e.g., "azure_openai/chatgpt-4k" 
    * The three environment variables have to be setup: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`
    * Ask providers for specific model names 
* MoonShot, e.g., "moonshot/moonshot-v1-8k" 
    *  The environment variable has to be setup: `MOONSHOT_API_KEY`
    
