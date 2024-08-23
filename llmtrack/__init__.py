from .openai_model import AnyOpenAILLM

def get_llm(model_name:str, **kwargs):
    # if start with openai,  azure_openai, moonshot 
    if model_name.startswith("openai") or model_name.startswith("azure_openai") or model_name.startswith("moonshot"):
        return AnyOpenAILLM(model_name, **kwargs)
    else:
        raise ValueError(f"Model {model_name} is not supported")