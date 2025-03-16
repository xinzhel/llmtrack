from .openai_model import OpenAILLM
from .config import get_root_dir, set_root_dir  

def get_llm(model_name:str, **kwargs):
    # if start with openai,  azure_openai, moonshot 
    if model_name.startswith("openai") or model_name.startswith("azure_openai") or model_name.startswith("moonshot") or model_name.startswith("groq"):
        return OpenAILLM(model_name, **kwargs)
    else:
        raise ValueError(f"Model {model_name} is not supported")
    
def list_providers():
    return ["openai", "azure_openai", "moonshot", "groq"]