import os
import openai
from openai import OpenAI, AzureOpenAI
import numpy as np
from typing import Optional, List, Any
from .language_model import LanguageModel

class OpenAILLM(LanguageModel):
    def __init__(self, model_name:str, tool_def: Optional[list] = None, **kwargs):
        super().__init__(model_name, cache= kwargs.pop("cache", False), log=kwargs.pop('log', False), token_usage=kwargs.pop("token_usage", False), **kwargs)
        self.client = self._get_client(model_name)
        self.tool_def = tool_def
        
        API_KEY = os.getenv("AZURE_OPENAI_API_KEY", None)
        if API_KEY is None:
            raise ValueError("OPENAI_API_KEY not set, please run `export OPENAI_API_KEY=<your key>` to ser it")
        else:
            openai.api_key = API_KEY

    def _get_client(self, model_name):
        if model_name.startswith("azure_openai"):
            return AzureOpenAI(  
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ['AZURE_OPENAI_API_KEY'],  
                api_version=os.environ['AZURE_OPENAI_API_VERSION'], 
            )
        elif model_name.startswith("openai"):
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif model_name.startwith("moonshot"):
            return OpenAI(
                api_key = "$MOONSHOT_API_KEY",
                base_url = "https://api.moonshot.cn/v1",
            )
        elif model_name.startwith("groq"):
            from groq import Groq
            return Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
        else:
            raise ValueError(f"Model {model_name} is not supported")
    
    def _generate_messages(self, usr_msg: str, system_msg: str = '', history: Optional[List[str]] = None):
        if not history:
            messages= [{"role": "system", "content": system_msg}, 
                      {"role": "user", "content": usr_msg} ]
        else:
            messages= [{"role": "system", "content": system_msg}]
            # loop every two messages in history
            for i in range(0, len(history), 2):
                messages.append({"role": "user", "content": history[i]})
                messages.append({"role": "assistant", "content": history[i+1]})
            messages.append({"role": "user", "content": usr_msg})
        return messages

    def _respond(self,
                usr_msg: str,
                system_msg: str = '', 
                history: Optional[List[str]] = None, 
                **kwargs: Any):

        messages = self._generate_messages(usr_msg, system_msg, history)
        client_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages ,
            temperature= self.config["temperature"] if kwargs.get("temperature") is None else kwargs.get("temperature"),
            max_tokens=self.config["max_tokens"] if kwargs.get("max_tokens") is None else kwargs.get("max_tokens"),
            top_p=self.config["top_p"] if kwargs.get("top_p") is None else kwargs.get("top_p"),
            frequency_penalty=0,
            presence_penalty=0,
            tools=self.tool_def,
            stop=self.config["stop"] if kwargs.get("stop") is None else kwargs.get("stop"),
            n=self.config["num_return_sequences"] if kwargs.get("num_return_sequences") is None else kwargs.get("num_return_sequences"),
        )
        
        return client_response

    def _extract_log_prob(self, client_response):
        return None

    def _extract_text(self, client_response):
        return [choice.message.content for choice in client_response.choices]
    
    def _extract_token_usage(self, client_response):
        return (client_response.usage.prompt_tokens, client_response.usage.completion_tokens, client_response.usage.total_tokens)

    def _extract_tool_calls(self, client_response):
        return [choice.message.tool_calls for choice in client_response.choices]

