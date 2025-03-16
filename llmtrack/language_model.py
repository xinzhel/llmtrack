from tabnanny import verbose
from typing import Union, NamedTuple, Optional
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any
import json
import os
import traceback
import diskcache as dc
from .logging_util import setup_logger
import time
from dataclasses import dataclass
import warnings
from .config import get_root_dir

# define a data type: ClientResponse, which can be an arbitrary type of response from the LLM API (e.g., openAI, Azure OpenAI, MoonShot, Groq)
ClientResponse = Any
Message = dict[str, str]

class TokenUsageRecord:
    def __init__(self, file_path):
        self.file_path = file_path 

    def update_usage(self, prompt_tokens, completion_tokens, total_tokens):
        episode_usage = {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
        
        # insert a line to the file
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(episode_usage) + '\n')
    
    def check_usage(self):
        # read the file
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            
        accumulated_usage = {"prompt": 0, "completion": 0, "total": 0}
        for line in lines:
            usage = json.loads(line)
            for key in accumulated_usage.keys():
                accumulated_usage[key] += usage[key]
        return accumulated_usage
    
    def __str__(self):
        return str(self.check_usage())

@dataclass
class GenerateOutput:
    raw: Any = None  # any type of output from different LLM API
    token_usage: tuple = None
    text: list[str] = None
    log_prob: list[np.ndarray] = None
    tool_calls: list[dict] = None

class LanguageModel(ABC):
    def __init__(self, model_name, log: bool = False, cache: bool = False, token_usage:bool = False, token_usage_file_path=None, verbose: bool = False, **kwargs):
        if '/' not in model_name:
            raise ValueError("model_name should be in the format of <api_provider>/<model_name>")
        names = model_name.split('/')
        if len(names) == 2:
            self.api_provider, self.model_name = names[0], names[1]
        elif len(names) == 1:
            self.api_provider, self.model_name = 'default', names[1]
        else:
            raise ValueError('Incorrect Format for `model_name`')

        # for cache textual response
        self.cache = None
        if cache:
            self.cache_path = os.path.join(get_root_dir(), model_name, f"cache.db")
            self.cache = dc.Cache(self.cache_path)

        # for logging
        self.logger = None
        if log:
            self.info_begin = "===== Begin =====\n"
            self.info_end = "===== End ====\n\n"
            log_name = self.__class__.__name__  
            self.logger = setup_logger(name=log_name)
        
        # for token usage
        self.token_usage = None
        if token_usage and self._check_token_usage_func():
            self.token_usage_path = os.path.join(get_root_dir(), model_name, 'token_usage.json')
            self.token_usage = TokenUsageRecord(file_path=self.token_usage_path)
        
        if verbose:
            print("The path of token usage is:", self.token_usage_path)
            print("The path of cache is:", self.cache_path)

            
        # pre-define config
        self.config = {
            "temperature": kwargs.pop("temperature", 1), # < 1.0: more random
            "max_tokens": kwargs.pop("max_tokens", 2048),
            "top_p": kwargs.pop("top_p", 0.99),
            "stop": kwargs.pop("stop", None),
            "num_return_sequences": kwargs.pop("num_return_sequences", 1),
        }
        if kwargs:
            raise ValueError(f"Arguments for LLM config are not supported: {kwargs}")
    
    def clear_cache(self):
        if self.cache:
            self.cache.clear()
        else:
            raise ValueError("Cache is not enabled.")
            
    def _check_token_usage_func(self):
        func_defined = "Defined" if self._extract_token_usage.__code__.co_code != (lambda: None).__code__.co_code else "Undefined"
        if func_defined == "Undefined":
            raise ValueError("Token usage tracking is not supported for the LLM client, since the `_extract_token_usage` method is not implemented.")
        return True

    def check_usage(self):
        if self.token_usage and self._check_token_usage_func():
            return self.token_usage.check_usage()
        else:
            raise ValueError("Token usage is not enabled.")

    @abstractmethod
    def _respond(self,
                 usr_msg: str,
                 system_msg: str = '', 
                 history: Optional[List[str]] = None, 
                 **kwargs: Any) -> GenerateOutput:
        """Generate text from a list of prompts.

        :param usr_msg: User message input.
        :param system_msg: System message input, defaults to an empty string.
        :param history: List of previous messages for context, defaults to None.
        :param temperature: Temperature for sampling, defaults to None.
        :param max_tokens: Maximum number of tokens to generate, defaults to None.
        :param top_p: Top-p for sampling, defaults to None.
        :param num_return_sequences: Number of sequences to return, defaults to 1.
        :param stop: Stop sequence for generation, defaults to None.
        :param kwargs: Additional keyword arguments.
        :return: GenerateOutput object with the generated text.
        """
        pass

    @abstractmethod
    def _generate_messages(self, usr_msg: str, system_msg: str = '', history: Optional[List[str]] = None) -> list[Message]:
        pass    

    def get_cache_key(self, usr_msg: str, system_msg: str = '', history: Optional[List[str]] = None) -> str:
        return str(self._generate_messages(usr_msg, system_msg, history))

    def respond(self,
                usr_msg: str,
                system_msg: str = '', 
                history: Optional[List[str]] = None, 
                max_invocation = 5,
                verbal=False,
                **kwargs: Any) -> ClientResponse: 

        # check data types
        if not isinstance(usr_msg, str):
            raise ValueError("usr_msg must be a string")

        if not isinstance(system_msg, str):
            raise ValueError("system_msg must be a string")

        if history is not None and not isinstance(history, list):
            raise ValueError("history must be a list")
        
        # cache
        cache_key = self.get_cache_key(usr_msg, system_msg, history)
        if verbal:
            print("Cache key:", cache_key)
        if self.cache and cache_key in self.cache:
            if verbal:
                print("Cache hit!")
            client_response = self.cache[cache_key]
            
        else:
            # call _generate; if fail, retry 5 times after wait time: 1, 2, 4, 8, 16 with exponential factor  2
            num_invocation = 0
            while True:
                try:
                    num_invocation += 1
                    client_response = self._respond(usr_msg, system_msg, history=history, **kwargs)
                    break
                except Exception as e:
                    if num_invocation >= max_invocation:
                        raise e
                    else:
                        print(traceback.format_tb(e.__traceback__))
                        print(f"\n\nRetry {num_invocation} times.")
                        time.sleep(2**num_invocation)

            # cache client_response
            self.cache[cache_key] = client_response

            # token usage if _extract_token_usage is implemented
            if self.token_usage:
                token_usage = self._extract_token_usage(client_response)
                if token_usage[2] != token_usage[0] + token_usage[1]:
                    print(f"Token usage not consistent: {token_usage}")
                self.token_usage.update_usage(prompt_tokens=token_usage[0], completion_tokens=token_usage[1], total_tokens=token_usage[2])

        # log
        if self.logger:
            self.logger.info(self.info_begin, "System Msg:\n%s", system_msg + '\n' + "User Msg:\n" + usr_msg, self.info_end)
            self.logger.info(self.info_begin, "Output:\n%s", self._extract_text(client_response), self.info_end)
        
        return client_response
    
    def respond_txt(self, usr_msg: str,
                system_msg: str = '', 
                history: Optional[List[str]] = None, 
                **kwargs: Any) -> str: 
        client_response = self.respond(usr_msg, system_msg, history=history, **kwargs)
        response_txt = self._extract_text(client_response)
        response_txt = [txt.strip() for txt in response_txt]
        if len(response_txt)>1:
            warnings.warn(f"{len(response_txt)} responses are returned, only the first one is returned.")
        return response_txt[0]

    @abstractmethod
    def _extract_text(self, client_response):
        pass

    def _extract_token_usage(self, client_response):
        pass

    def _extract_log_prob(self, client_response):
        return None

    def _extract_tool_calls(self, client_response):
        return None

    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              postprocess: Optional[str] = None,
                              **kwargs) -> list[np.ndarray]:
        """ TODO: doc

        :param prompt:
        :param candidates:
        :param postprocess: optional, can be 'log_softmax' or 'softmax'. Apply the corresponding function to logits before returning
        :return:
        """
        pass

    def _extract_generate_output(self, client_response):
        text = self._extract_text(client_response)
        token_usage = self._extract_token_usage(client_response)
        log_prob = self._extract_log_prob(client_response)
        tool_calls = self._extract_tool_calls(client_response)

        return GenerateOutput(
            token_usage=token_usage,
            raw = client_response,
            text=text,
            log_prob=log_prob,
            tool_calls=tool_calls
            )

    @abstractmethod
    def get_loglikelihood(self,
                          prefix: str,
                          contents: list[str],
                          **kwargs) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        pass