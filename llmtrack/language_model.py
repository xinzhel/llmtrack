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

class TokenUsageRecord:
    def __init__(self, model_name, file_path=None):
        self.file_path = model_name+'_token_usage.json' if not file_path else file_path 

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
            
class GenerateOutput(NamedTuple):
    text: list[str]
    log_prob: list[np.ndarray] = None

class LanguageModel(ABC):
    def __init__(self, model_name, log: bool = False, cache: bool = False, token_usage:bool = False, token_usage_file_path=None, **kwargs):
        if '/' not in model_name:
            raise ValueError("model_name should be in the format of <api_provider>/<model_name>")
        names = model_name.split('/')
        if len(names) == 2:
            self.api_provider, self.model_name = names[0], names[1]
        elif len(names) == 1:
            self.api_provider, self.model_name = 'default', names[1]
        else:
            raise 'Incorrect Format for `model_name`'

        # for cache textual response
        self.cache = None
        if cache:
            self.cache_path = os.path.join('cache_llmtrack', f"{model_name}.db")
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
        if token_usage:
            self.token_usage = TokenUsageRecord(model_name=self.model_name, file_path=token_usage_file_path)
            
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
            
    def check_usage(self):
        if self.token_usage:
            return self.token_usage.check_usage()
        else:
            raise ValueError("Token usage is not enabled.")

    @abstractmethod
    def _generate(self,
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

    def generate(self,
                usr_msg: str,
                system_msg: str = '', 
                history: Optional[List[str]] = None, 
                max_invocation = 5,
                verbal=False,
                **kwargs: Any) -> Union[str, list[str]]: # list if num_return_sequences > 1
  
        # cache
        if self.cache and system_msg + '\n' + usr_msg in self.cache:
            if verbal:
                print("Cache hit!")
            response_txt = self.cache[system_msg + '\n' + usr_msg]
        else:
            # call _generate; if fail, retry 5 times after wait time: 1, 2, 4, 8, 16 with exponential factor  2
            num_invocation = 0
            while True:
                try:
                    num_invocation += 1
                    llm_output = self._generate(usr_msg, system_msg, history=history, **kwargs)
                    break
                except Exception as e:
                    if num_invocation >= max_invocation:
                        raise e
                    else:
                        print(traceback.format_tb(e.__traceback__))
                        print(f"\n\nRetry {num_invocation} times.")
                        time.sleep(2**num_invocation)
            
            response_txt = llm_output.text[0] if kwargs.get("num_return_sequences") == 1 else llm_output.text
            self.cache[system_msg + '\n' + usr_msg] = response_txt
            if verbal:
                print("Cache key:", system_msg + '\n' + usr_msg)

        # log
        if self.logger:
            self.logger.info(self.info_begin, "Prompt:\n%s", system_msg + '\n' + usr_msg, self.info_end)
            self.logger.info(self.info_begin, "Output:\n%s", response_txt, self.info_end)
            
        return response_txt.strip() if kwargs.get("num_return_sequences") == 1 else [txt.strip() for txt in response_txt]

    @abstractmethod
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