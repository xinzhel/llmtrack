from typing import Generic, TypeVar, Union, NamedTuple, Protocol, Optional, runtime_checkable, Tuple
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any
import json
import os
import diskcache as dc
from .logging_util import setup_logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import time

class TokenUsageRecord:
    def __init__(self, model_name, file_path=None):
        self.file_path = model_name+'_token_usage.json' if not file_path else file_path 

    def update_usage(self, prompt_tokens, completion_tokens, total_tokens):
        episode_usage = {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens, "time": time.time()}
        
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
    def __init__(self, model_name, log: bool = False, cache: bool = False, token_usage:bool = True):
        if '/' not in model_name:
            raise ValueError("model_name should be in the format of <api_provider>/<model_name>")
        self.api_provider, self.model_name = model_name.split('/')

        # for cache textual response
        self.cache = None
        if cache:
            cache_name = self.__class__.__name__ + '_' + self.model_name
            self.cache_path = os.path.join('cahe_llm_agent', f"{cache_name}.db")
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
            self.token_usage = TokenUsageRecord(model_name=self.model_name)
            
    def check_usage(self):
        if self.token_usage:
            return self.token_usage.check_usage()
        else:
            raise ValueError("Token usage is not enabled.")

    @abstractmethod
    def generate(self,
                 usr_msg: str,
                 system_msg: str = '', 
                 history: Optional[List[str]] = None, 
                 temperature: Optional[float] = None, 
                 max_tokens: Optional[int] = None, 
                 top_p: Optional[float] = None, 
                 num_return_sequences: int = 1,
                 stop: Optional[str] = None,
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

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5)) # wait time: 1, 2, 4, 8, 16 with exponential factor  2
    def __call__(self,
                usr_msg: str,
                system_msg: str = '', 
                history: Optional[List[str]] = None, 
                temperature: Optional[float] = None, 
                max_tokens: Optional[int] = None, 
                top_p: Optional[float] = None, 
                num_return_sequences: int = 1,
                stop: Optional[str] = None,
                **kwargs: Any) -> str:
        # cache
        if self.cache and system_msg + '\n' + usr_msg in self.cache:
            response_txt = self.cache[system_msg + '\n' + usr_msg]
        else:
            llm_output = self.generate(usr_msg, system_msg, history, temperature, max_tokens, top_p, num_return_sequences, stop, **kwargs)
            response_txt = llm_output.text[0]
            self.cache[system_msg + '\n' + usr_msg] = response_txt

        # log
        if self.log:
            self.logger.info(self.info_begin, "Prompt:\n%s", system_msg + '\n' + usr_msg, self.info_end)
            self.logger.info(self.info_begin, "Output:\n%s", response_txt, self.info_end)
            
        return response_txt.strip()

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