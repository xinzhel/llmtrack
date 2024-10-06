from typing import Union, Optional
import warnings
import copy
import time

from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
import torch
import numpy as np

from llmtrack.language_model import LanguageModel,GenerateOutput

class HFModel(LanguageModel):
    def __init__(self, model_name, device='mps', max_batch_size=1, quantized=None, peft_pth=None, load_awq_pth=None):
        """
        Initializes a new instance of the `HFModel` class.

        Args:
            quantized (str, optional): The type of quantization to use for the model. Can be "8bit", "nf4", "fp4", or "awq". Defaults to None.
            peft_pth (str, optional): The path to the directory containing the pre-trained PEFT model. Defaults to None.
            load_awq_pth (str, optional): The path to the directory containing the pre-trained AWQ model. Defaults to None.
            
        TODO:
        * record token_usages
        * test get_loglikelihood, get_next_token_logits functions
        * implement stop (vs eos)
        """
        super().__init__(model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, lagacy=False, padding_side="left") # https://discuss.huggingface.co/t/llama3-so-much-slow-compared-to-ollama/97638/3
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        
        if quantized is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                # load_in_8bit=True, # deprecated, pass a `BitsAndBytesConfig` object in `quantization_config` argument
                torch_dtype=torch.bfloat16
            )
        else:
            raise NotImplementedError(f"Quantization type {quantized} is not supported.")
        
        self.max_batch_size = max_batch_size
        self.device = device
        self.model.eval()
        
    def _generate(
                self,
                usr_msg: str,
                system_msg: str = 'You are a helpful assistant', 
                output_log_probs: bool = False,
                verb_time:bool=False,
                **kwargs,
            ):
        self.model.eval()
        # generation config
        generation_config = dict(
            temperature=self.config["temperature"] if kwargs.get("temperature") is None else kwargs.get("temperature"),
            max_length=self.config["max_tokens"] if kwargs.get("max_tokens") is None else kwargs.get("max_tokens"),
            top_p=self.config["top_p"] if kwargs.get("top_p") is None else kwargs.get("top_p"),
            max_new_tokens =self.config["max_tokens"] if kwargs.get("max_tokens") is None else kwargs.get("max_tokens"),
            top_k=50,
            num_beams=1, # different with `num_return_sequences`?
            do_sample = False,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # eos_token_id=self._convert_stop_to_end_token_id(self.config["stop"] if kwargs.get("stop") is None else kwargs.get("stop")),
        )
        
        # N samples
        N = self.config["num_return_sequences"] if kwargs.get("num_return_sequences") is None else kwargs.get("num_return_sequences")
        chats =[
            [{"role": "system", "content": system_msg}, 
              {"role": "user", "content": usr_msg}],
        ]
        if N > 1:
            generation_config['do_sample'] = True
            chats = chats * N

        # batching + tokenization + inference + decoding
        decoded_list = []
        log_prob_list = []
        for start in range(0, len(chats), self.max_batch_size):
            # 1. batching
            end = min(start + self.max_batch_size, len(chats))
            batch = chats[start:end]
            
            # 2. tokenization
            if verb_time:
                t0_1=time.time()
            
            encoded_inputs=self.tokenizer.apply_chat_template(batch, return_tensors="pt" ,
                                        add_generation_prompt=True,
                                        padding=True,
                                        return_dict=True).to(self.device)
            # encoded_inputs = self.tokenizer(inputs[start:end], return_tensors='pt', padding=True).to(self.device)
            if verb_time:
                t0_2=time.time()
                print("Tokenization time:",t0_2 - t0_1)
    
           # 3. inference
            if verb_time:
                t1=time.time()

            generation_output = self.model.generate(
                encoded_inputs["input_ids"], 
                attention_mask=encoded_inputs["attention_mask"],
                output_scores= output_log_probs,
                return_dict_in_generate=True,
                **generation_config,
            )

            if verb_time:
                t2=time.time()
                print ("Inference time:", t2-t1)
                
            # decoding 
            decoded = [self.tokenizer.decode(generation_output.sequences[i][len(encoded_inputs["input_ids"][i]):], skip_special_tokens=True) for i in range(len(generation_output.sequences))]

            log_prob = None
            if output_log_probs:
                log_prob = generation_output.scores
                log_prob_list.extend(log_prob)
            decoded_list.extend(decoded)
        if not output_log_probs:
            log_prob_list = None

        return GenerateOutput(decoded_list, log_prob_list)
    
    def _convert_stop_to_end_token_id(self, eos_token_id_input):
        eos_token_id = []
        if eos_token_id_input is not None:
            if not isinstance(eos_token_id_input, list):
                eos_token_id_input = [eos_token_id_input]
            for token in eos_token_id_input:
                tokenized = self.tokenizer.encode(token, add_special_tokens=False)
                if len(tokenized) != 1:
                    warnings.warn(f'the eos_token {repr(token)} is encoded into {tokenized} with length != 1, '
                                f'using {tokenized[-1]} as the eos_token_id')
                token = tokenized[-1]
                    
        eos_token_id.append(self.tokenizer.eos_token_id)
        return eos_token_id
        

    @torch.no_grad()
    def get_next_token_logits(
        self,
        prompt: Union[str, list[str]],
        candidates: Union[list[str], list[list[str]]]) -> list[np.ndarray]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)
        cand_tokens = []
        for candidate in candidates:
            cand_tokens.append([])
            for cand in candidate:
                token = self.tokenizer.encode(cand, add_special_tokens=False)
                if len(token) != 1:
                    warnings.warn(f'candidate {cand} corresponds to {len(token)} instead of 1')
                cand_tokens[-1].append(token[1] if len(token) > 1 else token[0])
        

        bsz = len(prompt)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)

        tokens = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            all_logits = self.model(**tokens, return_dict=True).logits[:,-1,:].squeeze(1)

        logits = []
        for case_logits, cand in zip(all_logits, cand_tokens):
            logits.append(case_logits[cand].cpu().numpy())
        return logits
    
    @torch.no_grad()
    def get_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:
        bsz = len(contents)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)
        prompts_tokens = self.tokenizer(contents, return_tensors='pt',add_special_tokens=False, padding=True).to(self.device)
        prefix_tokens = self.tokenizer(prefix, return_tensors='pt',add_special_tokens=False, padding=True).input_ids[0].to(self.device)
        
        for prompt_tokens in prompts_tokens.input_ids:
            assert torch.all(prompt_tokens[: len(prefix_tokens)] == prefix_tokens), (prompt_tokens, prefix_tokens)

        tokens = prompts_tokens
        logits = self.model(**tokens, return_dict=True).logits
        tokens = prompts_tokens.input_ids
        acc_probs = torch.zeros(bsz).to(self.device)
        for i in range(len(prefix_tokens), tokens.shape[1]):
            probs = torch.softmax(logits[:, i-1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])
        return acc_probs.cpu().numpy()
