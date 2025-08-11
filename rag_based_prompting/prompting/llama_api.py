from typing import List

import torch
import transformers

from RAG4Robots.src.manager import RAGManager
from pybullet_planning.vlm_tools.vlm_api import VLMApi


class LlamaClusterApi(VLMApi):
    name = "Llama 3.3 (Cluster)"

    def __init__(self, rag: RAGManager, **kwargs):
        super(LlamaClusterApi, self).__init__(**kwargs)
        self.model_name = "Llama-3.3-70B-Instruct"
        self.rag_manager = rag

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f'meta-llama/{self.model_name}')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_id = transformers.AutoModelForCausalLM.from_pretrained(
            f'meta-llama/{self.model_name}',
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.generation_pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            device_map="auto",  # finds GPU
        )

    def _ask(self, prompt: str, image_path: str = "None", continue_chat: bool = True, max_tokens: int = 1000,
             temperature: float = 0.0, **kwargs):
        messages = [
            {"role": "system", "content": kwargs.get("sys_msg", "")},
            {"role": "user", "content": prompt},
        ]
        prompt = self.generation_pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        terminators = [
            self.generation_pipe.tokenizer.eos_token_id,
            self.generation_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.generation_pipe(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=False,
            temperature=temperature,
            top_p=None
        )
        result = outputs[0]["generated_text"][len(prompt):]
        return result


class LlamaLocalApi(VLMApi):
    name = "Llama 3.2 (Local)"

    def __init__(self, rag: RAGManager, **kwargs):
        super(LlamaLocalApi, self).__init__(**kwargs)
        self.model_name = "Llama-3.2-1B-Instruct"
        self.rag_manager = rag

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f'meta-llama/{self.model_name}')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model_id = transformers.AutoModelForCausalLM.from_pretrained(
            f'meta-llama/{self.model_name}',
            torch_dtype=torch.float32,  # CPU-friendly
            device_map={"": "cpu"},  # Force CPU usage
            trust_remote_code=True
        )
        self.generation_pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            device_map={"": "cpu"},  # Explicit CPU mapping
        )

    def _ask(self, prompt: str, image_path: str = "None", continue_chat: bool = True, max_tokens: int = 2500,
             temperature: float = 0.0, **kwargs):
        messages = [
            {"role": "system", "content": kwargs.get("sys_msg", "")},
            {"role": "user", "content": prompt},
        ]
        prompt = self.generation_pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        terminators = [
            self.generation_pipe.tokenizer.eos_token_id,
            self.generation_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.generation_pipe(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=False,
            temperature=temperature,
            top_p=None
        )
        result = outputs[0]["generated_text"][len(prompt):]
        return result


def get_context_through_rag(rag: RAGManager, query: str) -> List[str]:
    cont_rank = rag.query_all_dbs(query)
    contexts = []
    for cont in cont_rank:
        contexts.append(cont[0])
    return contexts
