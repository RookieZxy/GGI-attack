
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, pipeline,  LlamaForCausalLM, LlamaTokenizer
import abc
from typing import List
import torch

#TODO: MODIFY RETURN VALUES
 
from utils.constants import PATH_TO_LLAMA_WEIGHTS, PATH_TO_LLAMA_TOKENIZER, PATH_TO_VICUNA

class LanguageModel():
    def __init__(self, max_gen_length, num_gen_seq):
        self.max_gen_length = max_gen_length
        self.num_gen_seq = num_gen_seq
        self.torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device_num = 0 if torch.cuda.is_available() else -1

    @abc.abstractmethod
    def load_model(self, model_name: str, **kwargs):
        """Load the language model."""
        pass

    def generate_text(self, prompts: List[str], seed_text = None) -> List[str]:
        """Generate text given a prompts."""
        if seed_text is not None:
            prompts = [seed_text + " " + cur_prompt for cur_prompt in prompts]
        return self._generate(prompts)

    @abc.abstractmethod
    def _generate(self, prompts:  List[str]) -> List[str]:
        pass

    @abc.abstractmethod
    def get_vocab_tokens(self) -> List[str]:
        '''
        Returns the vocabulary of the language model
        Necessary so that the tokenizer only searches over valid tokens
        '''
        pass

class HuggingFaceLanguageModel(LanguageModel):
    def load_model(self, model_name: str):
        valid_model_names = ["llama-7b-hf", "gpt2-xl", "opt-2.7b", "opt-6.7b"]
        assert model_name in valid_model_names, f"model name was {model_name} but must be one of {valid_model_names}"
        self.model_name = model_name
        # self.generator = pipeline("text-generation", model=model_name, device=self.device_num)

    # def _generate(self, prompts: List[str]) -> List[str]:
    #     gen_texts = self.generator(prompts, max_length=self.max_gen_length, num_return_sequences=self.num_gen_seq, top_k=50, do_sample = True)
    #     gen_texts = [(prompts[i],[cur_dict['generated_text'][len(prompts[i]):] for cur_dict in gen]) for i,gen in enumerate(gen_texts)]
    #     return gen_texts

    def get_vocab_tokens(self):

        if "gpt2" in self.model_name:
            lm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif self.model_name == 'llama-7b-hf':
            model_name = 'decapoda-research/llama-7b-hf'
            lm_tokenizer = LlamaTokenizer.from_pretrained(model_name)
        elif self.model_name == 'opt-2.7b':
            model_name = 'facebook/opt-2.7b'
            lm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif self.model_name == 'opt-6.7b':
            model_name = 'facebook/opt-6.7b'
            lm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        return list(lm_tokenizer.get_vocab().keys())


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False