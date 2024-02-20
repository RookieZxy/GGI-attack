from typing import List, Tuple
import torch
import abc
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline, pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer


class LossFunction():
    def __init__(self):
        self.goal = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.device_num = 0 if torch.cuda.is_available() else -1
        self.device_num = 0 if torch.cuda.is_available() else -1
    @abc.abstractmethod
    def __call__(self, generated_texts: List[Tuple[str,List[str]]], **kwargs) -> torch.Tensor:
        '''
        Compute the loss on the generated text and prompts
    
        '''
        pass


        
class TopicLoss(LossFunction):
    def __init__(self, model_name, label, attack_position, attack_optimizer):
        super().__init__()
        self.goal = 'maximize'
        self.model_name = model_name
        self.attack_optimizer = attack_optimizer
        
        # change the model here 
        if model_name == 'gpt2-xl':
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
        if model_name == 'llama-7b-hf':
            self.model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-7b-hf', torch_dtype=torch.float16, trust_remote_code=True,).to(self.device).eval()
            self.tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf', trust_remote_code=True)
        if 'opt' in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True,).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.label = label
        self.attack_position = attack_position

    def __call__(self,  sentences_list, labels) -> torch.Tensor:
        n_prompts = len(sentences_list)

        probs = []
        with torch.no_grad():
            for sentence, label in zip(sentences_list, labels):
                generated = self.tokenizer(sentence, return_tensors='pt').to(self.device)
                output = self.model(**generated, output_attentions=True)
                a = output[0][:, -1, :]
                next_token_logtis = torch.tensor(a)
                p = torch.nn.functional.softmax(next_token_logtis).tolist() 

                if self.model_name == 'gpt2-xl':
                    if label == 0:
                        probs.append(p[0][6894])
                    elif label == 1:
                        probs.append(p[0][32945])
                    elif label == 2:
                        probs.append(p[0][22680])
                    elif label == 3:
                        probs.append(p[0][45503])
                elif self.model_name == 'llama-7b-hf':
                    if label == 0:
                        probs.append(p[0][11526])
                    elif label == 1:
                        probs.append(p[0][29879])
                    elif label == 2:
                        probs.append(p[0][8262])
                    elif label == 3:
                        probs.append(p[0][21695])
                    # if label == 0:
                    #     output=None
                    #     a = None
                    #     torch.cuda.empty_cache()
                    #     loss = p[0][11526]
                    # elif label == 1:
                    #     output=None
                    #     a = None
                    #     torch.cuda.empty_cache()

                    #     loss = p[0][29879]
                    #     sentence += "s"
                    #     generated = self.tokenizer(sentence, return_tensors='pt').to(self.device)
                    #     output = self.model(**generated, output_attentions=True)
                    #     a = output[0][:, -1, :]
                    #     next_token_logtis = torch.tensor(a)
                    #     p = torch.nn.functional.softmax(next_token_logtis).tolist() 
                    #     loss *= p[0][4011]
                    #     probs.append(loss)
                    # elif label == 2:
                    #     output=None
                    #     a = None
                    #     torch.cuda.empty_cache()

                    #     loss = p[0][8262]
                    #     sentence += "bus"
                    #     generated = self.tokenizer(sentence, return_tensors='pt').to(self.device)
                    #     output = self.model(**generated, output_attentions=True)
                    #     a = output[0][:, -1, :]
                    #     next_token_logtis = torch.tensor(a)
                    #     p = torch.nn.functional.softmax(next_token_logtis).tolist() 
                    #     loss *= p[0][3335]
                    #     probs.append(loss)
                    # elif label == 3:
                    #     output=None
                    #     a = None
                    #     torch.cuda.empty_cache()

                    #     loss = p[0][21695]
                    #     sentence += "techn"
                    #     generated = self.tokenizer(sentence, return_tensors='pt').to(self.device)
                    #     output = self.model(**generated, output_attentions=True)
                    #     a = output[0][:, -1, :]
                    #     next_token_logtis = torch.tensor(a)
                    #     p = torch.nn.functional.softmax(next_token_logtis).tolist() 
                    #     loss *= p[0][3002]
                    #     probs.append(loss)

        probs = torch.tensor(probs).reshape(n_prompts, 1)

        return probs
    
    def Next_generated_label_idx(self, generated_text):
        generated = self.tokenizer(generated_text, return_tensors='pt').to(self.device)
        # attention_mask = generated['attention_mask'].tolist()
        # position = attention_mask[0].index(0)-1
        
        output = self.model(**generated, output_attentions=True)
        # a = output[0][:, position, :].tolist()
        a = output[0][:, -1, :].tolist()
        next_token_logtis = torch.tensor(a)
        p = torch.nn.functional.softmax(next_token_logtis).tolist() 
        max = 0
        idx = 0
        for i in range(len(p[0])):
            if max < p[0][i]:
                max = p[0][i]
                idx = i

        # generated = self.tokenizer(f"{generated_text}", return_tensors='pt').to('cuda:1')
        sample_outputs = self.model.generate(**generated ,do_sample=False, temperature=0)
        predicted_text = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

        return idx, predicted_text



class SentimentLoss(LossFunction):
    def __init__(self, model_name, label, attack_position, attack_optimizer):
        super().__init__()
        self.goal = 'maximize'
        self.model_name = model_name
        self.attack_optimizer = attack_optimizer
        
        # https://huggingface.co/martin-ha/toxic-comment-model
        # model_path = "martin-ha/toxic-comment-model"
        # model_dir = '/home/xiangyu/Project/Model/'
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # change the model here 
        if model_name == 'gpt2-xl':
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
        elif model_name == 'llama-7b-hf':
            self.model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-7b-hf', torch_dtype=torch.float16, trust_remote_code=True,).to(self.device).eval()

            self.tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf', trust_remote_code=True)
        elif model_name == 'opt-2.7b':
            self.model = AutoModelForCausalLM.from_pretrained('facebook/opt-2.7b', torch_dtype=torch.float16, trust_remote_code=True,).to(self.device).eval()

            self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-2.7b', trust_remote_code=True)
        elif model_name == 'opt-6.7b':
            self.model = AutoModelForCausalLM.from_pretrained('facebook/opt-6.7b', torch_dtype=torch.float16, trust_remote_code=True,).to(self.device).eval()

            self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-6.7b', trust_remote_code=True)
        self.label = label
        self.attack_position = attack_position


        # self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer,device=self.device_num)

    def __call__(self,  sentences_list, labels) -> torch.Tensor:
        # if self.attack_optimizer == 'random':
        #     return 1
        # elif self.attack_optimizer == 'greedy':
            # for sentences in sentences_list:
        n_prompts = len(sentences_list)

        probs = []
        with torch.no_grad():
            for sentence, label in zip(sentences_list, labels):
            
                generated = self.tokenizer(sentence, return_tensors='pt').to(self.device)

                output = self.model(**generated, output_attentions=True)
                a = output[0][:, -1, :]
                next_token_logtis = torch.tensor(a)
                p = torch.nn.functional.softmax(next_token_logtis).tolist() 

                if self.model_name == 'gpt2-xl':
                    if label == 1:
                        probs.append(p[0][31591])
                    elif label == 0:
                        probs.append(p[0][24561])
                if self.model_name == 'llama-7b-hf':
                    if label == 1:
                        probs.append(p[0][22198])
                    elif label == 0:
                        output=None
                        a = None
                        torch.cuda.empty_cache()
                        loss = p[0][1066]
                        sentence += "pos"
                        generated = self.tokenizer(sentence, return_tensors='pt').to(self.device)
                        output = self.model(**generated, output_attentions=True)
                        a = output[0][:, -1, :]
                        next_token_logtis = torch.tensor(a)
                        p = torch.nn.functional.softmax(next_token_logtis).tolist() 
                        loss *= p[0][3321]
                        output=None
                        a = None
                        torch.cuda.empty_cache()
                        probs.append(loss)
                    # if label == 1:
                    #     probs.append(p[0][22198])
                    # elif label == 0:
                    #     probs.append(p[0][1066])
                elif 'opt' in self.model_name:
                    if label == 1:
                        probs.append(p[0][33407])
                    elif label == 0:
                        probs.append(p[0][22173])


        probs = torch.tensor(probs).reshape(n_prompts, 1)

        return probs
    
    def Next_generated_label_idx(self, generated_text):
        generated = self.tokenizer(generated_text, return_tensors='pt').to(self.device)
        # attention_mask = generated['attention_mask'].tolist()
        # position = attention_mask[0].index(0)-1
        
        output = self.model(**generated, output_attentions=True)
        # a = output[0][:, position, :].tolist()
        a = output[0][:, -1, :].tolist()
        next_token_logtis = torch.tensor(a)
        p = torch.nn.functional.softmax(next_token_logtis).tolist() 
        max = 0
        idx = 0
        for i in range(len(p[0])):
            if max < p[0][i]:
                max = p[0][i]
                idx = i

        # generated = self.tokenizer(f"{generated_text}", return_tensors='pt').to('cuda:1')
        sample_outputs = self.model.generate(**generated ,do_sample=False, temperature=0)
        predicted_text = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

        return idx, predicted_text

