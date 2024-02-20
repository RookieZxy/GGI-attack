import torch 
import re
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import string

import abc

class TokenEmbedding():
    def __init__(self,  language_model_vocab_tokens):
        self.embed_dim = None
        self.language_model_vocab_tokens = language_model_vocab_tokens
        self.torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.all_embeddings = None
        self.sorted_vocab_keys = None # a tensor of vocab tokens sorted by index
        self.all_vocab = None
        self.filterTokenizer = AutoTokenizer.from_pretrained('martin-ha/toxic-comment-model')
        self.filterModel = AutoModelForSequenceClassification.from_pretrained('martin-ha/toxic-comment-model',num_labels=2)

    @abc.abstractmethod
    def set_embeddings(self):
        pass

    @abc.abstractmethod
    def batched_tokens_to_embeddings(self, batched_tokens):
        pass
    
    #TODO: Change this from batched to vectorized
    def batched_embeddings_to_tokens(self, batched_embeddings):
        '''
            Given batched_embeddings, convert to tokens
            args:
                embeddings: (batch_size, num_tokens, embed_dim) word embedding
            returns:
                proj_tokens: (batch_size, num_tokens) projected tokens
        '''
        batch_size = batched_embeddings.shape[0]
        proj_tokens = []
        for i in range(batch_size):
            subset_embedding = batched_embeddings[i,:,:]
            proj_tokens.append(self.proj_embeddings_to_tokens(subset_embedding))
        return proj_tokens

    def proj_embeddings_to_tokens(self, embeddings):
        distances = torch.norm(self.all_embeddings.cuda(0) - embeddings.cuda(0).unsqueeze(1), dim = 2)
        n_prompts = embeddings.shape[0]
        vocab_size = self.all_embeddings.shape[1]
        assert distances.shape == (n_prompts, vocab_size), \
            f"Distances was shape {distances.shape} but should be  ({embeddings.shape[0]}, {self.embed_dim})"
            
        min_indices = torch.argmin(distances, axis = 1).tolist()
        # SPACES BETWEEN EACH TOKEN
        proj_tokens = " ".join(self.sorted_vocab_keys[index] for index in min_indices) 
        return proj_tokens

    def get_vocab(self):
        return self.sorted_vocab_keys
        
    def get_embeddings_mean_and_std(self):
        return torch.mean(self.all_embeddings).item(), torch.std(self.all_embeddings).item()

    def get_embeddings_avg_dist(self):
        '''
        Computes average Euclidean distance between embeddings
        '''
        embeds = self.all_embeddings
        distances = torch.cdist(embeds,embeds)


        # Get the upper triangular part of the matrix, excluding the diagonal
        upper_triangular = torch.triu(distances, diagonal=1)

        # Compute the average pairwise distance
        # print(upper_triangular.numel())
        # print(embeds.shape[0])
        # print(upper_triangular.numel() - embeds.shape[0])
        avg_distance = torch.sum(upper_triangular) / (upper_triangular.numel() - embeds.shape[0]) * 2
        return avg_distance.item()


class GPTEmbedding(TokenEmbedding):
    def __init__(self, language_model_vocab_tokens):
        super().__init__(language_model_vocab_tokens)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Can be changed to other models
        embedding_model = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.all_vocab = tokenizer.get_vocab()

        lm = AutoModelForCausalLM.from_pretrained(embedding_model).to(self.torch_device)
        print(lm.get_input_embeddings())
        self.all_embeddings = lm.get_input_embeddings().weight # torch tensor 50272, 768
        self.set_embeddings() #self.all_embeddings has dim 1, 50272, 768 after unsqueezing
        self.embed_dim = self.all_embeddings.shape[2]# 768

    def testSingleInput(self):
        original_vocab_keys = self.valid_vocab_keys.copy()

        temp = original_vocab_keys.copy()
        english_chars = set(string.ascii_letters)
        for token in temp :
            if not all(char in english_chars for char in token):
                original_vocab_keys.remove(token) 


        self.valid_vocab_keys = original_vocab_keys
        return True

    
    def set_embeddings(self):
        # pipeline = TextClassificationPipeline(model=self.filterModel, tokenizer=self.filterTokenizer)
        # Valid vocab is the intersection between all vocab and the language model vocab
        self.valid_vocab_keys = set(self.all_vocab.keys()).intersection(set(self.language_model_vocab_tokens))
        # Filter the toxic vocab
        self.testSingleInput()
        # Filter the valid embeddings from the original embeddings tensor
        self.valid_indices = sorted([self.all_vocab[key] for key in self.valid_vocab_keys])
        
        self.all_embeddings = self.all_embeddings[self.valid_indices].unsqueeze(0)
        # Create a new list with valid keys sorted by their indices
        self.sorted_vocab_keys = [key for key, value in sorted(self.all_vocab.items(), key=lambda item: item[1]) if key in self.valid_vocab_keys]

    def proj_embeddings_to_tokens(self, embeddings):
        proj_tokens =  super().proj_embeddings_to_tokens(embeddings)
        return proj_tokens.replace("Ġ", " ")
    

class OPTEmbedding(TokenEmbedding):
    def __init__(self, language_model_vocab_tokens):
        super().__init__(language_model_vocab_tokens)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Can be changed to other models
        embedding_model = 'facebook/opt-2.7b'
        tokenizer = AutoTokenizer.from_pretrained(embedding_model, use_fast=False)
        self.all_vocab = tokenizer.get_vocab()

        lm = AutoModelForCausalLM.from_pretrained(embedding_model).to(self.torch_device)
        print(lm.get_input_embeddings())
        self.all_embeddings = lm.get_input_embeddings().weight # torch tensor 50272, 768
        self.set_embeddings() #self.all_embeddings has dim 1, 50272, 768 after unsqueezing
        self.embed_dim = self.all_embeddings.shape[2]# 768

    def testSingleInput(self):
        original_vocab_keys = self.valid_vocab_keys.copy()

        temp = original_vocab_keys.copy()
        english_chars = set(string.ascii_letters)
        for token in temp :
            if not all(char in english_chars for char in token):
                original_vocab_keys.remove(token) 
        self.valid_vocab_keys = original_vocab_keys
        return True

    
    def set_embeddings(self):
        self.valid_vocab_keys = set(self.all_vocab.keys()).intersection(set(self.language_model_vocab_tokens))
        # Filter the toxic vocab
        self.testSingleInput()
        # Filter the valid embeddings from the original embeddings tensor
        self.valid_indices = sorted([self.all_vocab[key] for key in self.valid_vocab_keys])
        
        self.all_embeddings = self.all_embeddings[self.valid_indices].unsqueeze(0)
        # Create a new list with valid keys sorted by their indices
        self.sorted_vocab_keys = [key for key, value in sorted(self.all_vocab.items(), key=lambda item: item[1]) if key in self.valid_vocab_keys]

    def proj_embeddings_to_tokens(self, embeddings):
        proj_tokens =  super().proj_embeddings_to_tokens(embeddings)
        return proj_tokens.replace("Ġ", " ")
    

class LlamaEmbedding(TokenEmbedding):
    def __init__(self, language_model_vocab_tokens):
        super().__init__(language_model_vocab_tokens)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Can be changed to other models
        embedding_model = "decapoda-research/llama-7b-hf"
        tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')
        self.all_vocab = tokenizer.get_vocab()

        lm = LlamaForCausalLM.from_pretrained(embedding_model, torch_dtype=torch.float16).to(self.torch_device)
        print(lm.get_input_embeddings())
        self.all_embeddings = lm.get_input_embeddings().weight # torch tensor 50272, 768
        self.set_embeddings() #self.all_embeddings has dim 1, 50272, 768 after unsqueezing
        self.embed_dim = self.all_embeddings.shape[2]# 768

    def testSingleInput(self):
        original_vocab_keys = self.valid_vocab_keys.copy()
        # f = open("/home/xiangyu/Project/lab_source_code-master/Project/practice/toxic_words.txt")
        # line  = f.readline()
        # # for item in self.valid_vocab_keys:
        # #     if item in lines:
        # #         original_vocab_keys.remove(item)
        # while line:
        #     original_vocab_keys.remove(line[:-1])
        #     line = f.readline()
        # f.close()
        temp = original_vocab_keys.copy()
        english_chars = set(string.ascii_letters)
        for token in temp :
            if not all(char in english_chars for char in token):
                original_vocab_keys.remove(token) 
        # print(len(original_vocab_keys))


        # end for
        # pool = list(original_vocab_keys)
        # batch = []
        # n_iter = 10
        # n = int(len(pool) / n_iter)
        # for i in range(n_iter-1):
        #     batch.append(pool[i*n : (i+1)*n])
        # batch.append(pool[n*(n_iter-1):])
        # for item in batch:
        #     key_toxicity = pipeline(item, truncation=True, max_length=512, padding=True)
        #     for j in range(len(key_toxicity)):
        #         if key_toxicity[j]['label'] == 'toxic':
        #             original_vocab_keys.remove(item[j])

        self.valid_vocab_keys = original_vocab_keys
        return True

    
    def set_embeddings(self):
        # pipeline = TextClassificationPipeline(model=self.filterModel, tokenizer=self.filterTokenizer)
        # Valid vocab is the intersection between all vocab and the language model vocab
        self.valid_vocab_keys = set(self.all_vocab.keys()).intersection(set(self.language_model_vocab_tokens))
        # Filter the toxic vocab
        self.testSingleInput()
        # Filter the valid embeddings from the original embeddings tensor
        self.valid_indices = sorted([self.all_vocab[key] for key in self.valid_vocab_keys])
        
        self.all_embeddings = self.all_embeddings[self.valid_indices].unsqueeze(0)
        # Create a new list with valid keys sorted by their indices
        self.sorted_vocab_keys = [key for key, value in sorted(self.all_vocab.items(), key=lambda item: item[1]) if key in self.valid_vocab_keys]

    def proj_embeddings_to_tokens(self, embeddings):
        proj_tokens =  super().proj_embeddings_to_tokens(embeddings)
        return proj_tokens.replace("Ġ", " ")

    
