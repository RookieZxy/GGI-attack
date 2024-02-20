import torch
import gpytorch
import numpy as np 
import sys 
import copy 
from gpytorch.mlls import PredictiveLogLikelihood 
from utils.bo_utils.trust_region import (
    TrustRegionState, 
    generate_batch, 
    update_state
)
from utils.bo_utils.ppgpr import (
    GPModelDKL,
)
from torch.utils.data import (
    TensorDataset, 
    DataLoader
) 

from utils.objectives.image_generation_objective import ImageGenerationObjective  
from utils.constants import trimmed_mean
 
import math 
import os 
# import wandb


import random 
from utils.constants import PREPEND_TASK_VERSIONS
from tqdm import tqdm
import abc
from utils.objectives.text_generation_objective import TextGenerationObjective

class RunOptim():
    def __init__(self, 
        n_tokens,
        max_n_calls,
        max_allowed_calls_without_progress,
        acq_func,
        failure_tolerance,
        success_tolerance,
        init_n_epochs,
        n_epochs,
        n_init_per_prompt,
        hidden_dims,
        lr,
        batch_size,
        vocab,
        best_baseline_score,
        prompts_to_texts,
        texts_to_losses,
        token_embedder,  
        num_shots,
        break_after_success,
        attack_position,
        # tracker,
        # table,
        # at      
        ):

        self.n_tokens = n_tokens
        self.max_n_calls = max_n_calls
        self.max_allowed_calls_without_progress = max_allowed_calls_without_progress
        self.acq_func = acq_func

        self.failure_tolerance = failure_tolerance
        self.success_tolerance = success_tolerance
        self.init_n_epochs = init_n_epochs 
        self.n_epochs = n_epochs 
        self.n_init_per_prompt = n_init_per_prompt
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.vocab = vocab
        self.best_baseline_score = best_baseline_score
        self.beat_best_baseline = False # Has not beat baseline yet
        self.prompts_to_texts = prompts_to_texts
        self.texts_to_losses = texts_to_losses
        self.token_embedder = token_embedder
        self.token_position = ['' for i in range(int(num_shots))]
        self.token_list = []
        self.break_after_success = break_after_success
        self.attack_position = attack_position

        
        self.num_optim_updates = 0

        # Optimization has losses, prompts, and generated text
        self.losses = None
        self.prompts = None 
        self.generated_text = None
        self.embeddings = None
        self.losses_variety = None

        self.num_calls = 0

        
        self.seed_text = None
        
        # flags for wandb recording 
        self.update_state_fix = True 
        self.update_state_fix2 = True 
        self.update_state_fix3 = True 
        self.record_most_probable_fix2 = True 
        self.flag_set_seed = True
        self.flag_fix_max_n_tokens = True
        self.flag_fix_args_reset = True  
        self.flag_reset_gp_new_data = True # reset gp every 10 iters up to 1024 data points 
        self.n_init_pts = self.batch_size * self.n_init_per_prompt
        assert self.n_init_pts % self.batch_size == 0

        self.objective = None
    def log_values(self):
        losses = self.losses
        prompts = self.prompts
        token_list = self.token_list
        generated_text = self.generated_text
       
        best_index = losses.argmax()

        best_token_list = token_list[best_index]
        print(f"BEST TOKEN LIST: {best_token_list}")

        print(f"BEST LOSS: {round(losses[best_index].item(),3)}")

        
        print("\n\n")


    def get_init_prompts(self):
        # TODO: Fix this to be just concatenating tokens
        starter_vocab = self.vocab
        prompts = [] 
        iters = math.ceil(self.n_init_pts / self.n_init_per_prompt) 
        for i in range(iters):
            prompt = ""
            for j in range(self.n_tokens): # N
                prompt += random.choice(starter_vocab)
            prompts.append(prompt)

        return prompts
    
    def get_objective(self):
        return TextGenerationObjective(
            n_tokens=self.n_tokens,
            lb = None,
            ub = None,
            prompts_to_texts = self.prompts_to_texts,
            texts_to_losses = self.texts_to_losses,
            token_embedder = self.token_embedder,
        )

    def get_init_data(self,):
        assert self.objective is not None
        print("Computing Scores for Initialization Data")
        # initialize random starting data 
        self.losses, self.embeddings, self.prompts, self.generated_text, self.token_list = [], [], [], [], []

        # if do batches of more than 10, get OOM 
        n_batches = math.ceil(self.n_init_pts / self.batch_size) 

        for ix in range(n_batches): 
            cur_embedding = torch.normal(mean = self.weights_mean, std = self.weights_std, size=(self.batch_size, self.objective.dim))
            
            self.embeddings.append(cur_embedding)  
            prompts, losses, token_list = self.objective(cur_embedding.to(torch.float16), self.token_position, self.attack_position, )
            self.token_list = self.token_list + token_list
            self.losses_variety = []
            self.losses_variety += losses.tolist()

            self.losses.append(losses) 
            self.prompts = self.prompts + prompts
        self.losses = torch.cat(self.losses).detach().cpu().unsqueeze(-1)
        self.embeddings = torch.cat(self.embeddings).float().detach().cpu()

    def log_final_values(self, attack_optimizer):
        if attack_optimizer == "random":
            return self.prompts
        else:
            best_index = self.losses.argmax()
            best_token_list = self.token_list[best_index]
            return best_token_list, self.losses[best_index], self.losses_variety


    @abc.abstractmethod
    def optim(self):
        pass

    def call_oracle_and_update_next(self, embeddings_next):
        prompts, mean_loss, token_list = self.objective(embeddings_next.to(torch.float16), self.token_position, self.attack_position)
        self.losses_variety += mean_loss.tolist()
        
        self.token_list = self.token_list + token_list

        self.prompts = self.prompts + prompts # prompts 
        return mean_loss
        

class SquareGreedyAttackOptim(RunOptim):
    
    def optim(self):
        self.objective = self.get_objective()
        self.weights_mean, self.weights_std = self.token_embedder.get_embeddings_mean_and_std()
        self.get_init_data()
        
        print("Starting Square Attack")
        AVG_DIST_BETWEEN_VECTORS = self.token_embedder.get_embeddings_avg_dist()
        prev_best = -torch.inf 
        n_iters = 0
        n_calls_without_progress = 0
        prev_loss_batch_std = self.losses.std().item() 
        print("Starting Main Optimization Loop")
        pbar = tqdm(total = self.max_n_calls)
        while self.objective.num_calls < self.max_n_calls:

            if self.losses.max().item() > prev_best: 
                n_calls_without_progress = 0
                prev_best = self.losses.max().item() 
                self.log_values() # Log values due to improvement
                if prev_best > self.best_baseline_score: 
                    self.beat_best_baseline = True 
                # replace roken_position, if find better loss
                best_index = self.losses.argmax()
                self.token_position = self.token_list[best_index]
            else:
                n_calls_without_progress += self.batch_size 
            
            if n_calls_without_progress > self.max_allowed_calls_without_progress:
                break
            prev_loss_batch_std = max(prev_loss_batch_std, 1e-4)
            noise_level = AVG_DIST_BETWEEN_VECTORS / (10*prev_loss_batch_std) # One tenth of avg dist between vectors
            embedding_center = self.embeddings[self.losses.argmax(), :].squeeze() 
            embedding_next = [] 
            for _ in range(self.batch_size):
                embedding_n = copy.deepcopy(embedding_center)
                # select random 10% of dims to modify  
                dims_to_modify = random.sample(range(self.embeddings.shape[-1]), int(self.embeddings.shape[-1]*0.1))
                rand_noise =  torch.normal(mean=torch.zeros(len(dims_to_modify),), std=torch.ones(len(dims_to_modify),)*noise_level) 
                embedding_n[dims_to_modify] = embedding_n[dims_to_modify] + rand_noise 
                embedding_next.append(embedding_n.unsqueeze(0)) 
            embedding_next = torch.cat(embedding_next) 
            self.embeddings = torch.cat((self.embeddings, embedding_next.detach().cpu()), dim=-2) 
            losses_next = self.call_oracle_and_update_next(embedding_next)
            prev_loss_batch_std = losses_next.std().item() 
            losses_next = losses_next.unsqueeze(-1)
            self.losses = torch.cat((self.losses, losses_next.detach().cpu()), dim=-2) 
            n_iters += 1 
            pbar.update(self.objective.num_calls - pbar.n)
        pbar.close()
   
class GreedySearchOptim(RunOptim):
    
    def optim(self):
        self.losses, self.prompts, self.generated_text, self.losses_variety =torch.empty(0,1), [], [],[]
        print("Starting Random Search Attack")
        
        
        prev_best = -torch.inf 
        n_iters = 0
        n_calls_without_progress = 0
        
        print("Starting Main Optimization Loop")
        pbar = tqdm(total = self.max_n_calls)
        self.num_calls = 0
        while self.num_calls < self.max_n_calls:
            
            self.update_optim()

            if self.losses.max().item() > prev_best: 
                n_calls_without_progress = 0
                prev_best = self.losses.max().item() 
                self.log_values() # Log values due to improvement
                if prev_best > self.best_baseline_score: 
                    self.beat_best_baseline = True
                best_index = self.losses.argmax()
                
                self.token_position = self.token_list[best_index]
            else:
                n_calls_without_progress += self.batch_size 
            # random
            if n_calls_without_progress > self.max_allowed_calls_without_progress:
                break
            
            n_iters += 1 
            pbar.update(self.num_calls - pbar.n)
        pbar.close()

    def get_random_prompts(self,batch_size):
        prompts = []
        for i in range(batch_size):
            
            prompt = " ".join(random.sample(self.token_embedder.sorted_vocab_keys,self.n_tokens))
            
            prompts.append(prompt)
        return prompts
            
                
    def update_optim(self):
        prompts = self.get_random_prompts(self.batch_size)

        if (self.attack_position == 'demo_suffix' or self.attack_position == 'demo_prefix'):
        # generated_text = self.prompts_to_texts(prompts)
            generated_text_list, token_list, labels_list = self.prompts_to_texts(prompts, self.token_position, self.attack_position)
            num_shots = len(self.token_position)
            losses_list = []

            for generated_texts, labels in zip(generated_text_list, labels_list):
                losses = torch.Tensor(self.texts_to_losses(generated_texts, labels))
                mean_loss = torch.mean(losses, axis = 0)
                losses_list.append(mean_loss)
            num = len(prompts)
            temp_losses = []
            temp_token_list = []
            losses_list = torch.Tensor(losses_list)

            for i in range(num):
                temp = copy.deepcopy(losses_list[i* num_shots: i*num_shots +num_shots])
                idx = temp.argmax() + i* num_shots
                temp_losses.append(losses_list[idx].tolist())
                temp_token_list.append(token_list[idx])
            self.losses_variety += temp_losses
            losses = torch.Tensor(temp_losses)
            losses = losses.unsqueeze(-1)
            self.losses = torch.cat((self.losses, losses.detach().cpu()), dim=-2) 
            self.token_list = self.token_list + temp_token_list
            self.prompts = self.prompts + prompts 
            self.num_calls += len(prompts)
    
class RandomSearchOptim(RunOptim):
    def optim(self):
        self.losses, self.prompts, self.generated_text, self.losses_variety =torch.empty(0,1), [], [],[]
        print("Starting Random Search Attack")
        

        
        prompts = self.get_random_prompts(10)
        self.prompts = self.prompts + prompts 
       

    def get_random_prompts(self,batch_size):
        prompts = []
        for i in range(batch_size):
            prompt = []
            for i in range(self.n_tokens):
                temp = random.sample(self.token_embedder.sorted_vocab_keys,2)
                prompt += [' '.join(temp)]
            prompts.append(prompt)
        return prompts
            
                
    def update_optim(self):
        prompts = self.get_random_prompts(self.batch_size)
        if (self.attack_position == 'demo_suffix' or self.attack_position == 'demo_prefix'):
            generated_text_list, token_list, labels_list = self.prompts_to_texts(prompts, self.token_position, self.attack_position)
            num_shots = len(self.token_position)
            losses_list = []
            for generated_texts, labels in zip(generated_text_list, labels_list):
                losses = torch.Tensor(self.texts_to_losses(generated_texts, labels))
                mean_loss = torch.mean(losses, axis = 0)
                losses_list.append(mean_loss)

            losses_list = torch.Tensor(losses_list)
     
            self.losses_variety += losses_list.tolist()
            losses = torch.Tensor(losses_list)
            losses = losses.unsqueeze(-1)
            self.losses = torch.cat((self.losses, losses.detach().cpu()), dim=-2) 
            self.token_list = self.token_list + token_list
            self.prompts = self.prompts + prompts 
            self.num_calls += len(prompts)

class SquareAttackOptim(RunOptim):
    
    def optim(self):
        self.objective = self.get_objective()
        self.weights_mean, self.weights_std = self.token_embedder.get_embeddings_mean_and_std()
        self.get_init_data()
        torch.cuda.empty_cache()

        
        print("Starting Square Attack")
        AVG_DIST_BETWEEN_VECTORS = self.token_embedder.get_embeddings_avg_dist()
        prev_best = -torch.inf 
        n_iters = 0
        n_calls_without_progress = 0
        prev_loss_batch_std = self.losses.std().item() 
        print("Starting Main Optimization Loop")
        pbar = tqdm(total = self.max_n_calls)
        while self.objective.num_calls < self.max_n_calls:

            if self.losses.max().item() > prev_best: 
                n_calls_without_progress = 0
                prev_best = self.losses.max().item() 
                self.log_values() # Log values due to improvement
                if prev_best > self.best_baseline_score: 
                    self.beat_best_baseline = True 
            else:
                n_calls_without_progress += self.batch_size 
            
            if n_calls_without_progress > self.max_allowed_calls_without_progress:
                break
            prev_loss_batch_std = max(prev_loss_batch_std, 1e-4)
            noise_level = AVG_DIST_BETWEEN_VECTORS / (10*prev_loss_batch_std) # One tenth of avg dist between vectors
            embedding_center = self.embeddings[self.losses.argmax(), :].squeeze() 
            embedding_next = [] 
            for _ in range(self.batch_size):
                embedding_n = copy.deepcopy(embedding_center)
                # select random 10% of dims to modify  
                dims_to_modify = random.sample(range(self.embeddings.shape[-1]), int(self.embeddings.shape[-1]*0.1))
                print(torch.ones(len(dims_to_modify)))
                rand_noise =  torch.normal(mean=torch.zeros(len(dims_to_modify),), std=torch.ones(len(dims_to_modify),)*noise_level) 
                embedding_n[dims_to_modify] = embedding_n[dims_to_modify] + rand_noise 
                embedding_next.append(embedding_n.unsqueeze(0)) 
            embedding_next = torch.cat(embedding_next) 
            self.embeddings = torch.cat((self.embeddings, embedding_next.detach().cpu()), dim=-2) 
            losses_next = self.call_oracle_and_update_next(embedding_next)
            prev_loss_batch_std = losses_next.std().item() 
            losses_next = losses_next.unsqueeze(-1)
            self.losses = torch.cat((self.losses, losses_next.detach().cpu()), dim=-2) 
            n_iters += 1 
            pbar.update(self.objective.num_calls - pbar.n)
        pbar.close()
