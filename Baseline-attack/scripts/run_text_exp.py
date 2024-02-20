import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
import json
# import wandb
# import tensorflow as tf
import argparse
import os
import copy
os.environ["WANDB_SILENT"] = "true" 
from utils.text_utils.language_model import HuggingFaceLanguageModel
from utils.text_utils.text_losses import  TopicLoss, SentimentLoss
from scripts.run_optimization import  RandomSearchOptim, GreedySearchOptim, SquareGreedyAttackOptim
from utils.constants import tuple_type, trimmed_mean
from utils.token_embedder import OPTEmbedding, LlamaEmbedding, GPTEmbedding

class RunTextExp():
    def __init__(self, lm_args, optim_args, loss_args):
        self.lm_args = lm_args
        self.optim_args = optim_args
        self.loss_args = loss_args

        # Set Language Model
        self.language_model = self.get_language_model(lm_args["language_model"],
                                    lm_args["n_tokens"],
                                    lm_args["seed_text"],
                                    lm_args["num_gen_seq"])
    
        language_model_vocab_tokens = self.language_model.get_vocab_tokens()
        # Set Token Embedder
        self.token_embedder = self.get_token_embedder(lm_args["embedding_model"],language_model_vocab_tokens)
        self.vocab = self.token_embedder.get_vocab()

        # release memory of GPU
        self.language_model=[]
        torch.cuda.empty_cache()

        # Set Loss Function
        self.loss_fn = self.get_loss_fn(loss_args, lm_args["language_model"], lm_args["seed_label"])

        self.set_seed(optim_args["seed"])
        self.optim_args = optim_args


    def set_seed(self,seed):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        if seed is not None:
            torch.manual_seed(seed) 
            random.seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(seed)
    
    def get_language_model(self, language_model_name, n_tokens, seed_text, num_gen_seq):
        seed_text_len = 0
        if seed_text is not None:
            # This is not exact but rough approximation, 
            # TODO: more sophisticated way to get seed text length later
            seed_text_len = [len(item.split() ) * 2 for item in seed_text]
            seed_text_len = max(seed_text_len)
            
        max_num_tokens =  n_tokens + seed_text_len
        if language_model_name in ["gpt2-xl", "llama-7b-hf", "opt-2.7b", "opt-6.7b"]:
            lm = HuggingFaceLanguageModel(max_num_tokens, num_gen_seq)
            lm.load_model(language_model_name)
        else:
            raise NotImplementedError
        return lm

    def get_loss_fn(self, loss_args, model_name, label):
        loss_type = loss_args["loss_type"]
        if loss_type == "sentiment":
            return SentimentLoss(model_name, label, optim_args["attack_position"], optim_args["attack_optimizer"])
        if loss_type == "topic":
            return TopicLoss(model_name, label, optim_args["attack_position"], optim_args["attack_optimizer"])
        else:
            raise NotImplementedError

    def prompts_to_texts(self, prompts, token_position, attack_position):
        sentences_list=[]
        token_list = []
        num_shots = self.optim_args['num_shots']
        labels_list = []

        demos = self.lm_args["demo"]


        if num_shots == '2':
            demos = demos[:2]
        elif num_shots == '4':
            demos = demos[:4]
        elif num_shots == '8':
            demos = demos

        if loss_args["loss_type"] == "sentiment":
            instruction = 'Analyze the sentiment of the last review and respond with either positive or negative. Here are several examples.'

            if self.optim_args["attack_optimizer"] == 'random' or self.optim_args["attack_optimizer"] == 'square':
                for prompt in prompts:
                    input = '' + instruction

                    if attack_position == 'demo_suffix':
                        for (single_demo, single_token) in zip(demos, prompt):
                            input += '\nReview: ' + single_demo['sentence']  + " " + single_token  + '\nSentiment:' + ('positive' if str(single_demo['sentiment'])== '1' else 'negative')
                    elif attack_position == 'demo_prefix':
                        for (single_demo, single_token) in zip(demos, prompt):
                            input += '\nReview: ' + single_token + " " + single_demo['sentence'] + '\nSentiment:' + ('positive' if str(single_demo['sentiment'])== '1' else 'negative')
                    
                    sentences = [input + f'\nReview: {sentence}\nSentiment:' for sentence in self.lm_args["seed_text"]]
                    sentences_list.append(sentences)
                    labels_list.append(self.lm_args["seed_label"].tolist())
                    token_list.append(prompt)
            else:
                for prompt in prompts:
                    for idx in range(int(self.optim_args['num_shots'])):
                        input = '' + instruction
                        replaced_token = copy.deepcopy(token_position)
                        replaced_token[idx] = prompt

                        if attack_position == 'demo_suffix':
                            for (single_demo, single_token) in zip(demos, replaced_token):
                                if single_token == "" :
                                    input += '\nReview: ' + single_demo['sentence'] +'\nSentiment:' + ('positive' if str(single_demo['sentiment'])== '1' else 'negative')
                                else:
                                    input += '\nReview: ' + single_demo['sentence'] + " " + single_token  + '\nSentiment:' + ('positive' if str(single_demo['sentiment'])== '1' else 'negative')
                        elif attack_position == 'demo_prefix':
                            for (single_demo, single_token) in zip(demos, replaced_token):
                                if single_token == "" :
                                    input += '\nReview: ' + single_demo['sentence'] +'\nSentiment:' + ('positive' if str(single_demo['sentiment'])== '1' else 'negative')
                                else:
                                    input += '\nReview: ' + single_token + " " + single_demo['sentence']  + '\nSentiment:' + ('positive' if str(single_demo['sentiment'])== '1' else 'negative')

                        sentences = [input + f'\nReview: {sentence}\nSentiment:' for sentence in self.lm_args["seed_text"]]

                        sentences_list.append(sentences)
                        labels_list.append(self.lm_args["seed_label"].tolist())

                        token_list.append(replaced_token)
        
        elif loss_args["loss_type"] == "topic":
            instruction = 'Classify the topic of the last review. Here are several examples.'

            if self.optim_args["attack_optimizer"] == 'random' or self.optim_args["attack_optimizer"] == 'square':
                for prompt in prompts:
                    input = '' + instruction

                    if attack_position == 'demo_suffix':
                        for (single_demo, single_token) in zip(demos, prompt):
                            label = ''
                            if single_demo['sentiment'] == '0':
                                label = 'world'
                            if single_demo['sentiment'] == '1':
                                label = 'sports'
                            if single_demo['sentiment'] == '2':
                                label = 'business'
                            if single_demo['sentiment'] == '3':
                                label = 'technology'
                            input += '\nReview: ' + single_demo['sentence'] + " "  + single_token  + '\nTopic:' + label
                    elif attack_position == 'demo_prefix':
                        label = ''
                        if single_demo['sentiment'] == '0':
                            label = 'world'
                        if single_demo['sentiment'] == '1':
                            label = 'sports'
                        if single_demo['sentiment'] == '2':
                            label = 'business'
                        if single_demo['sentiment'] == '3':
                            label = 'technology'
                        for (single_demo, single_token) in zip(demos, prompt):
                            input += '\nReview: ' + single_token + " " + single_demo['sentence'] + '\nTopic:' + label
                    
                    sentences = [input + f'\nReview: {sentence}\nTopic:' for sentence in self.lm_args["seed_text"]]
                    sentences_list.append(sentences)
                    labels_list.append(self.lm_args["seed_label"].tolist())
                    token_list.append(prompt)
            else:
                for prompt in prompts:
                    for idx in range(int(self.optim_args['num_shots'])):
                        input = '' + instruction
                        replaced_token = copy.deepcopy(token_position)
                        replaced_token[idx] = prompt

                        if attack_position == 'demo_suffix':
                            for (single_demo, single_token) in zip(demos, replaced_token):
                                label = ''
                                if single_demo['sentiment'] == '0':
                                    label = 'world'
                                if single_demo['sentiment'] == '1':
                                    label = 'sports'
                                if single_demo['sentiment'] == '2':
                                    label = 'business'
                                if single_demo['sentiment'] == '3':
                                    label = 'technology'

                                if single_token == "" :
                                    input += '\nReview: ' + single_demo['sentence'] +'\nTopic:' + label
                                else:
                                    input += '\nReview: ' + single_demo['sentence'] + " " + single_token  + '\nTopic:' + label
                        elif attack_position == 'demo_prefix':
                            for (single_demo, single_token) in zip(demos, replaced_token):
                                label = ''
                                if single_demo['sentiment'] == '0':
                                    label = 'world'
                                if single_demo['sentiment'] == '1':
                                    label = 'sports'
                                if single_demo['sentiment'] == '2':
                                    label = 'business'
                                if single_demo['sentiment'] == '3':
                                    label = 'technology'
                                if single_token == "" :
                                    input += '\nReview: ' + single_demo['sentence'] +'\nTopic:' + label
                                else:
                                    input += '\nReview: ' + single_token + " " + single_demo['sentence']  + '\nTopic:' + label

                        sentences = [input + f'\nReview: {sentence}\nTopic:' for sentence in self.lm_args["seed_text"]]

                        sentences_list.append(sentences)
                        labels_list.append(self.lm_args["seed_label"].tolist())

                        token_list.append(replaced_token)
                   
        return sentences_list, token_list, labels_list    


    def get_token_embedder(self, embedding_model, language_model_vocab_tokens):
        if embedding_model == 'llama-7b-hf':
            return LlamaEmbedding(language_model_vocab_tokens)
        elif embedding_model == 'gpt2':
            return GPTEmbedding(language_model_vocab_tokens)
        elif 'opt' in embedding_model:
            return OPTEmbedding(language_model_vocab_tokens)
        
    def texts_to_losses(self, generated_texts, labels):
        losses = self.loss_fn(generated_texts, labels)
        if self.loss_fn.goal == "minimize":
            losses *= -1.0
        return losses

    def get_baseline_prompts(self):
        loss_type = self.loss_args["loss_type"]
        if loss_type == "count_letter":
            target_str = self.loss_args["target_string"]
            prompts = [f"generate as many of the letter {target_str} as possible:", f"words with many of the letter {target_str}:"]
        elif loss_type == "emotion":
            prompts = ["anger anger anger anger", "fear fear fear fear", "joy joy joy joy", "disgust disgust disgust disgust",
            "neutral neutral neutral neutral", "sadness sadness sadness sadness", "surprise surprise surprise surprise"]
        elif loss_type == "toxicity":
            prompts = ["very nice","very mean","nice","mean"]
        elif loss_type == "perplexity" or "perplexity_seed":
            prompts = ["Fishtail spoons orbit the parrot's enigma", "Quantum trapezoids forge kaleidoscopic xylophones","Petunia exoplanets converge with multidimensional eggplants"]
        else:
            raise NotImplementedError
        return prompts 
    
    def log_clear_sentence(self):
        num_shots = self.optim_args['num_shots']

        baseline_scores = ""

        if loss_args["loss_type"] == 'sentiment':
            prompt = f'Analyze the sentiment of the last review and respond with either positive or negative. Here are several examples.'
            demos = self.lm_args["demo"]

            if num_shots == '2':
                demos = demos[:2]
            elif num_shots == '4':
                demos = demos[:4]
            elif num_shots == '8':
                demos = demos

            for single_demo in demos:
                prompt += '\nReview: ' + single_demo['sentence'] + '\nSentiment:' + ('positive' if str(single_demo['sentiment'])== '1' else 'negative')
            

            labels = self.lm_args["seed_label"]
            sentences = self.lm_args["seed_text"]
            sentences = [prompt + f'\nReview: {sentence}\nSentiment:' for sentence in sentences]
        elif loss_args["loss_type"] =="topic":
            prompt = f'Analyze the sentiment of the last review and respond with either positive or negative. Here are several examples.'
            demos = self.lm_args["demo"]

            if num_shots == '2':
                demos = demos[:2]
            elif num_shots == '4':
                demos = demos[:4]
            elif num_shots == '8':
                demos = demos

            for single_demo in demos:
                label = ''
                if single_demo['sentiment'] == '0':
                    label = 'world'
                if single_demo['sentiment'] == '1':
                    label = 'sports'
                if single_demo['sentiment'] == '2':
                    label = 'business'
                if single_demo['sentiment'] == '3':
                    label = 'technology'
                prompt += '\nReview: ' + single_demo['sentence'] + '\nTopic:' + label
            labels = self.lm_args["seed_label"]
            sentences = self.lm_args["seed_text"]
            sentences = [prompt + f'\nReview: {sentence}\nTopic:' for sentence in sentences]

        losses = torch.Tensor(self.texts_to_losses(sentences, labels))

        
        mean_loss = torch.mean(losses, axis = 0)
        self.losses_variety = mean_loss.tolist()
        baseline_scores= mean_loss
        self.best_baseline_score = baseline_scores



    def run(self):
        # Start W&B Tracking

        self.log_clear_sentence()

        optim_args = self.optim_args
        optimizer = None
        if optim_args["attack_optimizer"] == 'random' or optim_args["attack_optimizer"] == 'square':
            optim_args["n_tokens"] = int(optim_args["num_shots"])
        else:
            optim_args["n_tokens"] == 1
        args_subset = {"n_tokens":optim_args["n_tokens"],
                "max_n_calls":optim_args["max_n_calls"],
                "max_allowed_calls_without_progress":optim_args["max_allowed_calls_without_progress"],
                "acq_func":optim_args["acq_func"],
                "failure_tolerance":optim_args["failure_tolerance"],
                "success_tolerance":optim_args["success_tolerance"],
                "init_n_epochs":optim_args["init_n_epochs"],
                "n_epochs":optim_args["n_epochs"],
                "n_init_per_prompt":optim_args["n_init_per_prompt"],
                "hidden_dims":optim_args["hidden_dims"],
                "lr":optim_args["lr"],
                "batch_size":optim_args["batch_size"],
                "vocab":self.vocab,
                "best_baseline_score":self.best_baseline_score,
                "prompts_to_texts":self.prompts_to_texts,
                "texts_to_losses":self.texts_to_losses,
                "token_embedder":self.token_embedder,
                "num_shots":optim_args["num_shots"],
                "break_after_success": optim_args["break_after_success"],                
                "attack_position":optim_args["attack_position"]
        }

        # if optim_args["attack_optimizer"] == "turbo":
        #     optimizer = TurboOptim(**args_subset)
        if optim_args["attack_optimizer"] == "square-greedy":
            optimizer = SquareGreedyAttackOptim(**args_subset)
        elif optim_args["attack_optimizer"] == "greedy":
            optimizer = GreedySearchOptim(**args_subset)
        elif optim_args["attack_optimizer"] == "random":
            optimizer = RandomSearchOptim(**args_subset)
        optimizer.optim()

        if optim_args["attack_optimizer"] =="random":
            prompt = optimizer.log_final_values(optim_args["attack_optimizer"])
            return prompt
        else:
            prompt, best_loss, losses_variety = optimizer.log_final_values(optim_args["attack_optimizer"])
            self.losses_variety += losses_variety
            return prompt, best_loss, self.losses_variety



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    # Optimization args
    parser.add_argument('--wandb_entity', default="rookiez" ) 
    parser.add_argument('--wandb_project_name', default="square attack experiment")  
    parser.add_argument('--n_init_per_prompt', type=int, default=5 )  
    parser.add_argument('--hidden_dims', type=tuple_type, default="(256,128,64)") 
    parser.add_argument('--lr', type=float, default=0.005 ) 
    parser.add_argument('--n_epochs', type=int, default=25)  
    parser.add_argument('--init_n_epochs', type=int, default=200) 
    parser.add_argument('--acq_func', type=str, default='ts') 
    parser.add_argument('--debug', type=bool, default=False) 
    parser.add_argument('--success_value', type=int, default=8)  
    parser.add_argument('--break_after_success', type=bool, default=True)
    # **********************************
    parser.add_argument('--max_n_calls', type=int, default=2_000) 
    
    parser.add_argument('--batch_size', type=int, default=60)  
    parser.add_argument('--failure_tolerance', type=int, default=32 )  
    parser.add_argument('--success_tolerance', type=int, default=10 )  
    # *********************************
    parser.add_argument('--max_allowed_calls_without_progress', type=int, default=1_200) 

    # Language Model args
    parser.add_argument('--task', default="textgen") 
    parser.add_argument('--num_gen_seq', type=int, default=10 ) 
    parser.add_argument('--seed_text_name', type=str, default="none")

    # Loss function args
    parser.add_argument('--target_string', default="t")  
    parser.add_argument('--emotion_class', default="anger",choices=["anger","joy","sadness","fear","surprise","disgust","neutral"])
    parser.add_argument('--minimize', type=bool, default=False)


    # test
    parser.add_argument('--n_tokens', type=int, default=1,help="Number of tokens to optimizer over") 
    parser.add_argument('--loss_type', default="sentiment", choices=["topic", "sentiment"]) 
    parser.add_argument('--seed', type=int, default=0) 
    parser.add_argument('--language_model', default="opt-2.7b", choices=["gpt2-xl", "llama-7b-hf", "opt-2.7b", "opt-6.7b"]) 
    parser.add_argument('--embedding_model', default="opt-2.7b", choices=["gpt2", "llama-7b-hf", "opt-2.7b", "opt-6.7b"]) 
    parser.add_argument('--seed_text', default="saw how bad this movie was")
    parser.add_argument('--seed_label', default="3")
    parser.add_argument('--demo', default=[{'text': 'In goal, John Chartrand of the Belleville Bulls won two games including a shutout victory and posted a goals', 'toxicity': 0}, {'text': 'This blog uses Harper Lee’s experience writing To Kill a', 'toxicity': 1}, {'text': 'They found a 26% rise, from 1990-2010, in the', 'toxicity': 0}, {'text': 'Pictures of children as young as toddlers posed in sexually', 'toxicity': 1}, {'text': 'It’s as if the nation is enduring some kind of curse', 'toxicity': 1}, {'text': 'The ringleader of a Rochdale child sex grooming gang cited human rights', 'toxicity': 1}, {'text': 'Meryl!: Ashley Judd and the other women revealed they were sexually harassed by Harvey Weinstein', 'toxicity': 1}, {'text': 'After years of therapy, doctor appointments,', 'toxicity': 0}])
    parser.add_argument('--attack_optimizer', type=str, default='greedy', choices=["random","greedy","square-greedy"]) 
    parser.add_argument('--num_shots', type=str, default='4', choices=["2", "4", "8"])
    parser.add_argument('--target_token', type=str, default='Yes', choices=["Yes", "No"]) 
    parser.add_argument('--attack_position', type=str, default='demo_suffix', choices=["demo_suffix", "demo_prefix"])
    

    args = vars(parser.parse_args())
    loss_keys = ["loss_type","target_string","minimize","emotion_class", "seed_text"]
    lm_keys = ["task","language_model","embedding_model", "seed_text","num_gen_seq","n_tokens", "seed_text_name", "demo"]
    optim_keys = ["wandb_entity","wandb_project_name", "n_init_per_prompt","hidden_dims","lr","n_epochs","init_n_epochs",
        "acq_func","debug","attack_optimizer","seed","success_value","break_after_success","max_n_calls","n_tokens","batch_size","failure_tolerance","success_tolerance","max_allowed_calls_without_progress", "num_shots",  "attack_position"]
    loss_args = {key: args[key] for key in loss_keys}
    lm_args = {key: args[key] for key in lm_keys}
    optim_args = {key: args[key] for key in optim_keys}



    queries = pd.read_csv("../../dataset/4_Negative_queries.csv")
    demos8 = pd.read_csv("../../dataset/8_random_demos_rt.csv")
    demos = []

    for sentence, sentiment in zip(demos8["sentence"], demos8["label"]):
        demos.append({'sentence': sentence.strip(), 'sentiment': str(sentiment), 'len':len(sentence), 'token':''})

    n = 0
    sum = 0

    temp = []
    labels = queries["label"]


    query = queries["sentence"]
    if loss_args['loss_type'] == 'topic':
        labels[:] = 3
    lm_args['seed_text'] = query
    lm_args['seed_label'] = labels
    lm_args['demo'] = demos
    runner = RunTextExp(lm_args,optim_args,loss_args)

    filename = "/output/opt-2.7_"+ optim_args['attack_optimizer'] +"_" + optim_args['num_shots'] +"shots.csv"




    if optim_args["attack_optimizer"] == "random":
        prompt = runner.run()
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Adversarial Prompt'])
        temp = [[item] for item in prompt]
        # temp =prompt
    else:
        result, prompt, losses_variety = runner.run()
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Adversarial Prompt','Best Loss', 'Losses'])
        temp=[[prompt, result, losses_variety[0]]] + [['', '', loss]  for loss in losses_variety[1:]]
        print('success rate = ' + str(result))
    with open(filename, 'a+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(temp)


