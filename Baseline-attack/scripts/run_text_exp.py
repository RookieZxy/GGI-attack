import os
os.environ["WANDB_SILENT"] = "False"
import random
from fastchat.conversation import register_conv_template, Conversation, SeparatorStyle, get_conv_template
register_conv_template(
    Conversation(
        name="llama-3",
        system_template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("<|start_header_id|>user<|end_header_id|>\n",
            "<|start_header_id|>assistant<|end_header_id|>\n"),
        system_message = "",
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="<|eot_id|>",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import copy
from utils.text_utils.language_model import HuggingFaceLanguageModel
from utils.text_utils.text_losses import  TopicLoss, SentimentLoss, JailbreakingLoss
from run_optimization import RandomSearchOptim, GreedySearchOptim, SquareGreedyAttackOptim
from utils.constants import tuple_type, trimmed_mean
from utils.token_embedder import OPTEmbedding, LlamaEmbedding, GPTEmbedding, Llama3_1Embedding, Llama2Embedding, VicunaEmbedding, MistralEmbedding

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
        if language_model_name in ["gpt2-xl", "llama-7b-hf", "opt-2.7b", "opt-6.7b", "Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-2-7b-chat-hf", 'vicuna-7b-v1.5', "Mistral-7B-Instruct-v0.3"]:
            lm = HuggingFaceLanguageModel(max_num_tokens, num_gen_seq)
            lm.load_model(language_model_name)
        else:
            raise NotImplementedError
        return lm

    def get_loss_fn(self, loss_args, model_name, label):
        loss_type = loss_args["loss_type"]
        if loss_type == "sentiment":
            return SentimentLoss(model_name, label, optim_args["attack_position"], optim_args["attack_optimizer"])
        elif loss_type == "topic":
            return TopicLoss(model_name, label, optim_args["attack_position"], optim_args["attack_optimizer"])
        elif loss_type == "jailbreaking":
            return JailbreakingLoss(model_name, label, optim_args["attack_position"], optim_args["attack_optimizer"])
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
                            if 'Llama-3' in lm_args["language_model"]:
                                pad = ' '
                            else:
                                pad = ''
                            input += '\nReview: ' + single_demo['sentence']  + " " + single_token  + '\nSentiment:' + pad + ('positive' if str(single_demo['label'])== '1' else 'negative')

                    sentences = [input + f'\nReview: {sentence}\nSentiment:' +pad for sentence in self.lm_args["seed_text"]]
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
                                if 'Llama-3' in lm_args["language_model"]:
                                    pad = ' '
                                else:
                                    pad = ''
                                if single_token == "" :
                                    input += '\nReview: ' + single_demo['sentence'] +'\nSentiment:' +pad+ ('positive' if str(single_demo['label'])== '1' else 'negative')
                                else:
                                    input += '\nReview: ' + single_demo['sentence'] + " " + single_token +pad + '\nSentiment:' + ('positive' if str(single_demo['label'])== '1' else 'negative')

                        sentences = [input + f'\nReview: {sentence}\nSentiment:' +pad for sentence in self.lm_args["seed_text"]]

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
                            if single_demo['label'] == '0':
                                label = 'world'
                            if single_demo['label'] == '1':
                                label = 'sports'
                            if single_demo['label'] == '2':
                                label = 'business'
                            if single_demo['label'] == '3':
                                label = 'technology'
                            input += '\nReview: ' + single_demo['sentence'] + " "  + single_token  + '\nTopic:' + label

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
                                if single_demo['label'] == '0':
                                    label = 'world'
                                if single_demo['label'] == '1':
                                    label = 'sports'
                                if single_demo['label'] == '2':
                                    label = 'business'
                                if single_demo['label'] == '3':
                                    label = 'technology'

                                if single_token == "" :
                                    input += '\nReview: ' + single_demo['sentence'] +'\nTopic:' + label
                                else:
                                    input += '\nReview: ' + single_demo['sentence'] + " " + single_token  + '\nTopic:' + label

                        sentences = [input + f'\nReview: {sentence}\nTopic:' for sentence in self.lm_args["seed_text"]]

                        sentences_list.append(sentences)
                        labels_list.append(self.lm_args["seed_label"].tolist())

                        token_list.append(replaced_token)

        elif loss_args["loss_type"] == "jailbreaking":
            conv_template = get_conv_template(self.optim_args["template_name"])
            if self.optim_args["attack_optimizer"] == 'random' or self.optim_args["attack_optimizer"] == 'square':
                for prompt in prompts:
                    input = ''
                    for (single_demo, single_token) in zip(demos, prompt):
                        input += 'Question: ' + single_demo['sentence'] + '\nSure, '  + single_token  + " " + single_demo['label'][:50] +'\n'
                    for sentence in self.lm_args["seed_text"]:
                        conv_template.append_message(conv_template.roles[0], f"{input}{sentence}")
                        conv_template.append_message(conv_template.roles[1], f"Sure")
                        sentences_list.append(conv_template.get_prompt())
                        labels_list.append(self.lm_args["seed_label"].tolist())
                        token_list.append(prompt)
                        print(conv_template.get_prompt())
                        conv_template.messages = []
            else:
                for prompt in prompts:
                    for idx in range(int(self.optim_args['num_shots'])):
                        conv_template.messages = []
                        input = ''
                        replaced_token = copy.deepcopy(token_position)
                        replaced_token[idx] = prompt

                        for (single_demo, single_token) in zip(demos, replaced_token):
                            conv_template.append_message(conv_template.roles[0], f"{single_demo['sentence']}")

                            if single_token == "" :
                                # input += 'Question: ' + single_demo['sentence'] + '\nSure, '  + single_demo['label'][:50] +'\n'
                                conv_template.append_message(conv_template.roles[1], f"Sure,{single_demo['label']}")
                            else:
                                # input += 'Question: ' + single_demo['sentence'] + '\nSure, '  + single_token  + " " + single_demo['label'][:50] +'\n'
                                conv_template.append_message(conv_template.roles[1], f"Sure, {single_token}{single_demo['label']}")


                        sentences =[]
                        for sentence in self.lm_args["seed_text"]:
                            temp_conv_template = conv_template.copy()
                            temp_conv_template.append_message(temp_conv_template.roles[0], f"{sentence}")
                            temp_conv_template.append_message(temp_conv_template.roles[1], f"Sure, here")
                            if 'vicuna' in self.lm_args['language_model'] or 'Mistral' in self.lm_args['language_model']:
                                sentences.append(temp_conv_template.get_prompt()[:-8])
                            else:   
                                sentences.append(temp_conv_template.get_prompt())
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
        elif 'Llama-3.1-8B' in embedding_model:
            return Llama3_1Embedding(language_model_vocab_tokens)
        elif 'Llama-2-7b-chat-hf' in embedding_model:
            return Llama2Embedding(language_model_vocab_tokens)
        elif 'vicuna-7b-v1.5' in embedding_model:
            return VicunaEmbedding(language_model_vocab_tokens)
        elif "Mistral-7B-Instruct-v0.3" in embedding_model:
            return MistralEmbedding(language_model_vocab_tokens)


        
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
        sentences_list = []
        demos = self.lm_args["demo"]
        if num_shots == '2':
            demos = demos[:2]
        elif num_shots == '4':
            demos = demos[:4]
        elif num_shots == '8':
            demos = demos

        if loss_args["loss_type"] == 'sentiment':
            prompt = f'Analyze the sentiment of the last review and respond with either positive or negative. Here are several examples.'

            for single_demo in demos:
                prompt += '\nReview: ' + single_demo['sentence'] + '\nSentiment:' + ('positive' if str(single_demo['label'])== '1' else 'negative')       

            labels = self.lm_args["seed_label"]
            sentences = self.lm_args["seed_text"]
            sentences_list = [prompt + f'\nReview: {sentence}\nSentiment:' for sentence in sentences]
        elif loss_args["loss_type"] =="topic":
            prompt = f'Classify the topic of the last review. Here are several examples.'

            for single_demo in demos:
                label = ''
                if single_demo['label'] == '0':
                    label = 'world'
                if single_demo['label'] == '1':
                    label = 'sports'
                if single_demo['label'] == '2':
                    label = 'business'
                if single_demo['label'] == '3':
                    label = 'technology'
                prompt += '\nReview: ' + single_demo['sentence'] + '\nTopic:' + label
            labels = self.lm_args["seed_label"]
            sentences = self.lm_args["seed_text"]
            sentences = [prompt + f'\nReview: {sentence}\nTopic:' for sentence in sentences]
        elif loss_args["loss_type"] == "jailbreaking":
            conv_template = get_conv_template(self.optim_args["template_name"])
            sentences_list=[]

            for single_demo in demos:
                conv_template.append_message(conv_template.roles[0], f"{single_demo['sentence']}")
                conv_template.append_message(conv_template.roles[1], f"Sure,{single_demo['label']}")

          
            for sentence in self.lm_args["seed_text"]:
                temp_conv_template = conv_template.copy()
                temp_conv_template.append_message(temp_conv_template.roles[0], f"{sentence}")
                temp_conv_template.append_message(temp_conv_template.roles[1], f"Sure, here")
                if 'vicuna' in self.lm_args['language_model'] or 'Mistral' in self.lm_args['language_model']:
                    sentences_list.append(temp_conv_template.get_prompt()[:-8])
                else:   
                    sentences_list.append(temp_conv_template.get_prompt())
            labels = self.lm_args["seed_label"]

        losses = torch.Tensor(self.texts_to_losses(sentences_list, labels))
        
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
            return prompt



    
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
    parser.add_argument('--num_gen_seq', type=int, default=10 ) 
    parser.add_argument('--seed_text_name', type=str, default="none")

    # Loss function args
    parser.add_argument('--target_string', default="t")  
    parser.add_argument('--emotion_class', default="anger",choices=["anger","joy","sadness","fear","surprise","disgust","neutral"])
    parser.add_argument('--minimize', type=bool, default=False)


    # test
    parser.add_argument('--n_tokens', type=int, default=3,help="Number of tokens to optimizer over") 
    parser.add_argument('--loss_type', default="jailbreaking", choices=["topic", "sentiment", "jailbreaking"]) 
    parser.add_argument('--seed', type=int, default=0) 
    parser.add_argument('--language_model', default="Mistral-7B-Instruct-v0.3", choices=["gpt2-xl", "llama-7b-hf", "opt-2.7b", "opt-6.7b", "Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-2-7b-chat-hf", "vicuna-7b-v1.5", "Mistral-7B-Instruct-v0.3"]) 
    parser.add_argument('--embedding_model', default="Mistral-7B-Instruct-v0.3", choices=["gpt2", "llama-7b-hf", "opt-2.7b", "opt-6.7b", "Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-2-7b-chat-hf", "vicuna-7b-v1.5", "Mistral-7B-Instruct-v0.3"]) 
    parser.add_argument('--seed_text', default="saw how bad this movie was")
    parser.add_argument('--seed_label', default="3")
    parser.add_argument('--demo', default=[])
    parser.add_argument('--attack_optimizer', type=str, default='square-greedy', choices=["random","greedy","square-greedy"]) 
    parser.add_argument('--template_name', type=str, default="llama-2", choices=["llama-2", "llama-3", "vicuna_v1.1"]) 
    parser.add_argument('--num_shots', type=str, default='2', choices=["2", "4", "8"])
    parser.add_argument('--target_token', type=str, default='Yes', choices=["Yes", "No"]) 
    parser.add_argument('--attack_position', type=str, default='demo_suffix', choices=["demo_suffix"])
    parser.add_argument('--demos_path', type=str, default='/home/hp6438/desktop/code/GGI-attack-main/dataset/jailbreaking/4_harmful_demos_new.csv') 
    parser.add_argument('--queries_path', type=str, default='/home/hp6438/desktop/code/GGI-attack-main/dataset/jailbreaking/6_harmful_queries.csv')


    args = vars(parser.parse_args())
    loss_keys = ["loss_type","target_string","minimize","emotion_class", "seed_text"]
    lm_keys = ["language_model","embedding_model", "seed_text","num_gen_seq","n_tokens", "seed_text_name", "demo"]
    optim_keys = ["wandb_entity","wandb_project_name", "n_init_per_prompt","hidden_dims","lr","n_epochs","init_n_epochs",
        "acq_func","debug","attack_optimizer","seed","success_value","break_after_success","max_n_calls","n_tokens","batch_size","failure_tolerance","success_tolerance","max_allowed_calls_without_progress", "num_shots",  "attack_position", "template_name"]
    loss_args = {key: args[key] for key in loss_keys}
    lm_args = {key: args[key] for key in lm_keys}
    optim_args = {key: args[key] for key in optim_keys}


    demos8 = pd.read_csv(args['demos_path'])
    queries = pd.read_csv(args['queries_path'])[:2]


    demos = []

    for sentence, label in zip(demos8["sentence"], demos8["label"]):
        if loss_args['loss_type'] == 'jailbreaking':
            secondpart = label.split(' ')
            label = ' '+' '.join(secondpart[1:45])+'\n'
        demos.append({'sentence': sentence.strip(), 'label': str(label), 'len':len(sentence), 'token':''})
    n = 0
    sum = 0

    temp = []
    labels = queries["label"]


    query = queries["sentence"]
    if loss_args['loss_type'] == 'topic':
        labels[:] = 3
    elif loss_args['loss_type'] == 'jailbreaking':
        labels[:] = "Sure, here"

    lm_args['seed_text'] = query
    lm_args['seed_label'] = labels
    lm_args['demo'] = demos

    runner = RunTextExp(lm_args,optim_args,loss_args)

    prompt = runner.run()
    print(f'Best prompt = {prompt}')


