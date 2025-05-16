import sys 
import copy
sys.path.append("../")
from utils.objective import Objective 
from utils.text_utils.text_losses import *
from utils.token_embedder import OPTEmbedding
from utils.text_utils.language_model import *

class TextGenerationObjective(Objective):
    def __init__(
        self,
        n_tokens,
        prompts_to_texts,
        texts_to_losses,
        token_embedder,
        lb = None,
        ub = None,
        **kwargs,
    ):

        self.prompts_to_texts = prompts_to_texts
        self.texts_to_losses = texts_to_losses
        self.token_embedder = token_embedder

        self.n_tokens = n_tokens
        super().__init__(
            num_calls = 0,
            task_id = 'adversarial4',
            dim = n_tokens * self.token_embedder.embed_dim,
            lb = lb,
            ub = ub,
            **kwargs,
        ) 

    def query_oracle(self, embeddings, token_position, attack_position):
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings, dtype=torch.float16)
        embeddings = embeddings.cuda(0) 
        embeddings = embeddings.reshape(-1, self.n_tokens, self.token_embedder.embed_dim) 
        prompts = self.token_embedder.batched_embeddings_to_tokens(embeddings)
        if attack_position == 'demo_suffix' or attack_position == 'demo_prefix':
            generated_text_list, token_list, labels_list = self.prompts_to_texts(prompts, token_position, attack_position)

        num_shots = len(token_position)
        losses_list = []
        for generated_text, labels in zip(generated_text_list, labels_list):
            losses = torch.Tensor(self.texts_to_losses(generated_text, labels))
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
        losses = torch.Tensor(temp_losses)
        token_list = temp_token_list
        # mean_loss = torch.mean(losses, axis = 0)
        # mean_loss = torch.mean(losses, axis = 1)
        # mean_loss = losses
        # return prompts, mean_loss, generated_text, token_list
        # return prompts, mean_loss, token_list
        return prompts, losses, token_list

