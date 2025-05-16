import gc

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_attacks import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids_list, input_slice_list, target_slice_list, loss_slice_list, device):
    embed_weights = get_embedding_matrix(model)
    loss_list = []
    embeds_nonQuery = None
    last_stop = None
    adv_token_ids = None
    num_adv_tokens = input_slice_list[0].stop - input_slice_list[0].start

    for _control_slice in input_slice_list:
        if adv_token_ids == None:
            adv_token_ids = input_ids_list[0][_control_slice] 
        else: 
            adv_token_ids = torch.cat([ adv_token_ids, input_ids_list[0][_control_slice]]).to(device)
        

    for index1, input_ids in enumerate(input_ids_list):
        embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
        full_embeds = []

        if embeds_nonQuery == None:
            one_hot = torch.zeros(
                adv_token_ids.shape[0],
                embed_weights.shape[0],
                device=model.device,
                dtype=embed_weights.dtype
                )
            
            one_hot.scatter_(
                1, 
                adv_token_ids.unsqueeze(1),
                torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
                )
            one_hot.requires_grad_()
            input_embeds = (one_hot @ embed_weights).unsqueeze(0)

            for index, _control_slice in enumerate(input_slice_list):
                if last_stop == None:
                    # embeds[:,:_control_slice.start,:]
                    embeds_nonQuery = torch.cat(
                    [
                        embeds[:,:_control_slice.start,:], 
                        input_embeds[:, index*num_adv_tokens:index*num_adv_tokens + num_adv_tokens, :], 
                    ], 
                    dim=1)
                else:
                    embeds_nonQuery = torch.cat(
                    [
                        embeds_nonQuery,
                        embeds[:,last_stop:_control_slice.start,:],
                        input_embeds[:, index*num_adv_tokens:index*num_adv_tokens + num_adv_tokens, :], 
                    ], 
                    dim=1)
                last_stop = _control_slice.stop

        full_embeds = torch.cat(
            [
                embeds_nonQuery, 
                embeds[:,input_slice_list[-1].stop:,:]
            ], 
            dim=1)
        
        logits = model(inputs_embeds=full_embeds).logits
        targets = input_ids[target_slice_list[index1]]

        loss = nn.CrossEntropyLoss()(logits[0,loss_slice_list[index1],:], targets)
        loss_list.append(loss)
        del embeds, full_embeds, logits, targets, input_ids



    loss = sum(loss_list)/len(loss_list)
    # loss = max(loss_list)
    loss.backward()


    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    one_hot.grad = None
    del one_hot, input_embeds, embeds_nonQuery, adv_token_ids, loss_list, embed_weights, loss
    torch.cuda.empty_cache()

    return grad



def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty
    
    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks



def get_filtered_cands(tokenizer,filter_cand, control_cand, model_name, num_tokens):
    cands, count = [], 0
    for i in range(len(control_cand)):
        valid = True
        # decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if "gpt" in model_name or "opt" in model_name:
                for token in control_cand[i]:
                    if token.count(" ") != num_tokens:
                        valid = False
                if valid:
                    cands.append(control_cand[i])
                else:
                    count += 1
            elif "Llama" or 'vicuna' in model_name:
                for token in control_cand[i]:
                    token_ids = torch.tensor(tokenizer(token, add_special_tokens=False).input_ids)
                    token_ids_with_space = torch.tensor(tokenizer(' ' + token, add_special_tokens=False).input_ids)
                    if 'Llama-3' in model_name:
                        if token_ids.size()[0] != num_tokens or token_ids_with_space.size()[0] != num_tokens:
                            valid = False
                    else:      
                        if token_ids.size()[0] != num_tokens:
                            valid = False
                    # if token_ids.size()[0] != num_tokens or token_ids_with_space.size()[0] != num_tokens:
                    # if token_ids.size()[0] != num_tokens:
                        # valid = False
                if valid:
                    cands.append(control_cand[i])
                else:
                    count += 1
        else:
            cands.append(control_cand[i])

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands


def get_logits_in_batches(model, tokenizer, input_ids_list, control_slice_list, test_controls, batch_size, return_ids=True, num_adv_tokens=2, num_shots=None, target_slice= None):
    total_losses = []

    # Number of batches
    num_batches = (len(test_controls) + batch_size - 1) // batch_size


    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        # Extract the current batch of input_ids
        batch_test_controls = test_controls[start_idx:end_idx]

        # Call get_logits for this batch
        logits, ids = get_logits(
            model=model,
            tokenizer=tokenizer,
            input_ids_list=input_ids_list,
            control_slice_list=control_slice_list,
            test_controls=batch_test_controls,
            return_ids=return_ids,
            batch_size=batch_size,  # This may or may not be needed depending on how `get_logits` is implemented
            num_adv_tokens=num_adv_tokens,
            num_shots=num_shots
        )


        losses = target_loss(logits, ids, target_slice)
        for loss in losses:
            total_losses += [loss.item()]
        logits = None
        ids = None
        losses=None
        torch.cuda.empty_cache()

    return total_losses


def get_logits(*, model, tokenizer, input_ids_list, control_slice_list, test_controls=None, return_ids=False, batch_size=512, num_adv_tokens, num_shots):
    if isinstance(test_controls[0][0], str):
        max_len = num_adv_tokens * num_shots
        attn_mask_list = []

        test_ids = []
        for controls in test_controls:
            one_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in controls 
            ]
            # test_ids.append(torch.cat(one_ids, dim=0))
            concat_ids = torch.cat(one_ids, dim=0)
            concat_ids = concat_ids[:max_len]  # 🛠️ ensure no sequence exceeds max_len
            test_ids.append(concat_ids)

        pad_tok = 0
        while any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        for input_ids in input_ids_list:
            while pad_tok in input_ids:
                pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")
    
    if not(test_ids[0].shape[0] == max_len):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {max_len}), " 
            f"got {test_ids.shape}"
        ))
    
    dome_locs = [control_slice.start +i for control_slice in control_slice_list for i in range(num_adv_tokens)]
    dome_locs.sort()

    locs = torch.tensor(dome_locs).repeat(test_ids.shape[0], 1).to(model.device)
    ids_list = []
    for input_ids in input_ids_list:
        ids = torch.scatter(
            input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
            1,
            locs,
            test_ids
        )


        if pad_tok >= 0:
            attn_mask = (ids != pad_tok).type(ids.dtype)
            attn_mask_list.append(attn_mask)
        else:
            attn_mask = None
        ids_list.append(ids)
    if return_ids:
        del locs, test_ids ; gc.collect()
        # return 1, ids_list
        return forward(model=model, input_ids_list=ids_list, attention_mask_list=attn_mask_list, batch_size=batch_size), ids_list
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids_list=ids_list, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits



def forward(*, model, input_ids_list, attention_mask_list, batch_size=512):
    logits = []
    for index, input_ids in enumerate(input_ids_list):
        for i in range(0, input_ids.shape[0], batch_size):
            batch_input_ids = input_ids[i:i+batch_size]
            attention_mask = attention_mask_list[index]
            if attention_mask is not None:
                batch_attention_mask = attention_mask[i:i+batch_size]
            else:
                batch_attention_mask = None
            with torch.no_grad():
                logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)
            gc.collect()

        del batch_input_ids, batch_attention_mask
    
    return logits



def target_loss(logits_list, ids, target_slice):
    loss_list = []
    for index, logits in enumerate(logits_list):
        crit = nn.CrossEntropyLoss(reduction='none')
        loss_slice = slice(target_slice[index].start-1, target_slice[index].stop-1)
        loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[index][:,target_slice[index]])
        loss_list.append(loss.mean(dim=-1))
    loss = sum(loss_list)/len(loss_list) 
    return loss

def load_model_and_tokenizer(model_name, tokenizer_path=None, device='cuda:0',  **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **kwargs
        ).to(device).eval()
    
    tokenizer_path = model_name if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'gpt2-xl' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'

    return model, tokenizer