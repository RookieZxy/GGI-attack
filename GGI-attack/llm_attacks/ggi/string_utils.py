import torch
import fastchat 
import copy
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
def load_conversation_template(template_name):
    # conv_template = fastchat.model.get_conversation_template(template_name)

    conv_template = get_conv_template(template_name)
    conv_template.set_system_message("")
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()

    return conv_template


def split_string(original_string, adv_token_pos):
    words = original_string.split()
    assert len(words) > adv_token_pos, "adv_token_pos is out of the len of demos"

    first_two_words = ' '.join(words[:adv_token_pos])
    remaining_words = ' '.join(words[adv_token_pos:])

    return first_two_words, remaining_words



class SuffixManager:
    def __init__(self, *, model_name, tokenizer, demos, queries, instruction, targets, adv_prompts, att_position, num_adv_tokens, adv_token_pos):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.demos = demos
        self.queries = queries
        self.instruction = instruction
        self.targets = targets
        self.adv_prompts = adv_prompts
        self.att_position = att_position
        self.num_adv_tokens = num_adv_tokens
        self.adv_token_pos = adv_token_pos
        self._demos_slice = [[] for i in range(len(demos))]
        self._control_slice = [[] for i in range(len(demos))]
        self._demos_label_slice = [[] for i in range(len(demos))]
        self._queries_slice = [[] for i in range(len(queries))]
        self._target_slice = [[] for i in range(len(queries))]
        self._loss_slice = [[] for i in range(len(queries))]
    
    def get_prompt(self, adv_prompts=None):
        if adv_prompts is not None:
            self.adv_prompts = adv_prompts
        
        prompts = []
        input = ""
        if self.att_position == 2 and len(self.demos[0]['sentence'][0]) == 1:
            temp = []
            for index, (demo) in enumerate(self.demos):
                substring1, substring2 = split_string(demo['sentence'], self.adv_token_pos)
                temp.append({'sentence': [("\nReview:" + substring1), " " + substring2], 'label': demo['label']})
            self.demos = temp

        for i, j  in zip(self.queries ,self.targets):
            if self.att_position == 0:
                prompts.append(self.instruction  + ''.join([ "\nReview:" + y + x['sentence'] + x['label'] for x, y in zip(self.demos, self.adv_prompts)]) +  i +j)

            elif self.att_position == 1:
                prompts.append(self.instruction  + ''.join([x['sentence'] + y + x['label'] for x, y in zip(self.demos, self.adv_prompts)]) +  i +j)

            elif self.att_position == 2:
                prompts.append(self.instruction  + ''.join([ x['sentence'][0] + y + x['sentence'][1] +  x['label'] for x, y in zip(self.demos, self.adv_prompts)]) +  i +j)


        input += self.instruction
        toks = self.tokenizer(input).input_ids
        self._instruction_slice = slice(None, len(toks))
        # demos and labels position
        if self.att_position == 0:
            for index, (demo, adv_prompt) in enumerate(zip(self.demos, self.adv_prompts)):
                input += "\nReview:"
                toks = self.tokenizer(input).input_ids
                temp_len = len(toks)

                # adv_prompt position
                input += adv_prompt
                toks = self.tokenizer(input).input_ids
                if temp_len == len(toks):
                    self._control_slice[index] = slice(temp_len-1, len(toks))
                else:
                    self._control_slice[index] = slice(temp_len, len(toks))

                # demo position
                input += demo['sentence']
                toks = self.tokenizer(input).input_ids
                self._demos_slice[index] = slice(self._instruction_slice.stop, len(toks))

                # demo label position
                input += demo['label']
                toks = self.tokenizer(input).input_ids
                self._demos_label_slice[index] = slice(self._demos_slice[index].stop, len(toks))
        elif self.att_position == 1:
            for index, (demo, adv_prompt) in enumerate(zip(self.demos, self.adv_prompts)):
                input += demo['sentence']
                toks = self.tokenizer(input).input_ids
                self._demos_slice[index] = slice(self._instruction_slice.stop, len(toks))
     
                input += adv_prompt
                toks = self.tokenizer(input).input_ids


                if self._demos_slice[index].stop + self.num_adv_tokens != len(toks):
                    self._control_slice[index] = slice(self._demos_slice[index].stop-1, len(toks))
                else:
                    self._control_slice[index] = slice(self._demos_slice[index].stop, len(toks))
                input += demo['label']
                toks = self.tokenizer(input).input_ids
                self._demos_label_slice[index] = slice(self._control_slice[index].stop, len(toks))
        elif self.att_position == 2:
            for index, (demo, adv_prompt) in enumerate(zip(self.demos, self.adv_prompts)):
                input += demo['sentence'][0]
                toks = self.tokenizer(input).input_ids
                self._demos_slice[index] = slice(self._instruction_slice.stop, len(toks))

                input += adv_prompt
                toks = self.tokenizer(input).input_ids
       

                if self._demos_slice[index].stop + self.num_adv_tokens != len(toks):
                    self._control_slice[index] = slice(self._demos_slice[index].stop-1, len(toks))
                else:
                    self._control_slice[index] = slice(self._demos_slice[index].stop, len(toks))
                input += demo['sentence'][1]
                input += demo['label']
                toks = self.tokenizer(input).input_ids
                self._demos_label_slice[index] = slice(self._control_slice[index].stop, len(toks))

        # query position
        for index, (query, target) in enumerate(zip(self.queries, self.targets)):
            temp_input = copy.deepcopy(input)
            temp_input += query
            toks = self.tokenizer(temp_input).input_ids
            self._queries_slice[index] = slice(self._demos_label_slice[-1].stop, len(toks))
            temp_input += target
            toks = self.tokenizer(temp_input).input_ids
            self._target_slice[index] = slice(self._queries_slice[index].stop, len(toks))
            self._loss_slice[index] = slice(self._queries_slice[index].stop-1, len(toks)-1)

        return prompts

    
    def get_input_ids(self, adv_prompts=None):
        # return input_ids
        prompt_list = self.get_prompt(adv_prompts=adv_prompts)
        input_ids_list = []
        for prompt, target_slice in zip(prompt_list, self._target_slice):
            toks = self.tokenizer(prompt).input_ids

            input_ids_list.append(torch.tensor(toks[:target_slice.stop]))
        return input_ids_list
        