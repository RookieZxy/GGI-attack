from gptlm import GPT2LM
import torch
import argparse
import pandas as pd



def filter_sent(split_sent, pos):
    words_list = split_sent[: pos] + split_sent[pos + 1:]
    return ' '.join(words_list)

def get_PPL(data):
    all_PPL = []
    from tqdm import tqdm
    for i, sent in enumerate(tqdm(data)):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)
        single_sent_PPL = []
        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)
            single_sent_PPL.append(LM(processed_sent))
        all_PPL.append(single_sent_PPL)

    assert len(all_PPL) == len(data)
    return all_PPL

def get_processed_sent(flag_li, orig_sent):
    sent = []
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
    return ' '.join(sent)

def get_processed_data(all_clean_PPL, clean_data, bar):
    processed_data = []
    data = [clean_data]
    for i, PPL_li in enumerate(all_clean_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1
        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)
        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        processed_data.append((sent, clean_data))
    assert len(all_clean_PPL) == len(processed_data)
    return processed_data[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_name = "opt-6.7b"
    parser.add_argument('--file_path_demos', default='../dataset/sentiment/8_random_demos_rt.csv')
    parser.add_argument('--save_path', default=f'../dataset/filtered_demos/{model_name}.csv')
    parser.add_argument('--adv_prompts', default=['Filter', 'Fresh', 'hated', 'And'], help='replace the token list with the one you learned')

    args = parser.parse_args()

    LM = GPT2LM(use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    demos_list = pd.read_csv(args.file_path_demos)
    instruction = "Analyze the sentiment of the last review and respond with either positive or negative. Here are several examples."
    input_sentence = instruction

    for demo_sentence, demo_label, prompt in zip(demos_list['sentence'], demos_list['label'], args.adv_prompts):
        input_sentence += "\nReview: " + str(demo_sentence).strip() + " " +  prompt + "\nSentiment:" + ("positive" if demo_label == 1 else 'negative')


    all_clean_PPL = get_PPL([input_sentence])
    for bar in range(-1, 0):
        processed_clean_loader = get_processed_data(all_clean_PPL, input_sentence, bar)
        print(processed_clean_loader[0])
        print(processed_clean_loader[1])
        print('bar: ', bar)
        print('*' * 89)
        df = pd.DataFrame({
            'prompt': [processed_clean_loader[0]]
        })
        # Save it to CSV
        df.to_csv(args.save_path, index=False)