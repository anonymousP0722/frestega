"""Main module."""
import os
import sys
import jsonlines
import math
import torch
from tqdm.autonotebook import tqdm
from baselines.encode import encoder
from baselines.encode import DRBG
from baselines.decode import decoder
import random
import numpy as np
import copy
import time 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


import argparse
import logging
from configparser import ConfigParser
import json

provably_secure_alg = ['meteor','group','group_b','group_s','discop','discop_base']
group_alg = ["group","group_b","group_s"]
discop_alg = ['discop','discop_base']
meteor_alg = ['meteor']





def prompt_template(prompt_text, model, tokenizer, mode = 'generate', role = 'user'):
    if mode == 'generate':
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
        return input_ids
    elif mode == 'chat':
        ######prompt_construct######
        messages = [{"role": role, "content": prompt_text},]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        return tokenized_chat
    else:
        raise ValueError("no such mode")


def calculate_entropy(prob):
    # 选取概率大于0的元素进行熵计算
    mask = prob > 0
    prob_nonzero = prob[mask]
    log_prob_nonzero = torch.log2(prob_nonzero)
    entropy = -torch.sum(prob_nonzero * log_prob_nonzero)
    return entropy.item()

def calculate_capacity(prob):
    mask = prob > 0
    prob = prob[mask]
    
    p, indices = torch.sort(prob, descending=True)
    n = torch.arange(1, len(p) + 1).to('cuda')
    formula = torch.sum(p * (n * torch.log2(n) - (n - 1) * torch.log2(torch.maximum(n - 1, torch.tensor(1)))))
    return formula.item()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def embed_single(model, tokenizer, bit_stream, 
                 prompt_text=None,
                 mode = "generate",
                 role = "user",
                 max_new_tokens=256, 
                 alg="ac",
                 mask_generator = None, 
                 kwargs = None,
                 topk = None,
                 frequency = None,
                model_frequency = None):
    start_time = time.time()

    bit_index = 0

    with torch.no_grad():
        
        input_ids = prompt_template(prompt_text, model, tokenizer, mode = mode, role = role)

        x = input_ids
        stega_sentence = []
        stega_bit = []
        stega_bit_num = []
        entropy = []
        initial_entropy = []
        temperature = []
        after_space_entropy=[]
        after_time_entropy = []
        capacity_test = []

        if alg.lower() in ["ac","rolro"]:
            max_val = 2 ** kwargs["precision"]  # num of intervals ; max_val = 2**52
            cur_interval = [0, max_val]
            seed=42

        past_key_values = None
        for i in range(max_new_tokens):

            if tokenizer.eos_token_id in stega_sentence:
                break

            # conditional probability distribution
            output = model(input_ids=x, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            log_prob = output.logits[:, -1, :]
            log_prob -= log_prob.max()
            prob = torch.exp(log_prob).reshape(-1)

            #开始fremax
            cal_prob=prob/prob.sum()
            
            initial_entropy.append(calculate_entropy(cal_prob))
            
            
            #先时间
            t = (args.baseT + args.theta * np.log(1 + args.c * calculate_entropy(cal_prob)))
            temperature.append(t)
            
            recovered_log_prob = torch.log(prob)
            recovered_log_prob = recovered_log_prob/t
            prob = torch.exp(recovered_log_prob).reshape(-1)
            
            after_time_prob = prob / prob.sum()
            
            

            after_time_entropy.append(calculate_entropy(after_time_prob))
            
            

             #再空间
            tmp_frequency = frequency.copy()
            tmp_model_frequency = model_frequency.copy()

            tmp_frequency= 1+tmp_frequency
            tmp_model_frequency = 1+tmp_model_frequency

            tmp_frequency = tmp_frequency / np.sum(tmp_frequency)
            tmp_model_frequency = tmp_model_frequency / np.sum(tmp_model_frequency)

            # # 重复惩罚
            for i in x[0]:
                tmp_frequency[i] = math.sqrt(tmp_frequency[i])
            
            prob = prob * (torch.log(args.beta + torch.pow(torch.Tensor((tmp_frequency)/(tmp_model_frequency)).to('cuda') ,args.alpha)))

            temp_cal_prob=prob/prob.sum()
            
        
        
        
        
        
            # time_entropy = -(temp_cal_prob * torch.log(temp_cal_prob)).sum()
            ttttemp_entropy=calculate_entropy(temp_cal_prob)
            after_space_entropy.append(ttttemp_entropy)
            # t = (args.baseT + args.theta * np.log(1 + args.c * time_entropy))
            # temperature.append(t)
            # # t =  (args.baseT + args.theta * torch.log(1 + time_entropy))
            # # final_prob = torch.exp(keep_log_prob / t).reshape(-1)
            # recovered_log_prob = torch.log(prob)
            # recovered_log_prob = recovered_log_prob/t
            # prob = torch.exp(recovered_log_prob).reshape(-1)


            final_prob = prob / prob.sum()
            prob = final_prob
            

            # after_time_entropy.append(calculate_entropy(final_prob))
            capacity_test.append(calculate_capacity(final_prob))


            # prob = prob / prob.sum()
            
            ######logits 预处理#######TODO
            prob, indices = prob.sort(descending=True)
            mask = prob > 0
            prob=prob[mask]
            indices = indices[mask]   
            prob = prob[:topk]
            indices = indices[:topk]         
            prob = prob / prob.sum()
            ########################

            ### prev and encode
            if alg.lower() in ["ac","rolro"]:
                cur_interval_pre = copy.deepcopy(cur_interval)
                seed+=1
                cur_interval, prev, num_bits_encoded = encoder(alg, prob, indices, bit_stream, bit_index, cur_interval,seed=seed, **kwargs)
            elif alg.lower() in provably_secure_alg:
                prev, num_bits_encoded = encoder(alg, prob, indices, bit_stream, bit_index,mask_generator = mask_generator, **kwargs)
            else:
                prev, num_bits_encoded = encoder(alg, prob, indices, bit_stream, bit_index, **kwargs)

            if int(prev) == tokenizer.eos_token_id:
                break
            entropy.append(calculate_entropy(prob))
            capacity_test.append(calculate_capacity(prob))

            stega_sentence.append(int(prev))
            x = prev
            stega_bit.append(bit_stream[bit_index:bit_index + num_bits_encoded])
            stega_bit_num.append(num_bits_encoded)
            bit_index += num_bits_encoded
    if tokenizer.eos_token_id in stega_sentence:
        stega_sentence.remove(tokenizer.eos_token_id)

    stega_text = tokenizer.decode(stega_sentence)

    cost_time = time.time()-start_time

    # print(mask_generator.test)
    return stega_sentence, stega_text, bit_index, stega_bit, stega_bit_num, cost_time, initial_entropy, after_space_entropy,after_time_entropy, temperature


def embed(model, tokenizer, bit_stream, output_filepath, 
          prompts=None, 
          mode = "generate",
          role = "user",
          sentence_num=2000,
          max_new_tokens=256, 
          alg="ac",
          kwargs = None,
          topk = None):

    with open(args.target_corpus,"r") as f:
        dic = json.load(f)
    frequency = []
    for i in range(len(dic)):
        frequency.append(dic[str(i)])
    frequency = np.array(frequency)
 
    with open(args.model_corpus,"r") as f:
        dic = json.load(f)
    model_frequency = []
    for i in range(len(dic)):
        model_frequency.append(dic[str(i)])
    model_frequency = np.array(model_frequency)

    mask_generator = None

    if alg in provably_secure_alg:
        input_key = bytes.fromhex(kwargs["input_key"])
        sample_seed_prefix = bytes.fromhex(kwargs["sample_seed_prefix"])
        input_nonce = bytes.fromhex(kwargs["input_nonce"])
        mask_generator = DRBG(input_key, sample_seed_prefix + input_nonce)


    with jsonlines.open(output_filepath, "w") as f_out:
        bit_index = 0
        for stega_idx in tqdm(range(sentence_num), desc=f"generate stegos {alg} into {output_filepath}"):
            while len(bit_stream[bit_index:]) <= max_new_tokens * math.log2(tokenizer.vocab_size):
                bit_stream_shuffle = np.random.randint(high=2,low=0, size=(1, 100000)).tolist()[0]
                random.shuffle(bit_stream_shuffle)
                bit_stream += "".join([str(b) for b in bit_stream_shuffle]) # add more bits
            # try:
            prompt_text = prompts[stega_idx]
            # while True:
            stega_tokens,  stega_text, num_bits_encoded,stega_bits, stega_bits_num, cost_time, initial_entropy ,after_space_entropy, after_time_entropy, temperature = embed_single(model, tokenizer, bit_stream[bit_index:], 
                                                                                                                                    prompt_text=prompt_text, 
                                                                                                                                    mode = mode,
                                                                                                                                    role = role,
                                                                                                                                    alg=alg, 
                                                                                                                                    max_new_tokens=max_new_tokens,
                                                                                                                                    mask_generator = mask_generator, 
                                                                                                                                    kwargs=kwargs,
                                                                                                                                    topk = topk,
                                                                                                                                    frequency = frequency,
                                                                                                                                        model_frequency = model_frequency)
            stega_bit = bit_stream[bit_index:bit_index+num_bits_encoded]
            bit_index += num_bits_encoded
                # print(bit_index)
                # print(len(stega_tokens))
                # print(num_bits_encoded)

                # if len(stega_tokens) > 0 and (num_bits_encoded>0 or alg == 'plain'):
                #     break
                # else:
                #     bit_index += 1
                #     print('error')

            if alg in provably_secure_alg:
                if alg in meteor_alg:
                    # print(mask_generator.test)
                    for _ in range((max_new_tokens+10)*(stega_idx+1)-mask_generator.test):
                        mask_generator.generate_bits(kwargs["precision"])
                    # print(mask_generator.test)
                elif alg in group_alg:
                    for _ in range((2*max_new_tokens+10)*(stega_idx+1)-mask_generator.test):
                        mask_generator.generate_bits(kwargs["precision"])    
                elif alg in discop_alg: 
                    # print(mask_generator.test)
                    for _ in range((math.ceil(math.log2(tokenizer.vocab_size)) * max_new_tokens+10)*(stega_idx+1)-mask_generator.test):
                        mask_generator.generate_bits(kwargs["precision"])       
                    # print(mask_generator.test)                  

            
            f_out.write({"idx": stega_idx,
                        "prompt": prompt_text,
                        "stego": stega_text,
                        "tokens": stega_tokens,
                        "bits": stega_bit,
                        "token_bits":stega_bits_num,
                        "time":cost_time,
                        "initial_entropy":initial_entropy,
                        "after_space_entropy":after_space_entropy,
                        "after_time_entropy":after_time_entropy,
                        "temperature":temperature
                        })


def extract_single(model, tokenizer, stego_text, 
                   prompt_text = None,
                   mode = "generate",
                   role = "user",
                   max_new_tokens = 256, 
                   alg="ac", 
                   mask_generator = None,
                   kwargs = None,
                   topk = None):
    start_time = time.time()

    bit_index = 0

    with torch.no_grad():

        input_ids = prompt_template(prompt_text, model, tokenizer, mode = mode, role = role)

        if alg.lower() in ["ac","rolro"]:
            max_val = 2 ** kwargs["precision"]  # num of intervals ; max_val = 2**52
            cur_interval = [0, max_val]
            seed=42

        full_bits = ""
        past_key_values = None

        tokens = []
        tokens_bits = []

        full_ids = torch.cat((input_ids,tokenizer.encode(stego_text, add_special_tokens=False,return_tensors="pt").to(model.device)),dim=1)

        # print(len(input_ids[0]), min(len(full_ids[0]), max_new_tokens+len(input_ids[0])))
        for i in tqdm(range(len(input_ids[0]), min(len(full_ids[0]), max_new_tokens+len(input_ids[0])))):
           
            output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            log_prob, past_key_values = output.logits, output.past_key_values  # logits: (1, Time, Vocab)
            log_prob = log_prob[0, -1, :]  # (Vocab,)
            log_prob -= log_prob.max()
            prob = torch.exp(log_prob).reshape(-1)
            prob = prob / prob.sum()

            ######logits 预处理#######TODO
            prob, indices = prob.sort(descending=True)
            mask = prob > 0
            prob=prob[mask]
            indices = indices[mask]   
            prob = prob[:topk]
            indices = indices[:topk]         
            prob = prob / prob.sum()
            ########################


            embed_id = full_ids[0][i].item()
            tokens.append(embed_id)

            try:
                if alg.lower() in ["ac","rolro"]:
                    cur_interval, extract_bits = decoder(alg, prob, indices, embed_id, cur_interval, **kwargs)
                elif alg.lower() in provably_secure_alg:
                    extract_bits = decoder(alg, prob, indices, embed_id ,mask_generator = mask_generator, **kwargs)
                else:
                    extract_bits = decoder(alg, prob, indices, embed_id, **kwargs)
            except:
                extract_bits = ""
                # mask_generator.generate_bits(kwargs["precision"])
            input_ids = full_ids[0][i].reshape(1,1)
            # exit()
            full_bits += extract_bits
            tokens_bits.append(extract_bits)
        # print(mask_generator.test)

    return full_bits,tokens_bits,tokens


def extract(model, tokenizer, steg_texts, 
            prompts=None,
            mode = "generate",
            role = "user",
            max_new_tokens=256, 
            alg="ac", 
            output_filepath=None,
            do_check=False,
            embedded_bits = None,
            check_tokens = None,
            kwargs = None,
            topk = None
            ):
    
    mask_generator = None
    right_index = 0
    
    if alg in provably_secure_alg:
        input_key = bytes.fromhex(kwargs["input_key"])
        sample_seed_prefix = bytes.fromhex(kwargs["sample_seed_prefix"])
        input_nonce = bytes.fromhex(kwargs["input_nonce"])
        mask_generator = DRBG(input_key, sample_seed_prefix + input_nonce)



    if output_filepath is None:
        for stega_idx in tqdm(range(len(steg_texts))):
            prompt_text = prompts[stega_idx]            
            steg_text = steg_texts[stega_idx]

            extract_bits, tokens_bits, tokens = extract_single(model, tokenizer, steg_text, 
                                          prompt_text = prompt_text,
                                          mode = mode,
                                          role = role,
                                          max_new_tokens = max_new_tokens, 
                                          alg = alg,
                                          mask_generator = mask_generator,
                                          kwargs=kwargs,
                                          topk = topk
                                          )
            print("extract result:",extract_bits)

            if do_check:
                if embedded_bits is None:
                    print("请提供校验所用的原始嵌入bits流")
                else:
                    if extract_bits != embedded_bits[stega_idx]:
                        if tokens != check_tokens[stega_idx]:
                            print("tokenaizer error")
                            print("extract_tokens:",tokens)
                            print("real_tokens",check_tokens[stega_idx])
                            print(right_index,'/',stega_idx+1)
                        else:
                            print("unknown error")
                            print(f"extract_error @ {steg_text}")
                            print(tokens_bits)
                            print(tokens)
                            print(check_tokens[stega_idx])
                            print(extract_bits)
                            print(embedded_bits[stega_idx])
                            print(right_index,'/',stega_idx+1)
                    else:
                        right_index+=1
                        print("right!",right_index,'/',stega_idx+1)
                        
            if alg in provably_secure_alg:
                if alg in meteor_alg:
                    # print(mask_generator.test)
                    for _ in range((max_new_tokens+10)*(stega_idx+1)-mask_generator.test):
                        mask_generator.generate_bits(kwargs["precision"])
                    # print(mask_generator.test)
                elif alg in group_alg:
                    for _ in range((2*max_new_tokens+10)*(stega_idx+1)-mask_generator.test):
                        mask_generator.generate_bits(kwargs["precision"])    
                elif alg in discop_alg: 
                    for _ in range((math.ceil(math.log2(tokenizer.vocab_size)) * max_new_tokens+10)*(stega_idx+1)-mask_generator.test):
                        mask_generator.generate_bits(kwargs["precision"])    

    else:
        with jsonlines.open(output_filepath, "w") as f_out:
            
            for stega_idx in tqdm(range(len(steg_texts)),desc=f"decoding stegos {alg} into {output_filepath}"):
                prompt_text = prompts[stega_idx]
                steg_text = steg_texts[stega_idx]

                extract_bits, tokens_bits, tokens = extract_single(model, tokenizer, steg_text, 
                                              prompt_text,
                                              mode,
                                              role,
                                              max_new_tokens=max_new_tokens, 
                                              alg=alg,
                                              mask_generator = mask_generator,
                                              kwargs=kwargs,
                                              topk = topk)
                results = {"extract_tokens":tokenizer.encode(prompt_text+steg_text)[len(tokenizer.encode(prompt_text)):],
                           "extract_bits":extract_bits,
                           "stego":steg_text,
                           "tokens_bits": tokens_bits}
                print("extract result:",extract_bits)
                if do_check:
                    results["check_result"] = (extract_bits == embedded_bits[stega_idx])
                    if embedded_bits is None:
                        print("请提供校验所用的原始嵌入bits流")
                    else:
                        if extract_bits != embedded_bits[stega_idx]:
                            if tokens != check_tokens[stega_idx]:
                                print("tokenaizer error")
                                print("extract_tokens:",tokens)
                                print("real_tokens",check_tokens[stega_idx])
                                print(right_index,'/',stega_idx+1)
                            else:
                                print("unknown error")
                                print(f"extract_error @ {steg_text}")
                                print(tokens_bits)
                                print(tokens)
                                print(check_tokens[stega_idx])
                                print(extract_bits)
                                print(embedded_bits[stega_idx])
                                print(right_index,'/',stega_idx+1)
                        else:
                            right_index+=1
                            print("right!",right_index,'/',stega_idx+1)

                if alg in provably_secure_alg:
                    if alg in meteor_alg:
                        # print(mask_generator.test)
                        for _ in range((max_new_tokens+10)*(stega_idx+1)-mask_generator.test):
                            mask_generator.generate_bits(kwargs["precision"])
                        # print(mask_generator.test)
                    elif alg in group_alg:
                        for _ in range((2*max_new_tokens+10)*(stega_idx+1)-mask_generator.test):
                            mask_generator.generate_bits(kwargs["precision"])     
                    elif alg in discop_alg: 
                        # print(mask_generator.test)
                        for _ in range((math.ceil(math.log2(tokenizer.vocab_size)) * max_new_tokens+10)*(stega_idx+1)-mask_generator.test):
                            mask_generator.generate_bits(kwargs["precision"])      
                        # print(mask_generator.test)

                f_out.write(results)
    if do_check:          
        print("right:",right_index/len(steg_texts))




def parse_arg_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuid", type=str, default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--generate_config", type=str, default="src/config_template.json")
    parser.add_argument("--model_config", type=str, default="src/config_LLM.json")
    parser.add_argument("--generate_num", type=int, default=1508)
    parser.add_argument("--secret_bits_file", type=str, default ="src/bit_stream.txt")
    parser.add_argument("--T", type=float,default=1.0)
    parser.add_argument("--P", type=float,default=1.0)
    parser.add_argument("--theta", type=float,default=0.01) # 熵因子
    parser.add_argument("--alpha", type=float,default=0) # 乘方因子
    parser.add_argument("--c", type=float,default=0.5) #乘在改变前
    parser.add_argument("--beta", type=float,default=2)#bias of log()
    parser.add_argument("--cache_dir", type=str, default="/home/baiminhao/.cache/huggingfacee")
    # return parser.parse_args(["--use_lora"])
    parser.add_argument("--target_corpus", type=str, help="Target corpus to adjust porbs",default="XXXX/trainChatglm_tokenized_frequency.json")
    parser.add_argument("--model_corpus", type=str, help="Model corpus to adjust porbs",default="XXXX/config_LLM-config_template-1508_chatglm_tokenized_frequency.json")
    parser.add_argument("--baseT",type=float,default=1)
    return parser.parse_args()



if __name__ == '__main__':

    # load os configs
    args = parse_arg_main()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    seed = args.seed
    out_dir = args.out_dir
    sampling_configs = json.load(open(args.generate_config, "r"))
    model_configs = json.load(open(args.model_config, "r"))
    generate_num = args.generate_num
    

    # load secret bits
    if args.secret_bits_file is not None:
        with open(args.secret_bits_file, 'r', encoding='utf8') as f:
            bit_stream_ori = f.read().strip()
        bit_stream = list(bit_stream_ori)
        bit_stream = ''.join(bit_stream)
    else:
        bit_stream = np.random.randint(high=2,low=0, size=(1, 500000)).tolist()[0]
        bit_stream = "".join([str(b) for b in bit_stream])

        


    # load language model
    model_name_or_path = model_configs['model_name_or_path']
    if model_configs["precision"] == 'half':
        if "glm" in model_name_or_path:
            tokenizer=AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name_or_path,trust_remote_code=True).half().cuda()  
        else:       
            tokenizer=AutoTokenizer.from_pretrained(model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path).half().cuda()
    else:
        if "glm" in model_name_or_path:
            tokenizer=AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name_or_path,trust_remote_code=True).cuda()    
        else:
            tokenizer=AutoTokenizer.from_pretrained(model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path).cuda()
    

    tokenizer.name = model_name_or_path
    model = model.eval()
    set_seed(seed)
    bit_index = 0

    # load alg config  
    alg = sampling_configs["algorithm"]
    kwargs = sampling_configs[alg]
    # print(kwargs)

    # load model config
    max_new_tokens = model_configs["max_new_tokens"]
    topk = model_configs["topk"] #topk == 0 
    if topk == 0:
        topk = tokenizer.vocab_size
    
    # load prompts
    prompt_mode = model_configs["prompt_template"]
    prompts = [model_configs["prompt"]] * generate_num
    # prompts=[]
    # with open('/data1/pky/dataset/m-a-p/xhs/train.jsonl', 'r') as f:
    #     for line in f:
    #         prompts.append(json.loads(line)['instruction'])
    # prompts = [model_configs["prompt"]] * generate_num
    
    prompts = prompts[:generate_num]





    role = model_configs["role"]
    
    # outfile name 
    model_config_name, _ = os.path.splitext(os.path.basename(args.model_config))
    alg_config_name, _ = os.path.splitext(os.path.basename(args.generate_config))
    # outfile = "-".join([model_config_name, alg_config_name, str(generate_num)])+'.json'
    outfile = "-".join([model_config_name, alg_config_name, str(args.generate_num),"alpha",str(args.alpha),"theta",str(args.theta),"baseT",str(args.baseT),"c",str(args.c)])+'.json'
    outfile = os.path.join(out_dir,outfile)

    embed(model, tokenizer, bit_stream, 
        output_filepath=outfile, 
        prompts=prompts,
        mode = prompt_mode, 
        role = role,
        sentence_num=generate_num,
        alg=alg, 
        max_new_tokens=max_new_tokens, 
        kwargs = kwargs,
        topk = topk)
    

    # load check_file
    stega_texts_path = outfile
    stega_texts = []
    embedded_bits = []
    check_tokens = []

    with open(stega_texts_path, "r", encoding = "utf-8") as f:
        for stega_data in f:
            stega_texts.append(json.loads(stega_data)['stego'])
            embedded_bits.append(json.loads(stega_data)['bits'])
            check_tokens.append(json.loads(stega_data)['tokens'])

    

    extract(model, tokenizer, stega_texts,
            prompts=prompts,
            mode = prompt_mode,
            role = role,
            max_new_tokens=max_new_tokens, 
            alg=alg, 
            output_filepath="output/test.json",  
            do_check=True,
            embedded_bits = embedded_bits,
            check_tokens = check_tokens,
            kwargs = kwargs,
            topk = topk)