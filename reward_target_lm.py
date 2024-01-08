from transformers import AutoTokenizer,GenerationConfig,AutoModelForCausalLM
from safe_rlhf.models import AutoModelForScore
import os
import torch.nn as nn
import torch
import itertools
from collections import defaultdict as ddict
import numpy as np
import torch
import copy
from torch.nn import CrossEntropyLoss
loss_fct = CrossEntropyLoss(reduction="none")

def cal_loss_avg(loss):
    non_zero_mask = loss != 0
        # 注意，如果一行全部为零，则其平均值将是NaN
    average_ignoring_zeros = torch.zeros(loss.size(0))
    for i in range(loss.size(0)):
        non_zero_values = loss[i, non_zero_mask[i]]
        if len(non_zero_values) > 0:
            average_ignoring_zeros[i] = non_zero_values.mean()
        else:
            average_ignoring_zeros[i] = float('nan')  # 如果一行全是0，则平均值为NaN
    return average_ignoring_zeros

def check_torch_dtype(config):
    kwargs = {}
    if config.torch_dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    return kwargs


# 为什么我不用deepspeed，本质上可以用，但是我一个gpu的vram太小了，所以多放几个model，不一定放得下，还不如分开。。。
def create_reward(config):
        
    if os.environ.get("RANK", "0") == "0":

        class RewardModel(nn.Module): 
            def __init__(
                self, 
                config,
                device_map
            ): 
                super().__init__()
                model_name = config.model_name
                self.template = config.template
                self.batch_size = config.batch_size
                self.config = config
                kwargs = check_torch_dtype(config)
                if config.debug:
                    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs,**device_map)
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    tokenizer.padding_side = "left"
                else:
                    model = AutoModelForScore.from_pretrained(model_name, **kwargs,**device_map)
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    tokenizer.padding_side = "right"
                if not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token
                self.model = model
                self.tokenizer= tokenizer

            def _reward_run_batch(self,batch,device):
                input_ids = self.tokenizer(batch, return_tensors='pt',padding= True).to(device)
                outputs = self.model(**input_ids).end_scores
                return outputs

            def _reward_run(self, q_and_p_s, ans_s, device):
                outputs_l = []
                batch_size = self.batch_size
                for i in range(0,len(q_and_p_s),batch_size):
                    
                    batch_inputs = q_and_p_s[i: i +batch_size]
                    batch_outputs = ans_s[i: i +batch_size]
                    batch = [self.template.format(model_input = batch_inputs[index], model_output = batch_outputs[index]) for index in range(len(batch_inputs))]
                    try:
                        outputs_l.append(self._reward_run_batch(batch,device))
                    except:
                        print("run one by one for _reward_run")
                        single_outputs_l = []
                        for single_batch in batch:
                            single_outputs_l.append(self._reward_run_batch(single_batch,device))
                        outputs_l.extend(single_outputs_l)        
                return torch.cat(outputs_l,dim = 0).view(-1),outputs_l
            
            def _reward_run_debug(self, q_and_p_s, ans_s, device):
                # outputs_l = []
                # batch_size = self.batch_size
                # for i in range(0,len(q_and_p_s),batch_size):
                #     batch_inputs = q_and_p_s[i: i +batch_size]
                #     batch_outputs = ans_s[i: i +batch_size]
                #     batch = [self.template.format(model_input = batch_inputs[index], model_output = batch_outputs[index]) for index in range(len(batch_inputs))]
                #     try:
                #         input_ids = self.tokenizer(batch, return_tensors='pt',padding= True).to(device)
                #         outputs = self.model(**input_ids).logits[:,-1,0].unsqueeze(-1)
                #         outputs_l.append(outputs)
                #     except:
                #         print("run one by one")
                #         single_outputs_l = []
                #         for single_batch in batch:
                #             input_ids = self.tokenizer(single_batch, return_tensors='pt',padding= True).to(device)
                #             outputs = self.model(**input_ids).logits[:,0,0].unsqueeze(-1)
                #             single_outputs_l.append(outputs)
                #         outputs_l.extend(single_outputs_l)        
                # return torch.cat(outputs_l,dim = 0).view(-1),outputs_l
                pass
            
            def reward_run(self, q_and_p_s, ans_s, device, mode):
                # "q_and_p_s are 'harmful input + prompt' question + prompt, ans_s are cost_lm's response"
                if self.config.debug:
                    scores, _ = self._reward_run_debug(q_and_p_s, ans_s, device)
                else:
                    scores, _ = self._reward_run(q_and_p_s, ans_s, device)

                return scores

        device_map = {"device_map":"auto"}
        reward_model_device = "cuda:0"

        reward_model = RewardModel(config.reward_lm,device_map=device_map)
        reward_model.eval()
        reward_model.requires_grad_(False) 

        @torch.no_grad()
        def get_reward(q_s,ans_s,mode = "train"):
            scores = reward_model.reward_run(q_s,ans_s,device = reward_model_device, mode = mode)
            return scores
        
        
    else:
        get_reward = True

    return get_reward
        
def create_targetlm(config):

    if os.environ.get("RANK", "0") == "0":

        class Target_Model(nn.Module): 
            def __init__(
                self, 
                config,
                device_map
            ): 
                super().__init__()
                model_name = config.model_name
                self.template = config.template
                if config.system_message:
                    self.template = self.template.format(system_message = config.system_message, input = "{input}", prompt = "{prompt}")
                else:
                    self.template = self.template.format(input = "{input}", prompt = "{prompt}")
                self.ppl_template = config.ppl_template
                self.batch_size = config.batch_size
                kwargs = check_torch_dtype(config)

                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs,**device_map)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.padding_side = "left"
                if not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token
                self.model = model
                self.tokenizer= tokenizer
                self.gen_kwargs = {"pad_token_id":self.tokenizer.pad_token_id, "eos_token_id":self.tokenizer.eos_token_id, "bos_token_id":self.tokenizer.bos_token_id}
                
            def create_gen_config(self,gen_config):
                self.gen_config = GenerationConfig(**gen_config, **self.gen_kwargs)

            @torch.no_grad()
            def get_target_lm_generation(self, q_s,p_s,num_return_sequences = -1,after_sys_tokens = None):
                # q_s : questions  p_s:prompts

                generation_configs = config.target_lm.generation_configs
                if num_return_sequences != -1:
                    generation_configs.num_return_sequences = num_return_sequences
                target_model.create_gen_config(generation_configs)
                assert len(q_s) == len(p_s)
                if after_sys_tokens is not None:
                    assert len(q_s) == len(after_sys_tokens)
                
                target_model.after_sys_tokens = after_sys_tokens
                generation = target_model.targetlm_run(q_s,p_s,device = self.target_model_device)
                return generation    
            
            def _targetlm_run_batch(self,batch,device):
                input_ids = self.tokenizer(batch, return_tensors='pt',padding= True).to(device)
                output = self.model.generate(**input_ids,generation_config = self.gen_config)
                output = output[:,input_ids["input_ids"].shape[-1]:]
                output_text = self.tokenizer.batch_decode(output,skip_special_tokens= True) 
                return output_text

            # q_s questions, p_s prompts
            def _targetlm_run(self, q_s, p_s, device):
                outputs_l = []
                batch_size = self.batch_size
                for i in range(0,len(q_s),batch_size):    
                    batch_inputs = q_s[i: i +batch_size]
                    batch_outputs = p_s[i: i +batch_size]
                    batch = [self.template.format(input = batch_inputs[index], prompt = batch_outputs[index]) for index in range(len(batch_inputs))]
                    if self.after_sys_tokens is not None:
                        # space here is important!!
                        batch = [batch[i] + " " + self.after_sys_tokens[i] for i in range(len(batch))]
                    if i < 2:
                        print(batch[0])
                        print(self.tokenizer.decode(self.tokenizer.encode(batch[0])))
                        print("Add special tokens should be True")
                    try: 
                        outputs_l.extend(self._targetlm_run_batch(batch,device))
                    except:
                        print("run one by one for _targetlm_run")
                        single_outputs_l = []
                        for single_batch in batch:
                            single_outputs_l.extend(self._targetlm_run_batch(single_batch,device))
                        outputs_l.extend(single_outputs_l)        
                return outputs_l
            
            def targetlm_run(self, q_s, p_s, device):
                generations = self._targetlm_run(q_s, p_s, device)

                return generations
            
            @torch.no_grad()
            def ppl_run(self,q_s,p_s):
                ppl = self._ppl_run(q_s, p_s, device = self.target_model_device)
                return ppl
            def _ppl_run_batch(self,batch,device):
                input_ids = self.tokenizer(batch, return_tensors='pt',padding= True).to(device)
                attention_mask = input_ids.attention_mask
                labels = copy.deepcopy(input_ids.input_ids)
                labels = torch.where(attention_mask == 0, torch.tensor(-100), labels)
                logits = self.model(**input_ids).logits
                shifted_labels = labels[...,1:].contiguous()
                shifted_logits = logits[...,:-1,:].contiguous()
                shifted_logits = shifted_logits.permute(0,2,1)
                loss = loss_fct(shifted_logits, shifted_labels)
                loss = cal_loss_avg(loss)
                ppl = torch.exp(loss)
                return ppl.detach().cpu().tolist()

            # q_s questions, p_s prompts
            def _ppl_run(self, q_s, p_s, device):
                outputs_l = []
                batch_size = self.batch_size
                for i in range(0,len(q_s),batch_size):    
                    batch_inputs = q_s[i: i +batch_size]
                    batch_outputs = p_s[i: i +batch_size]
                    batch = [self.ppl_template.format(input = batch_inputs[index], prompt = batch_outputs[index]).strip() for index in range(len(batch_inputs))]
                    # batch = [self.template.format(input = batch_inputs[index], prompt = batch_outputs[index]).strip() for index in range(len(batch_inputs))]
                    if i < 2:
                        print(batch[0])
                        print(self.tokenizer.decode(self.tokenizer.encode(batch[0])))
                        print("Add special tokens should be True")
                    try:
                        outputs_l.extend(self._ppl_run_batch(batch,device))
                    except:
                        print("run one by one for _ppl_run")
                        single_outputs_l = []
                        for single_batch in batch:
                            single_outputs_l.extend(self._ppl_run_batch(single_batch,device))
                        outputs_l.extend(single_outputs_l)        
                return outputs_l

        device_map = {"device_map":"auto"}
        target_model_device = "cuda:0"

        target_model = Target_Model(config.target_lm,device_map=device_map)
        target_model.target_model_device = target_model_device
        target_model.eval()
        target_model.requires_grad_(False)
        return target_model
        
    else:
        target_model = None

    return target_model





        
def create_prompterlm(config):

    if os.environ.get("RANK", "0") == "0":

        class Prompter_Model(nn.Module): 
            def __init__(
                self, 
                config,
                device_map
            ): 
                super().__init__()
                model_name = config.model_name
                self.batch_size = config.batch_size
                self.template = config.template
                kwargs = check_torch_dtype(config)
                model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs,**device_map)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.padding_side = "left"
                if not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token
                self.model = model
                self.tokenizer= tokenizer
                self.gen_kwargs = {"pad_token_id":self.tokenizer.pad_token_id, "eos_token_id":self.tokenizer.eos_token_id, "bos_token_id":self.tokenizer.bos_token_id}
                
            def create_gen_config(self,gen_config):
                self.gen_config = GenerationConfig(**gen_config, **self.gen_kwargs)
            def _prompterlm_run_batch(self,batch,device):
                input_ids = self.tokenizer(batch, return_tensors='pt',padding= True).to(device)
                output = self.model.generate(**input_ids,generation_config = self.gen_config)
                output = output[:,input_ids["input_ids"].shape[-1]:]
                output_text = self.tokenizer.batch_decode(output,skip_special_tokens= True) 
                return output_text

            # q_s questions, p_s prompts
            def _prompterlm_run(self, q_s, device):
                outputs_l = []
                batch_size = self.batch_size
                for i in range(0,len(q_s),batch_size):    
                    batch_inputs = q_s[i: i +batch_size]
                    batch = [self.template.format(input = batch_inputs[index]) for index in range(len(batch_inputs))]
                    if i < 2:
                        print(batch[0])
                        print(self.tokenizer.decode(self.tokenizer.encode(batch[0])))
                        print("Add special tokens should be True")
                    try:
                        outputs_l.extend(self._prompterlm_run_batch(batch,device))
                    except:
                        print("run one by one for _prompterlm_run")
                        single_outputs_l = []
                        for single_batch in batch:
                            single_outputs_l.extend(self._prompterlm_run_batch(single_batch,device))
                        outputs_l.extend(single_outputs_l)        
                return outputs_l
            
            def prompterlm_run(self, q_s, device):
                generations = self._prompterlm_run(q_s, device)

                return generations
            

        device_map = {"device_map":"auto"}
        prompter_model_device = "cuda:0"

        prompter_model = Prompter_Model(config.prompter_lm,device_map=device_map)
        prompter_model.eval()
        prompter_model.requires_grad_(False)

        @torch.no_grad()
        def get_prompter_lm_generation(q_s,num_return_sequences = -1):
            # q_s : questions  p_s:prompts

            generation_configs = config.prompter_lm.generation_configs
            if num_return_sequences != -1:
                generation_configs.num_return_sequences = num_return_sequences
            prompter_model.create_gen_config(generation_configs)
            
            generation = prompter_model.prompterlm_run(q_s,device = prompter_model_device)
            return generation
        
    else:
        get_prompter_lm_generation = True

    return get_prompter_lm_generation



