#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers

from torch.utils.data import Dataset
from transformers import Trainer
import torch.nn.functional as F
import json
# from accelerate.utils import set_seed
# set_seed(42)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    # "q_p_r": (
    #     "### Query:{query}### Prompt:{prompt}### Response:{response}"
    # ),
    "q_target_lm_generation_p": (
       "### Query:{q} ### Response:{target_lm_generation} ### Prompt:"
    ),
    "q_target_p": (
       "### Query:{q} ### Response:{target} ### Prompt:"
    ),
    "q_p": (
       "### Query:{q} ### Prompt:"
    ),
}

# https://github.com/facebookresearch/text-adversarial-attack/blob/5170ad0ef7e25758df394c4323b356e8615e61e4/whitebox_attack.py#L48
def log_perplexity(logits, coeffs, position):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    # only calculate loss for continuation not for instruction
    position_for_loss = position != torch.tensor(-100)
    position_for_loss = position_for_loss[:,:-1]
    B_T = -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1)
    B_T_for_loss = B_T[position_for_loss]
    return B_T_for_loss.mean()


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    sampled_queries: str = field(default=None, metadata={"help": "Path to the sampled queries."})
    split_path: str = field(default=None, metadata={"help": "Path to the split queries."})
    prompt_type: str = field(default="q_r", metadata = {"help": "chose which type to use"})
    debug_data: bool  = field(default=False, metadata = {"help": "chose which type to use"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    ppl_ratio: float = field(default=0.1, metadata={"help": "ratio of ppl loss"})
    ppl_loss: bool = field(default=False, metadata={"help": "If calculate ppl_loss"})


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        split_path = data_args.split_path
        sampled_queries = data_args.sampled_queries
        prompt_type = data_args.prompt_type

        with open(split_path) as f:
            train_splits = json.load(f)["train"]

        with open(sampled_queries) as f:
            sampled_queries = json.load(f)

        
        list_data_dict = []
        if data_args.debug_data:
            train_splits = train_splits[:1]
        for q in train_splits:
            list_data_dict.extend(sampled_queries[q])
        
        logging.warning("Formatting inputs...")
        prompt_template = PROMPT_DICT[prompt_type]

        sources = [
            prompt_template.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['p']}{tokenizer.eos_token}" for example in list_data_dict]
        print('list_data_dict[0]',list_data_dict[0])
        print('sources[0',sources[0])
        print('targets[0]',targets[0])

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):

            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss_metric = {"adv_loss":loss.item()}
            ppl_loss = None
            if self.ppl_loss:
                # ppl_inputs = copy.deepcopy(inputs)
                # ppl_inputs["labels"] = torch.where(ppl_inputs["labels"] == self.tokenizer.eos_token_id, torch.tensor(-100), ppl_inputs["labels"])
                logits = outputs["logits"]
                samples = F.gumbel_softmax(logits, hard=False,dim=-1)
                ppl_inputs_embeds = (samples @ model_ppl_embedding).to(torch.bfloat16)
                ppl_outputs = model_ppl(inputs_embeds = ppl_inputs_embeds)
                ppl_logits = ppl_outputs.logits
                ppl_loss = log_perplexity(ppl_logits,samples,position = torch.where(inputs["labels"] == self.tokenizer.eos_token_id, torch.tensor(-100), inputs["labels"]))
                loss = ppl_loss * self.ppl_ratio + loss
            if ppl_loss:
                loss_metric.update({"ppl_loss":ppl_loss.item()})
                self.log(loss_metric)

            


            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    if training_args.ppl_loss:
        with torch.no_grad():
            model_ppl = copy.deepcopy(model)
            model_ppl.requires_grad_(False)
            model_ppl.to(f"cuda:{training_args.local_rank}")
            model_ppl_embedding = model_ppl.get_input_embeddings()(torch.arange(0, len(tokenizer)).long().to(f"cuda:{training_args.local_rank}"))
            model_ppl_embedding.to(f"cuda:{training_args.local_rank}")


    trainer.ppl_loss = training_args.ppl_loss
    trainer.ppl_ratio = training_args.ppl_ratio
    trainer.train()
    # trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()