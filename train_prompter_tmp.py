import transformers
# name = "meta-llama/Llama-2-7b-hf"
# config = transformers.AutoConfig.from_pretrained(name)
# config.save_pretrained("/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_vicuna_ckpt/checkpoint-12000")

x = transformers.AutoModelForCausalLM.from_pretrained("/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_vicuna_ckpt/checkpoint-12000")
print(x)