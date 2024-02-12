from transformers import AutoTokenizer,AutoModelForCausalLM
t = AutoTokenizer.from_pretrained("cais/HarmBench-Llama-2-13b-cls")
m = AutoModelForCausalLM.from_pretrained("cais/HarmBench-Llama-2-13b-cls")