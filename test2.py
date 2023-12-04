
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def is_ascii(s):
	return s.isascii() and s.isprintable()

ascii_toks = []
ascii_wods = []
for i in range(3, tokenizer.vocab_size):
	if not is_ascii(tokenizer.decode([i])):
		ascii_toks.append(i)
		ascii_wods.append(tokenizer.decode(i))
print(ascii_toks)
print(ascii_wods)