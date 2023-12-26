import json
with open("/users/PAA0201/lzy37ld/why_attack_lzy/data/train_val_test.json") as f:
	split = json.load(f)
length = 0
for _ in ["train","test","val"]:
	length += len(split[_])
print(length)
