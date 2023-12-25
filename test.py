# def find_dividers(lst, num_parts):
#     lst.sort()
#     n = len(lst)
#     dividers = []

#     for i in range(1, num_parts):
#         # 计算每个分区的理想分界点
#         divider_index = int(n * i / num_parts)
#         dividers.append(lst[divider_index])

#     return dividers

# # 假设这是你的列表
# your_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]

# # 调用函数，找到9个分界点
# dividers = find_dividers(your_list, 10)

# print("Dividers:", len(dividers))


import jsonlines
train_offsets = ["offset_0", "offset_10", "offset_20", "offset_30", "offset_40", "offset_50", "offset_60", "offset_70", "offset_80", "offset_90", "offset_100", "offset_110", "offset_120", "offset_130", "offset_140", "offset_160", "offset_170", "offset_180", "offset_190", "offset_200", "offset_210", "offset_240", "offset_250", "offset_260", "offset_270", "offset_280", "offset_290", "offset_300", "offset_310", "offset_320", "offset_330", "offset_340", "offset_360", "offset_370", "offset_380", "offset_390", "offset_400", "offset_420", "offset_430", "offset_440"]
template = "/users/PAA0201/lzy37ld/why_attack/s_p_t_evaluate/llama2-7b-chat|max_new_tokens_60/{offset}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl"
for offset in train_offsets:
    path = template.format(offset = offset)
    with open(path) as f:
        lines = f.readlines()
        assert len(lines) == 1920000, "wrong"
        print(offset, len(lines))