# # import json
# # with open("/users/PAA0201/lzy37ld/why_attack_lzy/data/success_JB_victimmodel=llama2-7b-chat_sampleway=loss_100_nsample=200.json") as f:
# #     data_jb = json.load(f)
# # jb_qs = list(data_jb.keys())

# # with open("/users/PAA0201/lzy37ld/why_attack_lzy/data/offset_q_target.json") as f:
# #     offset_q_target = json.load(f)

# # all_qs = []
# # for offset in offset_q_target:
# #     for q in offset_q_target[offset]:
# #         all_qs.append(q)
# # print(len(set(all_qs)),"len(all_qs)")
        

# # all_need_offsets = []
# # for jb_q in jb_qs:
# #     for offset in offset_q_target:
# #         if jb_q in offset_q_target[offset]:
# #             all_need_offsets.append(offset)
# #             break
            
# # print(set(all_need_offsets))


# # import os
# # base="/fs/ess/PAA0201/lzy37ld/why_attack/data/s_p_t_evaluate/vicuna-7b-chat-v1.5|max_new_tokens_60"
# # for offset in set(all_need_offsets):
# #     get_it = False
# #     for now_Ihave in os.listdir(base):
# #         if offset in now_Ihave:
# #             get_it = True
# #             break
# #     if not get_it:
# #         print(offset,"dont get")

        
    

# # adv_string_init = open("test.txt", 'r').readlines()[0]
# # print(0)

# import json
# with open("/users/PAA0201/lzy37ld/why_attack_lzy/data/train_val_test.json") as f:
#     data = json.load(f)
#     print(len(data["train"]))
#     print(len(data["val"]))
#     print(len(data["test"]))
#     print(len(data["hard"]))
#     print(len(data["unknown"]))

# # with open("/users/PAA0201/lzy37ld/why_attack_lzy/data/success_JB_victimmodel=llama2-7b-chat_sampleway=step_nsample=200.json") as f:
# #     data = json.load(f)
# #     print(len(data))

    

# # with open("/users/PAA0201/lzy37ld/why_attack_lzy/data/allchecked_JB_victimmodel=llama2-7b-chat.json") as f:
# #     data = json.load(f)
# #     print(len(data["all_checked_queries"]))

import deepspeed
deepspeed.ops.adam.cpu_adam.CPUAdamBuilder().load()