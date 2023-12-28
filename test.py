# 


def select_indexes(lst):
    selected_indexes = []
    for i in range(0, len(lst), 3):
        # 获取当前组的元素和对应索引
        group = lst[i:i+3]
        group_indexes = list(range(i, min(i+3, len(lst))))

        # 检查组内是否有大于0的元素
        positive_indexes = [idx for idx, val in zip(group_indexes, group) if val > 0]

        # 如果有大于0的元素，选取第一个；否则，选取组内的第一个元素
        selected_index = positive_indexes[0] if positive_indexes else group_indexes[0]
        selected_indexes.append(selected_index)

    return selected_indexes

# 示例列表
your_list = [0, -1, 2, 0, 1, 0, -1, 2, 3]

# 获取选中的索引
selected_indexes = select_indexes(your_list)
print(selected_indexes)