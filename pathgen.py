import os
import pandas as pd

# 定义UCF101目录的路径
base_dir = '/data/xuchenheng/UCF101'

# 初始化一个空列表来存储文件信息
file_data = []

# 遍历UCF101目录下的所有子目录
for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if os.path.isdir(category_path):
        # 遍历每个子目录中的AVI文件
        for file_name in os.listdir(category_path):
            if file_name.endswith('.avi'):
                file_path = os.path.join(category_path, file_name)
                # 将文件路径和类别添加到列表中
                file_data.append([file_path, category])

# 将数据转换为DataFrame
df = pd.DataFrame(file_data, columns=['File Path', 'Category'])

# 将DataFrame写入Excel文件
output_file = 'UCF101_file_list.xlsx'
df.to_excel(output_file, index=False)

print(f"Excel文件已保存到: {output_file}")