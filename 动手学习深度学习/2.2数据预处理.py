import os
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data = pd.read_csv(data_file)
inputs, output = data.iloc[:, 0:2], data.iloc[:, 2]
# 寻找缺失值最多的列
missing_values = inputs.isnull().sum()
print(missing_values)
column_with_most_missing_values = missing_values.idxmax()
print(column_with_most_missing_values)
# 删除缺失值最多的列
inputs = inputs.drop(column_with_most_missing_values, axis=1)
# 对缺失值进行填充
inputs = inputs.fillna(inputs.mean())
inputs = pd.get_dummies(inputs, dummy_na=True)
# 将处理后的数据转换为张量格式
inputs = torch.tensor(inputs.values)
output = torch.tensor(output.values)
print(inputs)
print(output)