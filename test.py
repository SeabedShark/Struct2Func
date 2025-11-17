import torch

# 加载.pt文件
data = torch.load('/home/senyutang/protein/data/test24/processed/struct/A0A1W2PPG7.pt', map_location='cpu')  # 强制使用CPU避免GPU问题

# 打印基础信息
print("类型:", type(data))
print("键/结构:", data.keys() if isinstance(data, dict) else "非字典结构")

# 如果是模型参数（常见情况）
if isinstance(data, dict):
    for key, value in data.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else type(value)}")
else:
    print("内容示例:", data)