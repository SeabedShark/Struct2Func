#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
续训使用示例

使用方法：
1. 从头开始训练（正常使用）：
   python train_modal_experiments.py

2. 从检查点续训：
   修改 train_modal_experiments.py 中的 CONFIG["resume_from"] 为检查点路径
   例如：CONFIG["resume_from"] = "./modal_train_eval_test/w_o_struct/best.pt"
   然后运行：python train_modal_experiments.py

3. 从特定epoch续训：
   CONFIG["resume_from"] = "./modal_train_eval_test/w_o_struct/epoch_002.pt"

4. 从last.pt续训：
   CONFIG["resume_from"] = "./modal_train_eval_test/w_o_struct/last.pt"
"""

# 示例：如何修改 CONFIG 来启用续训
EXAMPLE_CONFIG = {
    # 续训检查点路径（选择其中一个）
    "resume_from": "",  # 空字符串表示从头开始
    
    # 可用的检查点路径示例：
    # "resume_from": "./modal_train_eval_test/w_o_struct/best.pt",      # 从最佳模型续训
    # "resume_from": "./modal_train_eval_test/w_o_struct/epoch_002.pt", # 从第2个epoch续训  
    # "resume_from": "./modal_train_eval_test/w_o_struct/last.pt",      # 从最后保存的模型续训
}

print("续训功能已添加！")
print("使用方法：")
print("1. 修改 train_modal_experiments.py 中的 CONFIG['resume_from'] 为检查点路径")
print("2. 运行 python train_modal_experiments.py")
print("\n可用的检查点文件：")
print("- best.pt: 最佳验证损失的模型")
print("- epoch_XXX.pt: 每个epoch保存的模型") 
print("- last.pt: 训练结束时的最后状态")





