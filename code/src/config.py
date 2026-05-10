# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 64,           # 大幅缩减：2M参数 vs 417样本导致过拟合，64维更合适
    'nhead': 4,             # 注意力头数量
    'num_layers': 2,        # 减少层数：防止过拟合
    'dim_feedforward': 128, # 缩减前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 60,       # 小模型收敛更快，60轮足够
    'learning_rate': 1e-4,  # 小模型可以用稍大的学习率
    'dropout': 0.35,        # 强正则化：防止过拟合
    'feature_num': feature_num,
    'max_grad_norm': 5.0,
    'drop_clip': False,     # 启用梯度裁剪，防止梯度爆炸

    'pairwise_weight': 2.0, # 加强配对损失权重，更好区分股票相对排名
    'base_weight': 1.0,     # 非top-k样本权重
    'top5_weight': 5.0,     # 大幅加强top-5样本权重，对齐评测指标

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}