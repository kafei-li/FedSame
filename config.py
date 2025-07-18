class Config:
    # 通用参数
    seed = 42
    device = 'cuda'  # 或 'cpu'
    num_rounds = 200
    num_clients = 50
    num_tasks = 10
    clients_per_round = 0.8
    batch_size = 32
    local_epochs = 1

    # FedSame参数
    alpha = 2
    beta = 2
    m = 10
    lambda_ = 0.1
    learning_rate = 0.01

    # 数据相关
    dirichlet_alpha = 0.1  # Non-IID