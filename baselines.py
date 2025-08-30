import torch
import numpy as np
from .models import get_mnist_model
from .utils import evaluate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def static_similarity_matrix(num_tasks):
    #FedBone baseline: Creates a static, pre-defined similarity matrix.
    #Here, it's an identity matrix plus a uniform off-diagonal value,
    #followed by row normalization.
    omega = np.eye(num_tasks) + 0.5 * (np.ones((num_tasks, num_tasks)) - np.eye(num_tasks))
    omega = omega / omega.sum(axis=1, keepdims=True)  # 行归一化
    return omega

def mtpl_projection(models, num_shared=20):
    # MTPL: 固定低维子空间投影
    # 取所有模型参数，拼接后做PCA，投影到低维空间再还原
    params = [torch.cat([p.data.flatten().cpu() for p in m.parameters()]) for m in models]
    params = torch.stack(params).numpy()
    pca = PCA(n_components=num_shared)
    low_dim = pca.fit_transform(params)
    recon = pca.inverse_transform(low_dim)
    # 还原到模型参数
    new_models = []
    for i, m in enumerate(models):
        offset = 0
        new_state = {}
        for name, p in m.named_parameters():
            numel = p.data.numel()
            new_param = torch.tensor(recon[i, offset:offset+numel], dtype=p.data.dtype).reshape(p.data.shape)
            new_state[name] = new_param
            offset += numel
        m.load_state_dict(new_state, strict=False)
        new_models.append(m)
    return new_models

def mocha_similarity(models):
    # MOCHA: 交替优化任务模型和相似性矩阵
    num_tasks = len(models)
    params = [torch.cat([p.data.flatten().cpu() for p in m.parameters()]) for m in models]
    params = torch.stack(params).numpy()
    omega = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        for j in range(num_tasks):
            omega[i, j] = np.dot(params[i], params[j]) / (np.linalg.norm(params[i]) * np.linalg.norm(params[j]) + 1e-8)
    omega = np.abs(omega)
    omega = omega / omega.sum(axis=1, keepdims=True)
    return omega

def mas_grouping(models, num_groups=2):
    # MAS: 动态分组，组内强聚合，组间弱聚合
    num_tasks = len(models)
    params = [torch.cat([p.data.flatten().cpu() for p in m.parameters()]) for m in models]
    params = torch.stack(params).numpy()
    kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(params)
    groups = [np.where(kmeans.labels_ == i)[0] for i in range(num_groups)]
    return groups

def fedem_em(models, prev_weights=None):
    # FedEM: EM混合建模
    # 这里用参数空间的高斯混合权重模拟E步
    num_tasks = len(models)
    params = [torch.cat([p.data.flatten().cpu() for p in m.parameters()]) for m in models]
    params = torch.stack(params).numpy()
    # E步：计算每个模型对所有任务的权重（用高斯核）
    weights = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        for j in range(num_tasks):
            dist = np.linalg.norm(params[i] - params[j])
            weights[i, j] = np.exp(-dist)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights

def train(method, train_loaders, test_loaders, models, config, client_task_map):
    num_tasks = config.num_tasks
    device = config.device

    for rnd in range(config.num_rounds):
        selected = np.random.choice(len(train_loaders), int(config.clients_per_round * len(train_loaders)), replace=False)
        local_models = [None] * num_tasks
        for i in selected:
            task = client_task_map[i][0]
            model = models[task]
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
            for _ in range(config.local_epochs):
                for x, y in train_loaders[i]:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    out = model(x)
                    loss = torch.nn.functional.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()
            local_models[task] = [p.data.cpu().numpy().copy() for p in model.parameters()]

        # 聚合方式
        if method == 'fedbone':
            # 静态相似性矩阵加权聚合
            omega = static_similarity_matrix(num_tasks)
            for k in range(num_tasks):
                agg = None
                for p in range(num_tasks):
                    if local_models[p] is not None:
                        w = local_models[p]
                    else:
                        w = [param.data.cpu().numpy().copy() for param in models[p].parameters()]
                    if agg is None:
                        agg = [omega[k, p] * w_i for w_i in w]
                    else:
                        agg = [a + omega[k, p] * w_i for a, w_i in zip(agg, w)]
                for param, new in zip(models[k].parameters(), agg):
                    param.data = torch.tensor(new, dtype=param.data.dtype, device=param.data.device)

        elif method == 'mtpl':
            # 固定低维子空间投影
            models = mtpl_projection(models, num_shared=20)

        elif method == 'mocha':
            # 交替优化：用参数余弦相似性矩阵加权聚合
            omega = mocha_similarity(models)
            for k in range(num_tasks):
                agg = None
                for p in range(num_tasks):
                    if local_models[p] is not None:
                        w = local_models[p]
                    else:
                        w = [param.data.cpu().numpy().copy() for param in models[p].parameters()]
                    if agg is None:
                        agg = [omega[k, p] * w_i for w_i in w]
                    else:
                        agg = [a + omega[k, p] * w_i for a, w_i in zip(agg, w)]
                for param, new in zip(models[k].parameters(), agg):
                    param.data = torch.tensor(new, dtype=param.data.dtype, device=param.data.device)

        elif method == 'mas':
            # 动态分组，组内平均，组间弱聚合
            groups = mas_grouping(models, num_groups=2)
            for group in groups:
                # 组内平均
                group_params = []
                for t in group:
                    if local_models[t] is not None:
                        group_params.append(local_models[t])
                    else:
                        group_params.append([param.data.cpu().numpy().copy() for param in models[t].parameters()])
                mean_params = [np.mean([gp[i] for gp in group_params], axis=0) for i in range(len(group_params[0]))]
                for t in group:
                    for param, new in zip(models[t].parameters(), mean_params):
                        param.data = torch.tensor(new, dtype=param.data.dtype, device=param.data.device)


        elif method == 'fedem':
            # EM混合建模
            weights = fedem_em(models)
            for k in range(num_tasks):
                agg = None
                for p in range(num_tasks):
                    if local_models[p] is not None:
                        w = local_models[p]
                    else:
                        w = [param.data.cpu().numpy().copy() for param in models[p].parameters()]
                    if agg is None:
                        agg = [weights[k, p] * w_i for w_i in w]
                    else:
                        agg = [a + weights[k, p] * w_i for a, w_i in zip(agg, w)]
                for param, new in zip(models[k].parameters(), agg):
                    param.data = torch.tensor(new, dtype=param.data.dtype, device=param.data.device)
        else:
            raise NotImplementedError(f"Unknown baseline: {method}")

        # 评估
        if rnd % 10 == 0 or rnd == config.num_rounds - 1:
            accs = evaluate(models, test_loaders, device)
            print(f"Round {rnd}: mean acc={np.mean(accs):.4f}, per-task acc={accs}")
