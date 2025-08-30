import torch
import numpy as np
from .models import get_mnist_model
from .utils import evaluate

def init_models(dataset, config, input_shape, num_classes):
    if dataset == 'mnist':
        return [get_mnist_model(num_classes).to(config.device) for _ in range(config.num_tasks)]

    raise NotImplementedError

def cosine_similarity(w1, w2, eps=1e-6):
    w1 = w1.flatten()
    w2 = w2.flatten()
    return float(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + eps))

def train(train_loaders, test_loaders, models, config, client_task_map):
    num_tasks = config.num_tasks
    # Initialize Beta distribution parameters for the task similarity matrix.
    # We use a symmetric prior Beta(alpha, beta) for all task pairs.
    alpha = np.full((num_tasks, num_tasks), config.alpha, dtype=np.float32)
    beta = np.full((num_tasks, num_tasks), config.beta, dtype=np.float32)
    m = config.m
    device = config.device

    for rnd in range(config.num_rounds):
        # 1. Client Selection and Local Training Phase
        selected = np.random.choice(len(train_loaders), int(config.clients_per_round * len(train_loaders)), replace=False)
        local_models = [None] * num_tasks # Stores updated parameters from each task's clients
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
            # After local training, save the updated parameters
            local_models[task] = [p.data.cpu().numpy().copy() for p in model.parameters()]

        # 2. Server Aggregation and Bayesian Update Phase
        # If a task wasn't updated this round, use its previous global parameters
        global_params = []
        for t in range(num_tasks):
            if local_models[t] is not None:
                global_params.append(local_models[t])
            else:
                global_params.append([p.data.cpu().numpy().copy() for p in models[t].parameters()])
        # 3. Core Bayesian Update: Update alpha and beta for each task pair
        for k in range(num_tasks):
            for p in range(num_tasks):
                w_k = np.concatenate([w.flatten() for w in global_params[k]])
                w_p = np.concatenate([w.flatten() for w in global_params[p]])
                cos_sim = cosine_similarity(w_k, w_p)
                # Update Beta posterior parameters using conjugacy.
                # The observation cos_sim is treated as "success" count (weighted by m),
                # and (1 - cos_sim) as "failure" count.
                alpha[k, p] += cos_sim * m
                beta[k, p] += (1 - cos_sim) * m
        omega = alpha / (alpha + beta)
        # 4. Weighted Aggregation using the updated similarity matrix
        for k in range(num_tasks):
            agg = None
            for p in range(num_tasks):
                w_p = global_params[p]
                if agg is None:
                    agg = [omega[k, p] * w for w in w_p]
                else:
                    agg = [a + omega[k, p] * w for a, w in zip(agg, w_p)]

            for param, new in zip(models[k].parameters(), agg):
                param.data = torch.tensor(new, dtype=param.data.dtype, device=param.data.device)

        if rnd % 10 == 0 or rnd == config.num_rounds - 1:
            accs = evaluate(models, test_loaders, device)
            print(f"Round {rnd}: mean acc={np.mean(accs):.4f}, per-task acc={accs}")
