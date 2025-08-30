import torch
from torchvision import datasets, transforms
import numpy as np
import os

def mnist_noniid_split(dataset, num_clients, alpha):
    labels = np.array(dataset.targets)
    num_classes = 10
    idxs = [np.where(labels == i)[0] for i in range(num_classes)]
    client_idxs = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        np.random.shuffle(idxs[c])
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idxs[c])).astype(int)[:-1]
        split = np.split(idxs[c], proportions)
        for i, idx in enumerate(split):
            client_idxs[i].extend(idx)
    return client_idxs

def celeba_noniid_split(trainset, num_clients, alpha, tasks_per_client=3):
    # Dirichlet分布划分CelebA
    idxs = np.arange(len(trainset))
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
    splits = np.split(idxs, proportions)
    return splits

def load_data(dataset, config):
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
        testset = datasets.MNIST('data/mnist', train=False, download=True, transform=transform)
        client_idxs = mnist_noniid_split(trainset, config.num_clients, config.dirichlet_alpha)
        train_loaders = []
        test_loaders = []
        client_task_map = []
        for i, idxs in enumerate(client_idxs):
            subset = torch.utils.data.Subset(trainset, idxs)
            loader = torch.utils.data.DataLoader(subset, batch_size=config.batch_size, shuffle=True)
            train_loaders.append(loader)

            task = int(np.argmax(np.bincount(np.array(trainset.targets)[idxs])))
            client_task_map.append([task])

            test_idxs = np.where(np.array(testset.targets) == task)[0]
            test_subset = torch.utils.data.Subset(testset, test_idxs)
            test_loader = torch.utils.data.DataLoader(test_subset, batch_size=128, shuffle=False)
            test_loaders.append(test_loader)
        input_shape = (1, 28, 28)
        num_classes = 10
        return train_loaders, test_loaders, client_task_map, input_shape, num_classes

    elif dataset == 'celeba':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = datasets.CelebA('data/celeba', split='train', download=True, transform=transform)
        testset = datasets.CelebA('data/celeba', split='test', download=True, transform=transform)
        num_clients = config.num_clients
        alpha = config.dirichlet_alpha

        # Dirichlet分布划分
        client_idxs = celeba_noniid_split(trainset, num_clients, alpha)
        train_loaders = []
        test_loaders = []
        client_task_map = []
        all_tasks = list(range(40))  # 40个属性任务

        for i, idxs in enumerate(client_idxs):
            subset = torch.utils.data.Subset(trainset, idxs)
            loader = torch.utils.data.DataLoader(subset, batch_size=config.batch_size, shuffle=True)
            train_loaders.append(loader)
            num_tasks = np.random.randint(3, 6)
            client_tasks = np.random.choice(all_tasks, num_tasks, replace=False)
            client_task_map.append(client_tasks.tolist())

            test_mask = np.zeros(40, dtype=bool)
            test_mask[client_tasks] = True
            test_idxs = np.where(np.any(testset.attr[:, test_mask], axis=1))[0]
            test_subset = torch.utils.data.Subset(testset, test_idxs)
            test_loader = torch.utils.data.DataLoader(test_subset, batch_size=128, shuffle=False)
            test_loaders.append(test_loader)
        input_shape = (3, 224, 224)
        num_classes = 40
        return train_loaders, test_loaders, client_task_map, input_shape, num_classes

    elif dataset == 'synthetic':
        # 生成合成数据
        num_clients = 20
        num_tasks = 4
        samples_per_client = 500
        mu = np.array([
            [np.cos(2 * np.pi * k / 4), np.sin(2 * np.pi * k / 4), 0] +
            np.random.uniform(-0.2, 0.2, 3)
            for k in range(num_tasks)
        ])
        I = np.eye(3)
        E = np.ones((3, 3))
        Sigma = [
            0.5 * I + 0.5 * E if k % 2 == 0 else I + 0.2 * E
            for k in range(num_tasks)
        ]
        Omega_true = np.zeros((num_tasks, num_tasks))
        for k in range(num_tasks):
            for p in range(num_tasks):
                diff = mu[k] - mu[p]
                cov = (Sigma[k] + Sigma[p]) / 2
                mahalanobis = diff @ np.linalg.inv(cov) @ diff.T
                Omega_true[k, p] = np.exp(-mahalanobis / 2)
        clients_per_task = num_clients // num_tasks
        client_data = []
        client_task_map = []
        for task_id in range(num_tasks):
            for client_id in range(task_id * clients_per_task, (task_id + 1) * clients_per_task):
                data = np.random.multivariate_normal(
                    mu[task_id],
                    Sigma[task_id],
                    samples_per_client
                )
                labels = np.full(samples_per_client, task_id)
                client_data.append((data, labels))
                client_task_map.append([task_id])
        # 转为TensorDataset和DataLoader
        train_loaders = []
        test_loaders = []
        for data, labels in client_data:
            tensor_x = torch.tensor(data, dtype=torch.float32)
            tensor_y = torch.tensor(labels, dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
            train_loaders.append(loader)

            test_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
            test_loaders.append(test_loader)
        input_shape = (3,)
        num_classes = num_tasks
        return train_loaders, test_loaders, client_task_map, input_shape, num_classes

    elif dataset == 'ophthalmic':
        DISEASES = ['HMM', 'RVO', 'PR', 'DME', 'PM', 'HR', 'G', 'MEM', 'MH', 'FE', 'AMD', 'DR']
        tasks = []
        for disease in DISEASES[:10]:
            tasks.append(f"{disease}_classification")
        tasks += ["AMD_grading", "DR_grading"]
        num_clients = 80
        client_task_map = []
        train_loaders = []
        test_loaders = []
        input_shape = (3, 224, 224)
        num_classes = 2  # 二分类，分级任务可单独处理
        for i in range(num_clients):
            num_tasks = np.random.randint(1, 3)
            client_tasks = np.random.choice(tasks, num_tasks, replace=False)
            client_task_map.append(client_tasks.tolist())

            all_imgs = []
            all_labels = []
            for task in client_tasks:
                img_dir = os.path.join('data/ophthalmic', task, 'images')
                label_path = os.path.join('data/ophthalmic', task, 'labels.csv')
                if not os.path.exists(img_dir) or not os.path.exists(label_path):
                    continue
                # 读取图片和标签
                import pandas as pd
                from PIL import Image
                df = pd.read_csv(label_path)
                for idx, row in df.iterrows():
                    img_path = os.path.join(img_dir, row['filename'])
                    if not os.path.exists(img_path):
                        continue
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((224, 224))
                    img = np.array(img) / 255.0
                    all_imgs.append(img.transpose(2, 0, 1))  # C,H,W
                    all_labels.append(row['label'])
            if len(all_imgs) == 0:
                # 跳过无数据的client
                continue
            tensor_x = torch.tensor(np.array(all_imgs), dtype=torch.float32)
            tensor_y = torch.tensor(np.array(all_labels), dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
            train_loaders.append(loader)

            test_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
            test_loaders.append(test_loader)
        return train_loaders, test_loaders, client_task_map, input_shape, num_classes

    else:
        raise NotImplementedError("Unknown dataset: {}".format(dataset))