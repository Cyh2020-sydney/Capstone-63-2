import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from collections import Counter
import csv
import time
from datetime import datetime
import random
import copy
random.seed(42)
# 定义包含LoRA的改进神经网络
# 定义LoRA模块，使用Xavier初始化
# 定义 LoRA 层
class LoRA(nn.Module):
    def __init__(self, input_dim, output_dim, rank, name_prefix):
        super(LoRA, self).__init__()
        self.rank = rank
        # 使用 name_prefix 来区分参数名称
        self.low_rank_a = nn.Parameter(torch.randn(input_dim, rank), requires_grad=True)
        self.low_rank_b = nn.Parameter(torch.randn(rank, output_dim), requires_grad=True)
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=True)

        # 初始化权重
        nn.init.xavier_uniform_(self.low_rank_a)
        nn.init.xavier_uniform_(self.low_rank_b)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # 计算输出
        return torch.matmul(x, self.weight + torch.matmul(self.low_rank_a, self.low_rank_b))

# 带有 LoRA 的神经网络模型
class SimpleNNWithLoRA(nn.Module):
    def __init__(self, rank_1=160, rank_2=100):
        super(SimpleNNWithLoRA, self).__init__()
        # 在 LoRA 中使用 rank_1 和 rank_2
        self.fc1 = LoRA(28 * 28, 200, rank_1, name_prefix='rank_1')
        self.fc2 = LoRA(200, 200, rank_2, name_prefix='rank_2')
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 简单的标准神经网络模型（无 LoRA）
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建带有 LoRA 的模型
def create_lora_model(rank_1=160, rank_2=100):
    # 使用传入的 rank_1 和 rank_2 创建模型
    model = SimpleNNWithLoRA(rank_1=rank_1, rank_2=rank_2)
    return model

# 创建标准模型
def create_model():
    model = SimpleNN()
    return model

def model_creator(rank_1,rank_2, use_lora):
    """
    根据超参数选择创建 LoRA 模型或标准模型。

    参数:
    - rank: LoRA 的秩，仅在 use_lora 为 True 时需要。
    - use_lora: 是否使用 LoRA。

    返回:
    - 创建的模型实例。
    """
    if use_lora:
        return create_lora_model(rank_1=rank_1,rank_2=rank_2)
    else:
        return create_model()  # 不传递 rank 给 create_model
# EWC相关函数
def compute_fisher_information(model, data_loader, criterion):
    fisher_information = {n: torch.zeros_like(p) for n, p in \
                          model.named_parameters() if p.requires_grad}
    model.eval()
    total_samples = 0
    for data, targets in data_loader:
        total_samples += len(data)
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()  # 计算梯度
        epsilon=1e-20
        for n, p in model.named_parameters():
            if p.requires_grad:
                fisher_information[n] += torch.log(p.grad.data ** \
                                                   2 + epsilon) / total_samples
        model.zero_grad()
    return fisher_information
# 定义 EWC 损失函数
def ewc_loss(model, fisher_information, old_params, lambda_ewc):
    loss_ewc = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            penalty=(fisher_information[n] * (p - old_params[n]) ** 2).sum()
            loss_ewc += lambda_ewc * penalty
    return loss_ewc
# 创建模型

def distribute_data_to_clients(train_dataset, num_clients=10, distribution_mode='standard'):
    # 统计数据集原本的每个标签的数量
    original_label_counts = Counter(train_dataset.targets.tolist())
    print("Original dataset label counts:")
    for label, count in original_label_counts.items():
        print(f"Label {label}: {count} samples")

    clients = [[] for _ in range(num_clients)]

    if distribution_mode == 'standard':
        for label in range(num_clients):
            indices = [i for i, target in enumerate(train_dataset.targets) if target == label]
            num_splits = num_clients - label
            split_indices = torch.chunk(torch.tensor(indices), num_splits)
            for i in range(num_splits):
                clients[label + i].extend(split_indices[i].tolist())
    elif distribution_mode == 'label_based':
        for label in range(num_clients):
            indices = [i for i, target in enumerate(train_dataset.targets) if target == label]
            clients[label].extend(indices)
    else:
        raise ValueError(f"Unknown distribution_mode: {distribution_mode}")

    client_subsets = [Subset(train_dataset, client_data) for client_data in clients]

    # 输出每个客户端的每个标签的数量
    for i, client_data in enumerate(client_subsets):
        client_targets = [train_dataset.targets[idx] for idx in client_data.indices]
        client_label_counts = Counter(client_targets)

        print(f"Client {i} label counts:")
        l = [0,0,0,0,0,0,0,0,0,0]
        for label, count in client_label_counts.items():
            l[label] += count
        print(l)

    return client_subsets

# 训练函数
def train(model, data_loader, criterion, optimizer, fisher_information, old_params, lambda_ewc, epochs, use_ewc,
          use_lora):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)

            # 添加 EWC 损失（如果启用）
            if use_ewc:
                loss += ewc_loss(model, fisher_information, old_params, lambda_ewc)

            loss.backward()
            optimizer.step()


# 联邦加权平均函数，考虑各客户端的数据量
def federated_weighted_avg(weights, num_samples,use_zeropadding):
    #print('展示最初参数：')
    #for t in weights[0]:
        #print(t.shape)
    if use_zeropadding is not True:
        avg_weights = []
        total_samples = sum(num_samples)
        for i in range(len(weights[0])):
            weighted_sum = sum(weights[j][i] * num_samples[j] / total_samples for j in range(len(weights)))
            avg_weights.append(weighted_sum)
        return avg_weights
    else:
        avg_weights = []
        max_rank_1 = 0#
        max_rank_2 = 0#
        for l in weights:
            t0 = l[0]
            t1 = l[1]
            t2 = l[3]#
            t3 = l[4]#
            #print("t?.shape",t0.shape, t1.shape, t2.shape, t3.shape)
            dim_1, dim_rank_1 = t0.shape#
            dim_rank_1, dim_2 = t1.shape#
            dim_3, dim_rank_2 = t2.shape#
            dim_rank_2, dim_4 = t3.shape#
            if dim_rank_1 > max_rank_1:
                max_rank_1 = dim_rank_1
            if dim_rank_2 > max_rank_2:
                max_rank_2 = dim_rank_2
        i = 0
        new_shape_0 = (dim_1, max_rank_1)
        new_shape_1 = (max_rank_1, dim_2)
        new_shape_2 = (dim_3, max_rank_2)
        new_shape_3 = (max_rank_2, dim_4)
        while i < len(weights):
            weights[i][0] = resize_tensor(weights[i][0], new_shape_0)
            weights[i][1] = resize_tensor(weights[i][1], new_shape_1)
            weights[i][3] = resize_tensor(weights[i][3], new_shape_2)
            weights[i][4] = resize_tensor(weights[i][4], new_shape_3)
            #print("weight_i_?.shape", weights[i][0].shape, weights[i][1].shape, weights[i][3].shape, weights[i][4].shape)
            i += 1
        total_samples = sum(num_samples)
        #print("total_samples:",total_samples)
        for i in range(len(weights[0])):
            weighted_sum = 0
            #print("i:",i)
            for j in range(len(weights)):
                #print("j:",j,weights[j][i].shape)
                #print("num_samples[j]",num_samples[j])
                #print("weights[j][i]]", weights[j][i])
                weighted_sum += weights[j][i] * num_samples[j] / total_samples
            #print("weighted_sum:",weighted_sum)
            avg_weights.append(weighted_sum)
        return avg_weights



def federated_learning(global_model, client_subsets, criterion, hyperparams, test_loader=None):
    """
    联邦学习主循环函数，包含 EWC 和 LoRA 算法的实现（可选）。

    参数:
    - global_model: 全局模型。
    - client_subsets: 每个客户端的数据子集列表。
    - criterion: 损失函数。
    - hyperparams: 超参数字典，包含 num_rounds、rank、learning_rate、batch_size、epochs_per_client、lambda_ewc、use_ewc 和 use_lora。
    - test_loader: 测试数据加载器。

    返回:
    - accuracy_history: 每轮测试的准确率列表。
    - final_accuracy: 最终全局模型的准确率。
    """
    num_rounds = hyperparams.get('num_rounds', 3)
    rank_1 = hyperparams.get('rank_1', 160)
    rank_2 = hyperparams.get('rank_2', 100)
    learning_rate = hyperparams.get('learning_rate', 0.05)
    batch_size = hyperparams.get('batch_size', 64)
    epochs_per_client = hyperparams.get('epochs_per_client', 5)
    lambda_ewc = hyperparams.get('lambda_ewc', 0.4)
    use_ewc = hyperparams.get('use_ewc', True)
    use_lora = hyperparams.get('use_lora', False)
    use_zeropadding= hyperparams.get('use_zeropadding', False)
    accuracy_history = []
    if use_zeropadding == False:
        for round_num in range(num_rounds):
            client_weights = []
            num_samples = []

            # 遍历每个客户端的数据子集
            for client_data in client_subsets:
                client_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)

                # 创建模型时根据 use_lora 是否使用 LoRA
                if use_lora:
                    model = create_lora_model(rank_1=rank_1, rank_2=rank_2)
                else:
                    model = create_model()

                model.load_state_dict(global_model.state_dict())

                # 如果使用 EWC，计算 Fisher 信息矩阵和旧参数
                if use_ewc:
                    fisher_information = compute_fisher_information(model, client_loader, criterion)
                    old_params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
                else:
                    fisher_information = None
                    old_params = None

                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                train(model, client_loader, criterion, optimizer, fisher_information, old_params, lambda_ewc,
                      epochs=epochs_per_client, use_ewc=use_ewc, use_lora=use_lora)

                # 保存客户端模型的权重
                model_weights = [param.data.clone() for param in model.parameters()]
                client_weights.append(model_weights)
                num_samples.append(len(client_data))

            # 使用加权平均聚合权重并更新全局模型
            new_weights = federated_weighted_avg(client_weights, num_samples,use_zeropadding=use_zeropadding)
            state_dict = global_model.state_dict()
            new_state_dict = {key: value for key, value in zip(state_dict.keys(), new_weights)}
            global_model.load_state_dict(new_state_dict)

            # 在测试集上评估全局模型并记录准确率
            if test_loader is not None:
                accuracy = test(global_model, test_loader)
                accuracy_history.append(accuracy)
                print(f'Round {round_num + 1} Test Accuracy: {accuracy * 100:.2f} %')
    else:

        # 设置超参数
        # 基准秩160（第一层），100，每个label对应0.1的系数
        # 使用列表推导将每个元素转换为整数
        ranks_1 = [int(rank_1 * i) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        ranks_2 = [int(rank_2 * i) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]

        learning_rate = 0.05
        batch_size = 64
        epochs_per_client = 5
        global_models = []
        global_model = create_lora_model(rank_1=rank_1,rank_2=rank_2)
        i = 0
        while i < 10:  # 后面把这个数改成client的个数
            global_models.append(create_lora_model(rank_1=ranks_1[i],rank_2=ranks_2[i]))
            i += 1
        criterion = nn.CrossEntropyLoss()

        # 记录每轮的准确率
        accuracy_history = []

        # 第一轮
        client_weights = []
        num_samples = []
        y = 0
        while y < len(client_subsets):
            client_data = client_subsets[y]
            client_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)



            if use_lora:
                model = create_lora_model(rank_1=ranks_1[y],rank_2=ranks_2[y])
            else:
                print("If u choose zero_padding True,u should use lora")
            # 如果使用 EWC，计算 Fisher 信息矩阵和旧参数
            if use_ewc:
                fisher_information = compute_fisher_information(model, client_loader, criterion)
                old_params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
            else:
                fisher_information = None
                old_params = None

            # print(global_model.state_dict())
            model.load_state_dict(global_models[y].state_dict())  # 加载全局模型的权重
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            train(model, client_loader, criterion, optimizer, fisher_information, old_params, lambda_ewc,
                  epochs=epochs_per_client, use_ewc=use_ewc, use_lora=use_lora)
            model_weights = [param.data.clone() for param in model.parameters()]  # 保存客户端模型的权重副本
            # 这里已经形状不对了
            client_weights.append(model_weights)
            num_samples.append(len(client_data))  # 记录客户端的数据样本数量
            y += 1
        # 聚合权重，使用加权平均
        new_weights = federated_weighted_avg(client_weights, num_samples,use_zeropadding=use_zeropadding)
        # 在测试集上评估全局模型并记录准确率
        accuracy = test(global_model, test_loader)
        accuracy_history.append(accuracy)
        print('Round', 1, 'Test Accuracy:', accuracy * 100, '%')

        # 联邦学习主循环
        for round_num in range(1, num_rounds):
            client_weights = []
            num_samples = []
            y = 0
            while y < len(client_subsets):
                new_weights_client = copy.deepcopy(new_weights)

                for t in new_weights:
                    print(t.shape)

                dim_r1, dim_c1 = new_weights[0].shape
                dim_r2, dim_c2 = new_weights[1].shape
                dim_r3, dim_c3 = new_weights[3].shape
                dim_r4, dim_c4 = new_weights[4].shape
                print("dim1:",dim_r1, dim_c1)
                print("dim2:", dim_r2, dim_c2)
                print("dim3:", dim_r3, dim_c3)
                print("dim4:", dim_r4, dim_c4)
                new_weights_client[0] = clip_tensor(new_weights[0], (dim_r1, ranks_1[y]))
                new_weights_client[1] = clip_tensor(new_weights[1], (ranks_1[y], dim_c2))
                new_weights_client[3] = clip_tensor(new_weights[3], (dim_r3, ranks_2[y]))
                new_weights_client[4] = clip_tensor(new_weights[4], (ranks_2[y], dim_c4))
                state_dict = global_models[y].state_dict()
                new_state_dict = {key: value for key, value in zip(state_dict.keys(), new_weights_client)}
                '''
                print('看新参数')
                for t in new_weights_client:
                    print(t.shape)
                '''
                global_models[y].load_state_dict(new_state_dict)

                client_data = client_subsets[y]
                client_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)
                model = create_lora_model(rank_1=ranks_1[y],rank_2=ranks_2[y])
                model.load_state_dict(global_models[y].state_dict())  # 加载全局模型的权重
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                train(model, client_loader, criterion, optimizer, fisher_information, old_params, lambda_ewc,
                      epochs=epochs_per_client, use_ewc=use_ewc, use_lora=use_lora)
                model_weights = [param.data.clone() for param in model.parameters()]  # 保存客户端模型的权重副本
                client_weights.append(model_weights)
                num_samples.append(len(client_data))  # 记录客户端的数据样本数量
                y += 1

            # 聚合权重，使用加权平均
            new_weights = federated_weighted_avg(client_weights, num_samples,use_zeropadding=use_zeropadding)
            # 将聚合后的新权重应用到 global_model
            state_dict = global_model.state_dict()
            new_state_dict = {key: value for key, value in zip(state_dict.keys(), new_weights)}
            global_model.load_state_dict(new_state_dict)
            # 在测试集上评估全局模型并记录准确率
            accuracy = test(global_model, test_loader)
            accuracy_history.append(accuracy)
            print('Round', round_num + 1, 'Test Accuracy:', accuracy * 100, '%')

    # 如果提供了测试加载器，则绘制准确率与联邦学习轮次的关系图
    if test_loader is not None and accuracy_history:
        plot_accuracy_history(accuracy_history)

    final_accuracy = accuracy_history[-1] if accuracy_history else None

    return accuracy_history, final_accuracy



# 测试函数
def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in data_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return accuracy
def save_accuracy_to_csv(accuracy_history, filename):
    # 将准确度写入 CSV 文件
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Test Accuracy"])  # 写入表头
        for i, accuracy in enumerate(accuracy_history, start=1):
            writer.writerow([i, accuracy * 100])  # 将准确度转换为百分比格式保存

    print(f"Accuracy history saved to {filename}")

def plot_accuracy_history(accuracy_history):
    return 0
def clip_tensor(tensor,new_shape):
    dim_row, dim_col = new_shape
    new_tensor = torch.zeros(new_shape, dtype=tensor.dtype)
    for i in range(dim_row):
        for j in range(dim_col):
            new_tensor[i,j] = tensor[i,j]
    return new_tensor


def resize_tensor(tensor, new_shape):
    # 获取当前张量的形状
    dim_row, dim_col = tensor.shape
    # 创建一个新的张量，形状为 (当前行数, 当前列数 + 1)，并用0填充
    new_tensor = torch.zeros(new_shape, dtype=tensor.dtype)
    # 将原始张量的内容复制到新张量中
    for i in range(dim_row):
        for j in range(dim_col):
            new_tensor[i,j] = tensor[i,j]
    return new_tensor

if __name__ == "__main__":

    torch.manual_seed(42)
    random.seed(42)
    #1:43
    hyperparams_list = [
        {
            'num_rounds': 10,
            'rank_1': 160,
            'rank_2': 100,
            'learning_rate': 0.05,
            'batch_size': 64,
            'epochs_per_client': 5,
            'lambda_ewc': 0.5,
            'use_ewc': True,
            'use_lora': False,
            'use_zeropadding': False,
            'distribution_mode': 'standard'  # 数据分割方式
        },
        {
            'num_rounds': 10,
            'rank_1': 160,
            'rank_2': 100,
            'learning_rate': 0.05,
            'batch_size': 64,
            'epochs_per_client': 5,
            'lambda_ewc': 0.05,
            'use_ewc': False,
            'use_lora': False,
            'use_zeropadding': False,
            'distribution_mode': 'standard'  # 数据分割方式
        }
    ]

    # 变换和加载数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 记录每轮的准确率
    for i, hyperparams in enumerate(hyperparams_list):
        print(f"Running federated learning with hyperparameters set {i + 1}")

        # 根据当前的超参数配置生成客户端数据子集
        client_subsets = distribute_data_to_clients(train_dataset, num_clients=10,
                                                    distribution_mode=hyperparams['distribution_mode'])

        # 使用 model_creator 函数来创建模型
        global_model = model_creator(rank_1=hyperparams['rank_1'],rank_2=hyperparams['rank_2'],\
                                     use_lora=hyperparams['use_lora'])

        accuracy_history, final_accuracy = federated_learning(
            global_model=global_model,
            client_subsets=client_subsets,
            criterion=nn.CrossEntropyLoss(),
            hyperparams=hyperparams,
            test_loader=test_loader
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"accuracy_history_{i + 1}_{timestamp}.csv"
        save_accuracy_to_csv(accuracy_history, filename)