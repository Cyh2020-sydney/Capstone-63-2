import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset


# 定义包含LoRA的改进神经网络
class SimpleNNWithLoRA(nn.Module):
    def __init__(self, rank_1=160, rank_2=100):
        super(SimpleNNWithLoRA, self).__init__()
        self.fc1 = LoRA(28 * 28, 200, rank_1)
        self.fc2 = LoRA(200, 200, rank_2)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义LoRA模块，使用Xavier初始化
class LoRA(nn.Module):
    def __init__(self, input_dim, output_dim, rank=100):
        super(LoRA, self).__init__()
        self.rank = rank
        self.low_rank_a = nn.Parameter(torch.randn(input_dim, rank))
        self.low_rank_b = nn.Parameter(torch.randn(rank, output_dim))
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))

        nn.init.xavier_uniform_(self.low_rank_a)
        nn.init.xavier_uniform_(self.low_rank_b)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.matmul(x, self.weight + torch.matmul(self.low_rank_a, self.low_rank_b))


# 动态创建模型，根据客户端设置的 rank
def create_model(rank_1=160, rank_2=100):
    model = SimpleNNWithLoRA(rank_1=rank_1, rank_2=rank_2)
    return model

# 根据标签将训练集分配到不同客户端
def distribute_data_to_clients(train_dataset, num_clients=10):
    clients = [[] for _ in range(num_clients)]

    for label in range(num_clients):
        indices = []
        for i, target in enumerate(train_dataset.targets):
            if target == label:
                indices.append(i)

        num_splits = num_clients - label
        split_indices = torch.chunk(torch.tensor(indices), num_splits)

        for i, split in enumerate(split_indices):
            clients[label + i].extend(split.tolist())

    client_subsets = []
    for client_data in clients:
        client_subsets.append(Subset(train_dataset, client_data))

    for i, client_data in enumerate(clients):
        print(f"Client {i} has {len(client_data)} samples")

    return client_subsets


# 参数填充函数，用 0 填充较小 rank 的矩阵
def pad_matrix(matrix, target_shape):
    # Create a zero matrix with the target shape
    padded_matrix = torch.zeros(target_shape)
    # Copy the original matrix into the zero matrix, truncating or padding as necessary
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix


def pad_client_weights(client_weights, max_rank1, input_dim1, output_dim1, max_rank2, input_dim2, output_dim2):
    padded_weights = []
    for weights in client_weights:
        padded_weight = []

        # 第一层 LoRA 权重填充
        low_rank_a1, low_rank_b1, fc1_weight = weights[0], weights[1], weights[2]
        padded_low_rank_a1 = pad_matrix(low_rank_a1, (input_dim1, max_rank1))  # 使用第一层的 max_rank1
        padded_low_rank_b1 = pad_matrix(low_rank_b1, (max_rank1, output_dim1))
        padded_weight.extend([padded_low_rank_a1, padded_low_rank_b1, fc1_weight])

        # 第二层 LoRA 权重填充
        low_rank_a2, low_rank_b2, fc2_weight = weights[3], weights[4], weights[5]
        padded_low_rank_a2 = pad_matrix(low_rank_a2, (input_dim2, max_rank2))  # 使用第二层的 max_rank2
        padded_low_rank_b2 = pad_matrix(low_rank_b2, (max_rank2, output_dim2))
        padded_weight.extend([padded_low_rank_a2, padded_low_rank_b2, fc2_weight])

        # 添加其他权重（例如 fc3）
        padded_weight.extend(weights[6:])
        padded_weights.append(padded_weight)
    return padded_weights


# 根据客户端的rank不同截断权重
# 获取LoRA层的low_rank_a和low_rank_b，并根据max_rank进行填充
# 处理后续层


def train(model, data_loader, criterion, optimizer, epochs): # fisher_information, old_params, lambda_ewc, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets) #+ ewc_loss(model, fisher_information, old_params, lambda_ewc)
            loss.backward()
            optimizer.step()

# 联邦加权平均函数，考虑各客户端的数据量
def federated_weighted_avg(weights, num_samples):
    avg_weights = []
    total_samples = sum(num_samples)
    for i in range(len(weights[0])):
        weighted_sum = sum(weights[j][i] * num_samples[j] / total_samples for j in range(len(weights)))
        avg_weights.append(weighted_sum)
    return avg_weights


# 定义截取参数的函数，确保根据客户端rank截取全局模型的权重
def load_truncated_state_dict(model, global_state_dict):
    # Get the current state dict of the model
    model_state_dict = model.state_dict()

    # Iterate through each parameter in the global state dict
    for name, param in global_state_dict.items():
        if name in model_state_dict:
            # Get the corresponding parameter in the model's state dict
            current_param = model_state_dict[name]

            # For low_rank_a and low_rank_b, dynamically load based on the rank
            if "low_rank_a" in name or "low_rank_b" in name:
                # If the global model's rank is larger than the client's, load the relevant parts
                if param.shape != current_param.shape:
                    if param.shape[0] > current_param.shape[0]:  # Check the first dimension (rows)
                        model_state_dict[name].copy_(param[:current_param.shape[0], :current_param.shape[1]])
                    elif param.shape[1] > current_param.shape[1]:  # Check the second dimension (columns)
                        model_state_dict[name].copy_(param[:, :current_param.shape[1]])
                    else:
                        model_state_dict[name].copy_(param)
            else:
                # For other parameters, directly load them
                model_state_dict[name].copy_(param)

    # Load the model's weights
    model.load_state_dict(model_state_dict, strict=False)


def aggregate_share_weighted(client_weights, num_samples):
    num_clients = len(client_weights)
    num_params_per_client = len(client_weights[0])

    # 将样本数量转换为张量，用于加权平均
    sample_weights = torch.tensor(num_samples, dtype=torch.float32)
    total_samples = torch.sum(sample_weights)  # 计算总样本数

    aggregated_weights = [None] * num_params_per_client

    for param_index in range(num_params_per_client):
        if param_index in {0,3}:  # 低秩矩阵a，按列聚合
            max_cols = max(weight[param_index].shape[1] for weight in client_weights)
            sum_matrix = torch.zeros((client_weights[0][param_index].shape[0], max_cols))
            count_matrix = torch.zeros((client_weights[0][param_index].shape[0], max_cols))

            for client_index in range(num_clients):
                client_matrix = client_weights[client_index][param_index]
                cols = client_matrix.shape[1]
                sum_matrix[:, :cols] += client_matrix # * sample_weights[client_index]
                count_matrix[:, :cols] += 1 # sample_weights[client_index]

            aggregated_weights[param_index] = sum_matrix / count_matrix

        elif param_index in {1,4}:  # 低秩矩阵b，按行聚合
            max_rows = max(weight[param_index].shape[0] for weight in client_weights)
            sum_matrix = torch.zeros((max_rows, client_weights[0][param_index].shape[1]))
            count_matrix = torch.zeros((max_rows, client_weights[0][param_index].shape[1]))

            for client_index in range(num_clients):
                client_matrix = client_weights[client_index][param_index]
                rows = client_matrix.shape[0]
                sum_matrix[:rows, :] += client_matrix # * sample_weights[client_index]
                count_matrix[:rows, :] += 1 # sample_weights[client_index]

            aggregated_weights[param_index] = sum_matrix / count_matrix


        else:  # 其他层，使用加权平均聚合
            weighted_sum = sum(weight[param_index] * sample_weights[i] for i, weight in enumerate(client_weights))
            aggregated_weights[param_index] = weighted_sum / total_samples

        # 计算平均值
        #if param_index in {0,3} or param_index in {1,4}:
        #    aggregated_weights[param_index] = sum_matrix / count_matrix
        #else:
        #    aggregated_weights[param_index] = weighted_sum / total_samples

    return aggregated_weights



# 联邦学习主循环函数
def federated_learning(global_model, client_subsets, criterion, hyperparams, test_loader=None):
    num_rounds = hyperparams.get('num_rounds', 3)
    learning_rate = hyperparams.get('learning_rate', 0.05)
    batch_size = hyperparams.get('batch_size', 64)
    epochs_per_client = hyperparams.get('epochs_per_client', 5)
    rank_1 = hyperparams.get('rank_1', 160)
    rank_2 = hyperparams.get('rank_2', 100)

    accuracy_history = []
    input_dim_1, output_dim_1 = 28 * 28, 200
    input_dim_2, output_dim_2 = 200, 200

    for round_num in range(num_rounds):
        client_weights = []
        num_samples = []
        max_rank_1 = rank_1
        max_rank_2 = rank_2

        # 遍历每个客户端的数据子集
        for index, client_data in enumerate(client_subsets):
            client_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)
            # Initialize an empty set to collect unique labels
            unique_labels = set()
            # Iterate through the DataLoader to collect labels
            for _, labels in client_loader:
                unique_labels.update(labels.tolist())
            # Get the number of unique labels
            num_labels = len(unique_labels)
            # Dynamically calculate the rank for the client based on the number of labels
            client_rank_1 = int((num_labels/10)*rank_1)
            client_rank_2 = int((num_labels/10)*rank_2)
            model = create_model(rank_1=client_rank_1, rank_2=client_rank_2)
            #model.load_state_dict(global_model.state_dict())
            load_truncated_state_dict(model, global_model.state_dict())

            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            train(model, client_loader, criterion, optimizer, epochs=epochs_per_client)

            # 保存客户端模型的权重
            model_weights = [param.data.clone() for param in model.parameters()]
            client_weights.append(model_weights)
            num_samples.append(len(client_data))

        #使用加权平均聚合权重并更新全局模型
        new_weights = aggregate_share_weighted(client_weights, num_samples)
        state_dict = global_model.state_dict()
        new_state_dict = {key: value for key, value in zip(state_dict.keys(), new_weights)}
        global_model.load_state_dict(new_state_dict)
        # 在测试集上评估全局模型并记录准确率
        if test_loader is not None:
            accuracy = test(global_model, test_loader)
            accuracy_history.append(accuracy)
            print(f'Round {round_num + 1} Test Accuracy: {accuracy * 100:.2f} %')

    if test_loader is not None and accuracy_history:
        plot_accuracy_history(accuracy_history)

    final_accuracy = accuracy_history[-1] if accuracy_history else None

    return accuracy_history, final_accuracy

    #return accuracy_history, accuracy_history[-1] if accuracy_history else None


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


def plot_accuracy_history(accuracy_history):
    """
    绘制测试准确率随联邦学习轮次的变化图。

    参数:
    - accuracy_history: 准确率历史列表。
    """
    x_values = list(range(1, len(accuracy_history) + 1))
    y_values = [acc * 100 for acc in accuracy_history]
    plt.plot(x_values, y_values)
    plt.xlabel('Federated Learning Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Federated Learning Rounds')
    plt.grid(True)
    plt.show()

    final_accuracy = accuracy_history[-1]
    print(f'Final Test Accuracy: {final_accuracy * 100:.2f} %')

transform = transforms.Compose([
    transforms.ToTensor(),
])
# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 调用函数
client_subsets = distribute_data_to_clients(train_dataset, num_clients=10)
# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
# 记录每轮的准确率
#accuracy_history = []

# 调用 federated_learning 函数时传入超参数字典
hyperparams = {
    'num_rounds': 50,
    'rank_1': 160,
    'rank_2': 100,
    'learning_rate': 0.05,
    'batch_size': 64,
    'epochs_per_client': 5
}

accuracy_history, final_accuracy = federated_learning(
    global_model=create_model(),
    client_subsets=client_subsets,
    criterion=nn.CrossEntropyLoss(),
    hyperparams=hyperparams,
    test_loader=test_loader
)

