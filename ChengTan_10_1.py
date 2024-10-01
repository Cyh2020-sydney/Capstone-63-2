# share 带加权
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from collections import Counter
import copy
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)


# 定义LoRA模块，使用Xavier初始化
class LoRA(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(LoRA, self).__init__()
        self.rank = rank
        self.low_rank_a = nn.Parameter(torch.randn(input_dim, rank))
        self.low_rank_b = nn.Parameter(torch.randn(rank, output_dim))
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))

        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.low_rank_a)
        nn.init.xavier_uniform_(self.low_rank_b)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.matmul(x, self.weight + torch.matmul(self.low_rank_a, self.low_rank_b))


# 定义数据预处理步骤
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


# 根据标签将训练集分配到不同客户端
def distribute_data_to_clients(train_dataset, num_clients=10):
    # 统计数据集原本的每个标签的数量
    original_label_counts = Counter(train_dataset.targets.tolist())
    print("Original dataset label counts:")
    for label, count in original_label_counts.items():
        print(f"Label {label}: {count} samples")

    clients = [[] for _ in range(num_clients)]

    for label in range(num_clients):
        indices = [i for i, target in enumerate(train_dataset.targets) if target == label]
        num_splits = num_clients - label
        split_indices = torch.chunk(torch.tensor(indices), num_splits)
        for i in range(num_splits):
            clients[label + i].extend(split_indices[i].tolist())

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


client_subsets = distribute_data_to_clients(train_dataset)

# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 定义包含LoRA的改进神经网络
class SimpleNNWithLoRA(nn.Module):
    def __init__(self, rank1, rank2):
        super(SimpleNNWithLoRA, self).__init__()
        self.fc1 = LoRA(28 * 28, 200, rank1)
        self.fc2 = LoRA(200, 200, rank2)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建模型
def create_model(rank_1, rank_2):
    model = SimpleNNWithLoRA(rank1 = rank_1, rank2 = rank_2)
    return model


# 训练函数
def train(model, data_loader, criterion, optimizer, epochs):
    model.train()

    total_epoch_loss = 0
    avg_losses = []  # 用于存储每个 epoch 的平均损失
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch_idx, (data, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_losses.append(avg_loss)  # 存储当前 epoch 的平均损失
        total_epoch_loss += avg_loss

    # 计算每个 epoch 平均损失与总损失的比例
    p = [loss / total_epoch_loss for loss in avg_losses]

    # 计算熵值 E
    epochs_count = epochs
    normalization_factor = -1 / (1 / np.log(abs(epochs_count)))  # 归一化因子
    entropy_sum = 0  # 熵和的初始值
    for p_i in p:
        if p_i > 0:
            entropy_sum += p_i * np.log(p_i)

    entropy_value = normalization_factor * entropy_sum  # 最终熵值

    return entropy_value, p  # 返回熵值和 p 列表


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


def share_tensor(tensor, i, j):
    # i表示矩阵位置，j表示第几个客户端
    # ranks_1和2要改成排序后的了
    # 这里可能要重新排序一下，将每个客户端的rank系数拉一个表，然后把对应的num_sample加进来组成元组，然后根据系数从小到大排序。最后算total_samples的时候要按照这个表中的顺序来截取。
    # 这里的sorted_num_samples是按照rate的大小拍的，不是按照sample的大小拍的。
    total_samples_0 = sum(sorted_num_samples)
    total_samples_1 = sum(sorted_num_samples[1:])
    total_samples_2 = sum(sorted_num_samples[2:])
    total_samples_3 = sum(sorted_num_samples[3:])
    total_samples_4 = sum(sorted_num_samples[4:])
    total_samples_5 = sum(sorted_num_samples[5:])
    total_samples_6 = sum(sorted_num_samples[6:])
    total_samples_7 = sum(sorted_num_samples[7:])
    total_samples_8 = sum(sorted_num_samples[8:])
    total_samples_9 = sum(sorted_num_samples[9:])
    if i == 0:
        # 处理第一层第一个，rank作用在列
        tensor[:, 0:sorted_ranks_1[0]] = tensor[:, 0:sorted_ranks_1[0]] * num_samples[j] / total_samples_0
        tensor[:, sorted_ranks_1[0]:sorted_ranks_1[1]] = tensor[:, sorted_ranks_1[0]:sorted_ranks_1[1]] * num_samples[j] / total_samples_1
        tensor[:, sorted_ranks_1[1]:sorted_ranks_1[2]] = tensor[:, sorted_ranks_1[1]:sorted_ranks_1[2]] * num_samples[j] / total_samples_2
        tensor[:, sorted_ranks_1[2]:sorted_ranks_1[3]] = tensor[:, sorted_ranks_1[2]:sorted_ranks_1[3]] * num_samples[j] / total_samples_3
        tensor[:, sorted_ranks_1[3]:sorted_ranks_1[4]] = tensor[:, sorted_ranks_1[3]:sorted_ranks_1[4]] * num_samples[j] / total_samples_4
        tensor[:, sorted_ranks_1[4]:sorted_ranks_1[5]] = tensor[:, sorted_ranks_1[4]:sorted_ranks_1[5]] * num_samples[j] / total_samples_5
        tensor[:, sorted_ranks_1[5]:sorted_ranks_1[6]] = tensor[:, sorted_ranks_1[5]:sorted_ranks_1[6]] * num_samples[j] / total_samples_6
        tensor[:, sorted_ranks_1[6]:sorted_ranks_1[7]] = tensor[:, sorted_ranks_1[6]:sorted_ranks_1[7]] * num_samples[j] / total_samples_7
        tensor[:, sorted_ranks_1[7]:sorted_ranks_1[8]] = tensor[:, sorted_ranks_1[7]:sorted_ranks_1[8]] * num_samples[j] / total_samples_8
        tensor[:, sorted_ranks_1[8]:sorted_ranks_1[9]] = tensor[:, sorted_ranks_1[8]:sorted_ranks_1[9]] * num_samples[j] / total_samples_9
    elif i == 1:
        # 处理第一层第二个，rank作用在行
        tensor[0:sorted_ranks_1[0], :] = tensor[0:sorted_ranks_1[0], :] * num_samples[j] / total_samples_0
        tensor[sorted_ranks_1[0]:sorted_ranks_1[1], :] = tensor[sorted_ranks_1[0]:sorted_ranks_1[1], :] * num_samples[j] / total_samples_1
        tensor[sorted_ranks_1[1]:sorted_ranks_1[2], :] = tensor[sorted_ranks_1[1]:sorted_ranks_1[2], :] * num_samples[j] / total_samples_2
        tensor[sorted_ranks_1[2]:sorted_ranks_1[3], :] = tensor[sorted_ranks_1[2]:sorted_ranks_1[3], :] * num_samples[j] / total_samples_3
        tensor[sorted_ranks_1[3]:sorted_ranks_1[4], :] = tensor[sorted_ranks_1[3]:sorted_ranks_1[4], :] * num_samples[j] / total_samples_4
        tensor[sorted_ranks_1[4]:sorted_ranks_1[5], :] = tensor[sorted_ranks_1[4]:sorted_ranks_1[5], :] * num_samples[j] / total_samples_5
        tensor[sorted_ranks_1[5]:sorted_ranks_1[6], :] = tensor[sorted_ranks_1[5]:sorted_ranks_1[6], :] * num_samples[j] / total_samples_6
        tensor[sorted_ranks_1[6]:sorted_ranks_1[7], :] = tensor[sorted_ranks_1[6]:sorted_ranks_1[7], :] * num_samples[j] / total_samples_7
        tensor[sorted_ranks_1[7]:sorted_ranks_1[8], :] = tensor[sorted_ranks_1[7]:sorted_ranks_1[8], :] * num_samples[j] / total_samples_8
        tensor[sorted_ranks_1[8]:sorted_ranks_1[9], :] = tensor[sorted_ranks_1[8]:sorted_ranks_1[9], :] * num_samples[j] / total_samples_9
    elif i == 3:
        # 处理第二层第一个，rank作用在列
        tensor[:, 0:sorted_ranks_2[0]] = tensor[:, 0:sorted_ranks_2[0]] * num_samples[j] / total_samples_0
        tensor[:, sorted_ranks_2[0]:sorted_ranks_2[1]] = tensor[:, sorted_ranks_2[0]:sorted_ranks_2[1]] * num_samples[j] / total_samples_1
        tensor[:, sorted_ranks_2[1]:sorted_ranks_2[2]] = tensor[:, sorted_ranks_2[1]:sorted_ranks_2[2]] * num_samples[j] / total_samples_2
        tensor[:, sorted_ranks_2[2]:sorted_ranks_2[3]] = tensor[:, sorted_ranks_2[2]:sorted_ranks_2[3]] * num_samples[j] / total_samples_3
        tensor[:, sorted_ranks_2[3]:sorted_ranks_2[4]] = tensor[:, sorted_ranks_2[3]:sorted_ranks_2[4]] * num_samples[j] / total_samples_4
        tensor[:, sorted_ranks_2[4]:sorted_ranks_2[5]] = tensor[:, sorted_ranks_2[4]:sorted_ranks_2[5]] * num_samples[j] / total_samples_5
        tensor[:, sorted_ranks_2[5]:sorted_ranks_2[6]] = tensor[:, sorted_ranks_2[5]:sorted_ranks_2[6]] * num_samples[j] / total_samples_6
        tensor[:, sorted_ranks_2[6]:sorted_ranks_2[7]] = tensor[:, sorted_ranks_2[6]:sorted_ranks_2[7]] * num_samples[j] / total_samples_7
        tensor[:, sorted_ranks_2[7]:sorted_ranks_2[8]] = tensor[:, sorted_ranks_2[7]:sorted_ranks_2[8]] * num_samples[j] / total_samples_8
        tensor[:, sorted_ranks_2[8]:sorted_ranks_2[9]] = tensor[:, sorted_ranks_2[8]:sorted_ranks_2[9]] * num_samples[j] / total_samples_9
    elif i == 4:
        # 处理第二层第二个，rank作用在行
        tensor[0:sorted_ranks_2[0], :] = tensor[0:sorted_ranks_2[0], :] * num_samples[j] / total_samples_0
        tensor[sorted_ranks_2[0]:sorted_ranks_2[1], :] = tensor[sorted_ranks_2[0]:sorted_ranks_2[1], :] * num_samples[j] / total_samples_1
        tensor[sorted_ranks_2[1]:sorted_ranks_2[2], :] = tensor[sorted_ranks_2[1]:sorted_ranks_2[2], :] * num_samples[j] / total_samples_2
        tensor[sorted_ranks_2[2]:sorted_ranks_2[3], :] = tensor[sorted_ranks_2[2]:sorted_ranks_2[3], :] * num_samples[j] / total_samples_3
        tensor[sorted_ranks_2[3]:sorted_ranks_2[4], :] = tensor[sorted_ranks_2[3]:sorted_ranks_2[4], :] * num_samples[j] / total_samples_4
        tensor[sorted_ranks_2[4]:sorted_ranks_2[5], :] = tensor[sorted_ranks_2[4]:sorted_ranks_2[5], :] * num_samples[j] / total_samples_5
        tensor[sorted_ranks_2[5]:sorted_ranks_2[6], :] = tensor[sorted_ranks_2[5]:sorted_ranks_2[6], :] * num_samples[j] / total_samples_6
        tensor[sorted_ranks_2[6]:sorted_ranks_2[7], :] = tensor[sorted_ranks_2[6]:sorted_ranks_2[7], :] * num_samples[j] / total_samples_7
        tensor[sorted_ranks_2[7]:sorted_ranks_2[8], :] = tensor[sorted_ranks_2[7]:sorted_ranks_2[8], :] * num_samples[j] / total_samples_8
        tensor[sorted_ranks_2[8]:sorted_ranks_2[9], :] = tensor[sorted_ranks_2[8]:sorted_ranks_2[9], :] * num_samples[j] / total_samples_9
    else:
        tensor = tensor * num_samples[j] / total_samples_0
    return tensor


# 联邦加权平均函数，考虑各客户端的数据量
def federated_weighted_avg(weights, num_samples):
    avg_weights = []
    max_rank_1 = 0
    max_rank_2 = 0
    for l in weights:
        t0 = l[0]
        t1 = l[1]
        t3 = l[3]
        t4 = l[4]
        dim_1, dim_rank_1 = t0.shape
        dim_rank_1, dim_2 = t1.shape
        dim_3, dim_rank_2 = t3.shape
        dim_rank_2, dim_4 = t4.shape
        if dim_rank_1 > max_rank_1:
            max_rank_1 = dim_rank_1
        if dim_rank_2 > max_rank_2:
            max_rank_2 = dim_rank_2
    i = 0
    new_shape_0 = (dim_1, max_rank_1)
    new_shape_1 = (max_rank_1, dim_2)
    new_shape_3 = (dim_3, max_rank_2)
    new_shape_4 = (max_rank_2, dim_4)
    while i < len(weights):
        weights[i][0] = resize_tensor(weights[i][0], new_shape_0)
        weights[i][1] = resize_tensor(weights[i][1], new_shape_1)
        weights[i][3] = resize_tensor(weights[i][3], new_shape_3)
        weights[i][4] = resize_tensor(weights[i][4], new_shape_4)
        i += 1
    for i in range(len(weights[0])):   # 先遍历同位置矩阵，再遍历client
        weighted_sum = sum(share_tensor(weights[j][i], i, j) for j in range(len(weights)))
        avg_weights.append(weighted_sum)
    return avg_weights


# 打印权重差异
def print_weight_differences(client_weights, global_weights):
    for i, (client_weight, global_weight) in enumerate(zip(client_weights, global_weights)):
        print(f"Layer {i}:")
        print(f"  Client Weight: {client_weight.shape}")
        print(f"  Global Weight: {global_weight.shape}")
        if client_weight.shape != global_weight.shape:
            print("  Size mismatch!")
        else:
            print(f"  Max Difference: {torch.max(torch.abs(client_weight - global_weight)).item()}")


# 设置超参数
# 基准秩160（第一层），100，每个label对应0.1的系数
ranks_1_base = 160
ranks_2_base = 100
num_rounds = 15
learning_rate = 0.05   # 各种情况区别不明显的话试试0.01
batch_size = 64
epochs_per_client = 5
global_models = []
global_model = create_model(rank_1 = 160, rank_2 = 100)
criterion = nn.CrossEntropyLoss()

i = 0
while i < 10:      # 后面把这个数改成client的个数
    global_models.append(create_model(rank_1 = 160, rank_2 = 100))
    i += 1


# 记录每轮的准确率
accuracy_history = []
# 第一轮
client_weights = []
num_samples = []
client_loss_sums = []  # 存储每个客户端的loss总和

client_lossesE_list = []
client_p_list = []
y = 0
while y < len(client_subsets):
    client_data = client_subsets[y]
    client_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)
    model = create_model(rank_1=160, rank_2=100)
    # 加载全局模型的权重
    model.load_state_dict(global_models[y].state_dict())
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # 在客户端训练模型并记录loss总和
    client_lossesE, client_p = train(model, client_loader, criterion, optimizer, epochs=epochs_per_client)

    # 打印当前客户端的损失
    #print(f"  client_losses: {client_lossesE}")
    #print(f"  client_p: {client_p}")
    client_lossesE_list.append(client_lossesE)
    client_p_list.append(client_p)
    # 打印客户端损失列表



    #client_loss_sums.append(client_loss)  # 将每个客户端的loss总和添加到列表中
    model_weights = [param.data.clone() for param in model.parameters()]  # 保存客户端模型的权重副本
    client_weights.append(model_weights)
    num_samples.append(len(client_data))  # 保存客户端样本数量
    y += 1
print("Client LossesE List:")
for i, losses in enumerate(client_lossesE_list):
    print(f"  Client {i + 1} Losses: {losses}")

# 打印 p 值列表
print("\nClient p Values List:")
for i, p_values in enumerate(client_p_list):
    print(f"  Client {i + 1} p Values: {p_values}")




rates = []
for loss in client_lossesE_list:
    rates.append(loss/max(client_lossesE_list))
# rates = [0.00822395865024862, 0.023054642949453835, 0.07011542170741193, 0.13822345908114744, 0.18815894255187074, 0.3005734714787031, 0.3871121103338647, 0.4991436035113997, 0.740006499512193, 1.0]
# num_samples = [593, 1343, 2088, 2964, 3938, 5023, 6503, 8592, 11518, 17438]
print(f'rates:{rates}')
ranks_1 = []
ranks_2 = []
for rate in rates:
    a = int(rate * ranks_1_base)+10
    if a > 160:
        a = 160
    b = int(rate * ranks_2_base)+10
    if b > 100:
        b = 100
    ranks_1.append(a)
    ranks_2.append(b)
# 由于暂不清楚出现相同的秩对share_tensor()函数的运行是否有影响，这里就先都加10
print(num_samples)
print(f'ranks_1:{ranks_1}')
print(f'ranks_2:{ranks_2}')
sorted_ranks_1 = sorted(ranks_1)
sorted_ranks_2 = sorted(ranks_2)
combine = []
j = 0
while j < 10:
    combine.append((rates[j],num_samples[j]))
    j += 1
sorted_combine = sorted(combine, key=lambda x: x[0])
sorted_num_samples = []
for x in sorted_combine:
    sorted_num_samples.append(x[1])

# # 聚合权重，使用加权平均
# new_weights = federated_weighted_avg(client_weights, num_samples)
# # 在测试集上评估全局模型并记录准确率
# accuracy = test(global_model, test_loader)
# accuracy_history.append(accuracy)
# print('Round',  1, 'Test Accuracy:', accuracy * 100, '%')


# 重新初始化，这一步后面有没有必要再写一个第一轮目前存疑。
i = 0
while i < 10:      # 后面把这个数改成client的个数
    global_models[i] = create_model(rank_1 = ranks_1[i], rank_2 = ranks_2[i])
    i += 1
# 记录每轮的准确率
accuracy_history = []
# 第一轮
client_weights = []
num_samples = []
y = 0
while y < len(client_subsets):
    client_data = client_subsets[y]
    client_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)
    model = create_model(rank_1 = ranks_1[y], rank_2 = ranks_2[y])
    # print(global_model.state_dict())
    model.load_state_dict(global_models[y].state_dict())  # 加载全局模型的权重
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train(model, client_loader, criterion, optimizer, epochs=epochs_per_client)  # 在客户端训练模型

    model_weights = [param.data.clone() for param in model.parameters()]  # 保存客户端模型的权重副本
    client_weights.append(model_weights)
    num_samples.append(len(client_data))  # 记录客户端的数据样本数量
    y += 1
# 聚合权重，使用加权平均
new_weights = federated_weighted_avg(client_weights, num_samples)
# 在测试集上评估全局模型并记录准确率
accuracy = test(global_model, test_loader)
accuracy_history.append(accuracy)
print('Round',  1, 'Test Accuracy:', accuracy * 100, '%')


# 联邦学习主循环
for round_num in range(1,num_rounds):
    client_weights = []
    num_samples = []
    y = 0
    while y < len(client_subsets):
        new_weights_client = copy.deepcopy(new_weights)

        # for t in new_weights:
        #     print(t.shape)

        dim_r1, dim_c1 = new_weights[0].shape
        dim_r2, dim_c2 = new_weights[1].shape
        dim_r3, dim_c3 = new_weights[3].shape
        dim_r4, dim_c4 = new_weights[4].shape
        # print(dim_r1, dim_c1, dim_r2, dim_c2, ranks[y])
        new_weights_client[0] = clip_tensor(new_weights[0],(dim_r1, ranks_1[y]))
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
        model = create_model(rank_1 = ranks_1[y], rank_2 = ranks_2[y])
        model.load_state_dict(global_models[y].state_dict())  # 加载全局模型的权重
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        train(model, client_loader, criterion, optimizer, epochs=epochs_per_client)  # 在客户端训练模型

        model_weights = [param.data.clone() for param in model.parameters()]  # 保存客户端模型的权重副本
        client_weights.append(model_weights)
        num_samples.append(len(client_data))  # 记录客户端的数据样本数量
        y += 1

    # 聚合权重，使用加权平均
    new_weights = federated_weighted_avg(client_weights, num_samples)
    # 将聚合后的新权重应用到 global_model
    state_dict = global_model.state_dict()
    new_state_dict = {key: value for key, value in zip(state_dict.keys(), new_weights)}
    global_model.load_state_dict(new_state_dict)
    # 在测试集上评估全局模型并记录准确率
    accuracy = test(global_model, test_loader)
    accuracy_history.append(accuracy)
    print('Round', round_num + 1, 'Test Accuracy:', accuracy * 100, '%')

# 绘制准确率关于训练轮次的折线图
x_values = list(range(1, len(accuracy_history) + 1))
y_values = [acc * 100 for acc in accuracy_history]
print(f'x_values:{x_values}')
print(f'y_values:{y_values}')
plt.plot(x_values, y_values)
plt.xlabel('Federated Learning Rounds')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs Federated Learning Rounds')
plt.grid(True)
plt.show()

# 最终全局模型的准确率
final_accuracy = test(global_model, test_loader)


# 此代码为异化秩后的
# 下周要做的事情：1.将神经网络的前两层都用Lora，初始秩为160和100，   2.可以设置一个全局随机数种子
# 3.根据tag设置每个client的rank,r=初始秩*(tag量/10)     4.share:新的聚合方法(仅把有的行平均）     5.加上ewc?
# 最后对比图的时候，可以放一个纯联邦学习，一个lora-0填充，一个lora-share,最后都加个ewc，展示的时候看看效果酌情挑选。
# fashion MNIST数据集，可以试试，更复杂。
# 信息熵，互信息motual info 信息增益（了解一下）   每个epoch的loss越高，训练数据的信息量越高
# 新秩方法：第一轮结束后设置秩，后面不变。每个客户端的loss为5个epoch的和，用其中最大的client的loss和作为分母，分子就是本client的loss和（如果出现某个client的loss和为0，特殊处理以下）

# 每个人把自己的对比图发给老师，并附上自己的参数以及rank。
# 下周使用信息熵来算rank，准备pre
# 可以记上一个问题最后pre讲：算个出来rate是0怎么处理？
