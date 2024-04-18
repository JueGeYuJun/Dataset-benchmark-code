import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import time
import logging

torch.manual_seed(1)  # 设置随机种子, 用于复现

# 配置 logging
log_filename = 'logs/CNN/CNNtest_2_02.log'  # 日志文件的路径
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置控制台输出的级别
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据目录
data_dir = 'dataset'
image_size = (28, 28)
validation_split = 0.2  # 定义验证集所占比例
num_classes = 2922 # 定义数据集的标签数量
# 超参数
EPOCH = 40  # 前向后向传播迭代次数
LR = 0.001  # 学习率 learning rate
BATCH_SIZE = 64  # 批量训练时候一次送入数据的size

# 定义数据增强的transform
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 创建数据集
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

# 划分训练集和验证集
train_size = int((1 - validation_split) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 创建训练集和验证集数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 输入的图片 （1，28，28）
                in_channels=1,
                out_channels=16,  # 经过一个卷积层之后 （16,28,28）
                kernel_size=5,
                stride=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化层处理，维度为（16,14,14）
        )
# LeNet
        self.conv2 = nn.Sequential(
            nn.Conv2d(  # 输入（16,14,14）
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # 输出（32,14,14）
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 输出（32,7,7）
        )

        self.out = nn.Linear(32 * 7 * 7, num_classes) # 图片大小涉及改这里

    def forward(self, x):
        x = self.conv1(x)  # （batch_size,16,14,14）
        x = self.conv2(x)  # 输出（batch_size,32,7,7）
        x = x.view(x.size(0), -1)  # (batch_size,32*7*7)
        out = self.out(x)  # (batch_size,10)
        return out

# 实例化模型
cnn = CNN()
# cnn.to(device)
print(cnn)  # 输出模型结构
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 定义优化器
loss_func = nn.CrossEntropyLoss()  # 定义损失函数


def get_accuracy(loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            # inputs = inputs.to(device)  # 将输入数据移动到GPU上
            # labels = labels.to(device)  # 将标签移动到GPU上
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy, correct, total

# 训练模型
for epoch in range(EPOCH):
    running_loss = 0.0
    running_corrects = 0  # 用于累积每个 batch 的正确预测数量
    start_time = time.time()  # 记录当前时间
    correct = 0
    total = 0
    for i, (batch_x, batch_y) in enumerate(train_loader, 1):
        # batch_x = batch_x.to(device)  # 将输入数据移动到GPU上
        # batch_y = batch_y.to(device)  # 将输入数据移动到GPU上
        pred_y = cnn(batch_x)
        loss = loss_func(pred_y, batch_y)
        optimizer.zero_grad()  # 清空上一层梯度
        outputs = cnn(batch_x)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新优化器的学习率，一般按照epoch为单位进行更新
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)  # 获取模型预测的类别
        # 计算正确的预测数量
        running_corrects += torch.sum(preds == batch_y.data)

        if i % 100 == 0:
            avg_batch_loss = running_loss / 100
            elapsed_time = time.time() - start_time  # 计算已经经过的时间
            batches_remaining = len(train_loader) - i
            time_per_batch = elapsed_time / i
            time_remaining = time_per_batch * batches_remaining  # 计算剩余时间
            avg_batch_loss = running_loss / 100
            epoch_acc = running_corrects.double() / (100 * BATCH_SIZE)  # 计算目前的平均准确率
            print(f"Epoch {epoch + 1}, Batch {i}/{len(train_loader)}, "
                  f"Average Training Loss: {avg_batch_loss:.5f}, "
                  f"Current Accuracy: {epoch_acc:.5f},"
                  f"Time Remaining: {time_remaining:.1f} seconds")

            # 在需要记录信息的地方使用 logging
            # 训练过程中
            logging.info(f"Epoch {epoch + 1}, Batch {i}/{len(train_loader)}, "
                         f"Average Training Loss: {avg_batch_loss:.5f}, "
                         f"Current Accuracy: {epoch_acc:.5f},"
                         f"Time Remaining: {time_remaining:.1f} seconds")

            running_loss = 0.0
            running_corrects = 0  # 重置为0，用于下一个 batch

    average_epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Training Loss: {average_epoch_loss:.5f}")

    # 计算验证集准确率
    val_accuracy, val_correct, val_total = get_accuracy(val_loader, cnn, device)
    print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")
    logging.info(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")

    # 获取验证集平均准确率
    average_val_accuracy = val_correct / val_total
    print(f"Epoch {epoch + 1}, Average Validation Accuracy: {average_val_accuracy}")
    logging.info(f"Epoch {epoch + 1}, Average Validation Accuracy: {average_val_accuracy}")

# 测试模型
test_accuracy, test_correct, test_total = get_accuracy(val_loader, cnn, device)
print('Test accuracy:', test_accuracy)
logging.info('Test accuracy: %f', test_accuracy)

# 获取测试集平均准确率
average_test_accuracy = test_correct / test_total
print('Average Test accuracy:', average_test_accuracy)
logging.info('Average Test accuracy: %f', average_test_accuracy)



