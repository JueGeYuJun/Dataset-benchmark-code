import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL.Image import Image
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn.functional as F
import time
import logging

torch.manual_seed(1)  # 设置随机种子, 用于复现

# 配置 logging
log_filename = 'logs/VGG/VGGTest6.log'  # 日志文件的路径
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
data_dir = 'data'
image_size = (64, 64)
validation_split = 0.2  # 定义验证集所占比例
num_classes = 2922  # 定义数据集的标签数量
# 超参数
EPOCH = 15  # 前向后向传播迭代次数
LR = 0.0005  # 学习率 learning rate
BATCH_SIZE = 64  # 批量训练时候一次送入数据的size

class AddSaltPepperNoise(object):
    def __init__(self, p=0.5, amount=0.02):
        self.p = p
        self.amount = amount

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            noisy_img = img.clone()
            salt_vs_pepper = 0.5
            num_salt = int(self.amount * img.numel() * salt_vs_pepper)
            num_pepper = int(self.amount * img.numel() * (1.0 - salt_vs_pepper))

            # 随机选择椒盐坐标
            salt_coords = [torch.randint(0, d, (num_salt,)) for d in img.shape]
            pepper_coords = [torch.randint(0, d, (num_pepper,)) for d in img.shape]

            # 将椒盐坐标设置为1或0
            noisy_img[tuple(salt_coords)] = 1
            noisy_img[tuple(pepper_coords)] = 0

            return noisy_img
        else:
            raise TypeError("Input should be a tensor")


# 定义数据增强的transforms，包括噪点
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
])


# 定义数据增强的transform
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    augmentation_transform,
    AddSaltPepperNoise(p=0.5, amount=0.02),  # 添加椒盐噪声
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

class VGG16(torch.nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      # (1(32-1)- 32 + 3)/2 = 1
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, np.sqrt(2. / n))
                m.weight.detach().normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.detach().normal_(0, 0.05)
                m.bias.detach().detach().zero_()

    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        logits = self.classifier(x.view(x.size(0), -1))
        return logits


vgg16 = VGG16(num_classes)
vgg16.to(device)
# 打印模型结构
print(vgg16)
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LR)  # 定义优化器
loss_func = nn.CrossEntropyLoss()  # 定义损失函数


def get_accuracy(loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)  # 将输入数据移动到GPU上
            labels = labels.to(device)  # 将标签移动到GPU上
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
        batch_x = batch_x.to(device)  # 将输入数据移动到GPU上
        batch_y = batch_y.to(device)  # 将输入数据移动到GPU上
        pred_y = vgg16(batch_x)
        loss = loss_func(pred_y, batch_y)
        optimizer.zero_grad()  # 清空上一层梯度
        outputs = vgg16(batch_x)
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
    val_accuracy, val_correct, val_total = get_accuracy(val_loader, vgg16, device)
    print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")
    logging.info(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")

    # 获取验证集平均准确率
    average_val_accuracy = val_correct / val_total
    print(f"Epoch {epoch + 1}, Average Validation Accuracy: {average_val_accuracy}")
    logging.info(f"Epoch {epoch + 1}, Average Validation Accuracy: {average_val_accuracy}")

# 测试模型
test_accuracy, test_correct, test_total = get_accuracy(val_loader, vgg16, device)
print('Test accuracy:', test_accuracy)
logging.info('Test accuracy: %f', test_accuracy)

# 获取测试集平均准确率
average_test_accuracy = test_correct / test_total
print('Average Test accuracy:', average_test_accuracy)
logging.info('Average Test accuracy: %f', average_test_accuracy)


