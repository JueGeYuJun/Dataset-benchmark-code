import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import time
import logging

torch.manual_seed(1)  # 设置随机种子, 用于复现

# 配置 logging
log_filename = 'logs/Inception/InceptionTest7.log'  # 日志文件的路径
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
image_size = (224, 224)
validation_split = 0.2  # 定义验证集所占比例
num_classes = 2922 # 定义数据集的标签数量
# 超参数
EPOCH = 15  # 前向后向传播迭代次数
LR = 0.001  # 学习率 learning rate
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

# 将数据增强的transforms添加到原始transforms.Compose中
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=3),
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

################################################
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            # 在官方的实现中，其实是3x3的kernel并不是5x5，这里我也懒得改了，具体可以参考下面的issue
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

################################################
googlenet = GoogLeNet(num_classes)
# 打印模型结构
googlenet.to(device)
print(googlenet)
optimizer = torch.optim.Adam(googlenet.parameters(), lr=LR)  # 定义优化器
loss_func = nn.CrossEntropyLoss()  # 定义损失函数


def get_accuracy(loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)  # 将输入数据移动到GPU上
            labels = labels.to(device)  # 将标签移动到GPU上
            outputs, _, _ = model(inputs)  # 获取主分类器的输出，并忽略辅助分类器的输出
            # outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # 使用主分类器的输出进行预测
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
        pred_y, aux2, aux1 = googlenet(batch_x)
        main_loss = loss_func(pred_y, batch_y)  # 计算主分类器的损失
        aux1_loss = loss_func(aux1, batch_y) if googlenet.aux_logits and googlenet.training else 0  # 计算辅助分类器1的损失
        aux2_loss = loss_func(aux2, batch_y) if googlenet.aux_logits and googlenet.training else 0  # 计算辅助分类器2的损失
        loss = main_loss + 0.3 * (aux1_loss + aux2_loss)  # 组合主分类器和辅助分类器的损失
        optimizer.zero_grad()  # 清空上一层梯度
        # outputs = googlenet(batch_x)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新优化器的学习率，一般按照epoch为单位进行更新
        running_loss += loss.item()
        _, preds = torch.max(pred_y, 1)  # 获取模型预测的类别
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
    val_accuracy, val_correct, val_total = get_accuracy(val_loader, googlenet, device)
    print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")
    logging.info(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")

    # 获取验证集平均准确率
    average_val_accuracy = val_correct / val_total
    print(f"Epoch {epoch + 1}, Average Validation Accuracy: {average_val_accuracy}")
    logging.info(f"Epoch {epoch + 1}, Average Validation Accuracy: {average_val_accuracy}")

# 测试模型
test_accuracy, test_correct, test_total = get_accuracy(val_loader, googlenet, device)
print('Test accuracy:', test_accuracy)
logging.info('Test accuracy: %f', test_accuracy)

# 获取测试集平均准确率
average_test_accuracy = test_correct / test_total
print('Average Test accuracy:', average_test_accuracy)
logging.info('Average Test accuracy: %f', average_test_accuracy)



