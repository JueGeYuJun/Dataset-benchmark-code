from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
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

#######################################################3
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def _mcfg(**kwargs):
    cfg = dict(se_ratio=0., bottle_ratio=1., stem_width=32)
    cfg.update(**kwargs)
    return cfg


model_cfgs = {
    "regnetx_200mf": _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13),
    "regnetx_400mf": _mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22),
    "regnetx_600mf": _mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16),
    "regnetx_800mf": _mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16),
    "regnetx_1.6gf": _mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18),
    "regnetx_3.2gf": _mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25),
    "regnetx_4.0gf": _mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23),
    "regnetx_6.4gf": _mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17),
    "regnetx_8.0gf": _mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23),
    "regnetx_12gf": _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19),
    "regnetx_16gf": _mcfg(w0=216, wa=55.59, wm=2.1, group_w=128, depth=22),
    "regnetx_32gf": _mcfg(w0=320, wa=69.86, wm=2.0, group_w=168, depth=23),
    "regnety_200mf": _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25),
    "regnety_400mf": _mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25),
    "regnety_600mf": _mcfg(w0=48, wa=32.54, wm=2.32, group_w=16, depth=15, se_ratio=0.25),
    "regnety_800mf": _mcfg(w0=56, wa=38.84, wm=2.4, group_w=16, depth=14, se_ratio=0.25),
    "regnety_1.6gf": _mcfg(w0=48, wa=20.71, wm=2.65, group_w=24, depth=27, se_ratio=0.25),
    "regnety_3.2gf": _mcfg(w0=80, wa=42.63, wm=2.66, group_w=24, depth=21, se_ratio=0.25),
    "regnety_4.0gf": _mcfg(w0=96, wa=31.41, wm=2.24, group_w=64, depth=22, se_ratio=0.25),
    "regnety_6.4gf": _mcfg(w0=112, wa=33.22, wm=2.27, group_w=72, depth=25, se_ratio=0.25),
    "regnety_8.0gf": _mcfg(w0=192, wa=76.82, wm=2.19, group_w=56, depth=17, se_ratio=0.25),
    "regnety_12gf": _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25),
    "regnety_16gf": _mcfg(w0=200, wa=106.23, wm=2.48, group_w=112, depth=18, se_ratio=0.25),
    "regnety_32gf": _mcfg(w0=232, wa=115.89, wm=2.53, group_w=232, depth=20, se_ratio=0.25)
}


def generate_width_depth(wa, w0, wm, depth, q=8):
    """Generates per block widths from RegNet parameters."""
    assert wa > 0 and w0 > 0 and wm > 1 and w0 % q == 0
    widths_cont = np.arange(depth) * wa + w0
    width_exps = np.round(np.log(widths_cont / w0) / np.log(wm))
    widths_j = w0 * np.power(wm, width_exps)
    widths_j = np.round(np.divide(widths_j, q)) * q
    num_stages, max_stage = len(np.unique(widths_j)), width_exps.max() + 1
    assert num_stages == int(max_stage)
    assert num_stages == 4
    widths = widths_j.astype(int).tolist()
    return widths, num_stages


def adjust_width_groups_comp(widths: list, groups: list):
    """Adjusts the compatibility of widths and groups."""
    groups = [min(g, w_bot) for g, w_bot in zip(groups, widths)]
    # Adjust w to an integral multiple of g
    widths = [int(round(w / g) * g) for w, g in zip(widths, groups)]
    return widths, groups


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 kernel_s: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 act: Optional[nn.Module] = nn.ReLU(inplace=True)):
        super(ConvBNAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=kernel_s,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_c)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RegHead(nn.Module):
    def __init__(self,
                 in_unit: int = 368,
                 out_unit: int = 1000,
                 output_size: tuple = (1, 1),
                 drop_ratio: float = 0.25):
        super(RegHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        self.fc = nn.Linear(in_features=in_unit, out_features=out_unit)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, se_ratio: float = 0.25):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class Bottleneck(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 stride: int = 1,
                 group_width: int = 1,
                 se_ratio: float = 0.,
                 drop_ratio: float = 0.):
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1)
        self.conv2 = ConvBNAct(in_c=out_c,
                               out_c=out_c,
                               kernel_s=3,
                               stride=stride,
                               padding=1,
                               groups=out_c // group_width)

        if se_ratio > 0:
            self.se = SqueezeExcitation(in_c, out_c, se_ratio)
        else:
            self.se = nn.Identity()

        self.conv3 = ConvBNAct(in_c=out_c, out_c=out_c, kernel_s=1, act=None)
        self.ac3 = nn.ReLU(inplace=True)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        if (in_c != out_c) or (stride != 1):
            self.downsample = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1, stride=stride, act=None)
        else:
            self.downsample = nn.Identity()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.se(x)
        x = self.conv3(x)

        x = self.dropout(x)

        shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.ac3(x)
        return x


class RegStage(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 depth: int,
                 group_width: int,
                 se_ratio: float):
        super(RegStage, self).__init__()
        for i in range(depth):
            block_stride = 2 if i == 0 else 1
            block_in_c = in_c if i == 0 else out_c

            name = "b{}".format(i + 1)
            self.add_module(name,
                            Bottleneck(in_c=block_in_c,
                                       out_c=out_c,
                                       stride=block_stride,
                                       group_width=group_width,
                                       se_ratio=se_ratio))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.children():
            x = block(x)
        return x


class RegNet(nn.Module):
    """RegNet model.

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    and refer to: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/regnet.py
    """

    def __init__(self,
                 cfg: dict,
                 in_c: int = 1,
                 num_classes: int = 1000,
                 zero_init_last_bn: bool = True):
        super(RegNet, self).__init__()

        # RegStem
        stem_c = cfg["stem_width"]
        self.stem = ConvBNAct(in_c, out_c=stem_c, kernel_s=3, stride=2, padding=1)

        # build stages
        input_channels = stem_c
        stage_info = self._build_stage_info(cfg)
        for i, stage_args in enumerate(stage_info):
            stage_name = "s{}".format(i + 1)
            self.add_module(stage_name, RegStage(in_c=input_channels, **stage_args))
            input_channels = stage_args["out_c"]

        # RegHead
        self.head = RegHead(in_unit=input_channels, out_unit=num_classes)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",  nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, "zero_init_last_bn"):
                    m.zero_init_last_bn()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def _build_stage_info(cfg: dict):
        wa, w0, wm, d = cfg["wa"], cfg["w0"], cfg["wm"], cfg["depth"]
        widths, num_stages = generate_width_depth(wa, w0, wm, d)

        stage_widths, stage_depths = np.unique(widths, return_counts=True)
        stage_groups = [cfg['group_w'] for _ in range(num_stages)]
        stage_widths, stage_groups = adjust_width_groups_comp(stage_widths, stage_groups)

        info = []
        for i in range(num_stages):
            info.append(dict(out_c=stage_widths[i],
                             depth=stage_depths[i],
                             group_width=stage_groups[i],
                             se_ratio=cfg["se_ratio"]))

        return info


def create_regnet(model_name="RegNetX_200MF", num_classes=1000):
    model_name = model_name.lower().replace("-", "_")
    if model_name not in model_cfgs.keys():
        print("support model name: \n{}".format("\n".join(model_cfgs.keys())))
        raise KeyError("not support model name: {}".format(model_name))

    model = RegNet(cfg=model_cfgs[model_name], num_classes=num_classes)
    return model

####################################################
# 实例化模型
num_classes = num_classes  # 替换为您的类别数
regnet = create_regnet(model_name="RegNetX_200MF", num_classes=num_classes)

# regnet.to(device)
print(regnet)  # 输出模型结构
optimizer = torch.optim.Adam(regnet.parameters(), lr=LR)  # 定义优化器
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
        pred_y = regnet(batch_x)
        loss = loss_func(pred_y, batch_y)
        optimizer.zero_grad()  # 清空上一层梯度
        outputs = regnet(batch_x)
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
    val_accuracy, val_correct, val_total = get_accuracy(val_loader, regnet, device)
    print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")
    logging.info(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy}")

    # 获取验证集平均准确率
    average_val_accuracy = val_correct / val_total
    print(f"Epoch {epoch + 1}, Average Validation Accuracy: {average_val_accuracy}")
    logging.info(f"Epoch {epoch + 1}, Average Validation Accuracy: {average_val_accuracy}")

# 测试模型
test_accuracy, test_correct, test_total = get_accuracy(val_loader, regnet, device)
print('Test accuracy:', test_accuracy)
logging.info('Test accuracy: %f', test_accuracy)

# 获取测试集平均准确率
average_test_accuracy = test_correct / test_total
print('Average Test accuracy:', average_test_accuracy)
logging.info('Average Test accuracy: %f', average_test_accuracy)


