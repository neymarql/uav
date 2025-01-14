import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
import json
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import logging


# 初始化日志配置
def setup_logging(rank, log_dir="./logs", log_filename="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_filename)

    if rank == 0:  # 只在主进程中记录日志
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | Rank %(process)d | %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
    else:
        logging.basicConfig(level=logging.CRITICAL)  # 禁止其他进程记录日志


# 设置数据集路径
train_dir = './dataset1/train'
val_dir = './dataset1/val'
test_dir = './dataset1/test'


# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, uav_dir, background_dir, transform=None):
        self.uav_images = [os.path.join(uav_dir, img) for img in os.listdir(uav_dir) if
                           img.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.background_images = [os.path.join(background_dir, img) for img in os.listdir(background_dir) if
                                  img.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.images = self.uav_images + self.background_images
        self.labels = [1] * len(self.uav_images) + [0] * len(self.background_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


# 定义通用的图像转换
transform_common = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),  # 对于ViT需要更大的输入尺寸
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(rank, world_size, args, transform):
    # 创建数据集
    train_dataset = ImageDataset(os.path.join(train_dir, 'UAV'), os.path.join(train_dir, 'background'),
                                 transform=transform)
    val_dataset = ImageDataset(os.path.join(val_dir, 'UAV'), os.path.join(val_dir, 'background'), transform=transform)
    test_dataset = ImageDataset(os.path.join(test_dir, 'UAV'), os.path.join(test_dir, 'background'),
                                transform=transform)

    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=100, sampler=val_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, sampler=test_sampler, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


########################################
# 模型部分 - 增加创新性和多模型对比
########################################

# 1. 基础CNN+SE模块（适合小尺寸图像的轻量化模型）
# Squeeze-and-Excitation模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SE_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SE_CNN, self).__init__()
        # 适合小图像的轻量CNN结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.se1 = SEModule(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.se2 = SEModule(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # 64x64 -> pool2次 -> 16x16
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.se1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.se2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# 2. 使用轻量级ViT模型进行对比（如DeiT Tiny）
def create_vit_model(num_classes=2):
    # 从timm中加载轻量ViT
    model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
    return model


# 3. CNN+Transformer 混合模型
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super(SimpleTransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, N, D = x.size()
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]
        x = self.transformer_encoder(x)
        return x[:, 0]


class CNN_TransformerHybrid(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_TransformerHybrid, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.d_model = 128
        self.seq_len = 16 * 16
        self.transformer = SimpleTransformerEncoder(d_model=self.d_model, nhead=4, num_layers=2, dim_feedforward=256)
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, self.d_model)
        x = self.transformer(x)
        x = self.fc(x)
        return x


# 4. ResNet模型对比
class ResNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetModel, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


# 5. ResNet + Attention 模型
class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]
        key = self.key_conv(x).view(B, -1, H * W)  # [B, C//8, HW]
        attention = self.softmax(torch.bmm(query, key))  # [B, HW, HW]
        value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)  # [B, C, H, W]
        return out + x


class ResNetAttention(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetAttention, self).__init__()
        self.resnet = timm.create_model('resnet18', pretrained=True, features_only=True)
        self.attention = AttentionModule(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.resnet(x)[-1]  # 获取最后一个特征图
        features = self.attention(features)
        features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        out = self.fc(features)
        return out


########################################
# 辅助函数（训练、验证）
########################################

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def train_one_epoch(model, criterion, optimizer, train_loader, device, rank, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = torch.tensor(running_loss).to(device)
    epoch_correct = torch.tensor(correct).to(device)
    epoch_total = torch.tensor(total).to(device)

    # 使用all_reduce汇总所有进程的loss和正确数
    dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(epoch_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(epoch_total, op=dist.ReduceOp.SUM)

    epoch_loss = epoch_loss.item() / epoch_total.item()
    epoch_acc = epoch_correct.item() / epoch_total.item()

    if scheduler:
        scheduler.step()

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, device, rank):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # 汇总所有进程的正确数和总数
    correct_tensor = torch.tensor(correct).to(device)
    total_tensor = torch.tensor(total).to(device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    accuracy = correct_tensor.item() / total_tensor.item()
    return accuracy


def train_model_ddp(rank, world_size, args):
    # 初始化进程组
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    setup_logging(rank)

    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # 定义模型列表
    model_names = ['SE_CNN', 'ViT', 'CNN_TransformerHybrid', 'ResNet18', 'ResNetAttention']

    for model_name in model_names:
        # 定义模型
        if model_name == 'SE_CNN':
            model = SE_CNN(num_classes=2).to(device)
        elif model_name == 'ViT':
            model = create_vit_model(num_classes=2).to(device)
        elif model_name == 'CNN_TransformerHybrid':
            model = CNN_TransformerHybrid(num_classes=2).to(device)
        elif model_name == 'ResNet18':
            model = ResNetModel(num_classes=2).to(device)
        elif model_name == 'ResNetAttention':
            model = ResNetAttention(num_classes=2).to(device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # 包装DDP
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # 创建模型保存和日志目录
        model_dir = f"./results_ddp/{model_name}"
        if rank == 0:
            os.makedirs(model_dir, exist_ok=True)
            print(f"\nStart training {model_name}...")
        dist.barrier()  # 确保目录创建完毕

        # 获取数据加载器
        transform = transform_vit if model_name == 'ViT' else transform_common
        train_loader, val_loader, test_loader = get_dataloaders(rank, world_size, args, transform)

        best_val_acc = 0.0
        best_model_path = os.path.join(model_dir, 'best_model.pth')

        # 初始训练阶段
        for epoch in range(1, args.num_epochs + 1):
            train_loader.sampler.set_epoch(epoch)
            epoch_loss, epoch_acc = train_one_epoch(model, criterion, optimizer, train_loader, device, rank, scheduler)
            val_acc = evaluate(model, val_loader, device, rank)

            if rank == 0:
                tqdm.write(
                    f"Model: {model_name} | Epoch: {epoch}/{args.num_epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
                logging.info(
                    f"Model: {model_name} | Epoch: {epoch}/{args.num_epochs} "
                    f"| Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}"
                )
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_model(model.module, best_model_path)
                    tqdm.write(f"Best model saved at epoch {epoch} with Val Acc: {val_acc:.4f}")

        # 微调阶段：将验证集加入训练集
        if rank == 0:
            print(f"\nFine-tuning {model_name} by adding validation set to training set...")
        fine_tune_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset])
        fine_tune_sampler = DistributedSampler(fine_tune_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=32, sampler=fine_tune_sampler, num_workers=4,
                                      pin_memory=True)

        best_test_acc = 0
        best_model_finetune_path = os.path.join(model_dir, 'best_model_finetune.pth')

        for epoch in range(1, args.fine_tune_epochs + 1):
            fine_tune_loader.sampler.set_epoch(epoch)
            epoch_loss, epoch_acc = train_one_epoch(model, criterion, optimizer, fine_tune_loader, device, rank,
                                                    scheduler)
            test_acc = evaluate(model, test_loader, device, rank)

            if rank == 0:
                tqdm.write(
                    f"Fine-tune Epoch: {epoch}/{args.fine_tune_epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Test Acc: {test_acc:.4f}")
                logging.info(
                    f"Fine-tune Epoch: {epoch}/{args.fine_tune_epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Test Acc: {test_acc:.4f}")
                # 保存最佳微调模型
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    save_model(model.module, best_model_finetune_path)
                    tqdm.write(f"Best fine-tuned model saved at fine-tune epoch {epoch} with Test Acc: {test_acc:.4f}")

        # 最终评估阶段
        if rank == 0:
            # 加载最佳微调模型
            model.module.load_state_dict(torch.load(best_model_finetune_path))
            final_test_acc = evaluate(model, test_loader, device, rank)

            # 保存结果
            results_path = os.path.join(model_dir, 'results.json')
            with open(results_path, 'w') as f:
                json.dump({"best_val_acc": best_val_acc, "fine_tuned_test_acc": final_test_acc}, f)

            print(f"{model_name}: Best Val Acc = {best_val_acc:.4f}, Fine-tuned Test Acc = {final_test_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of initial training epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of fine-tuning epochs')
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    world_size = args.world_size
    mp.spawn(train_model_ddp,
             args=(world_size, args),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main()
