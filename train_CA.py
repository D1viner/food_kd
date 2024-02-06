from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import dataset
import argparse
import torch
import torch.optim as optim
from model import *
import torchmetrics
import pathlib
import time
import mbv2
from collections import OrderedDict
import data
    

def parse_args():
    parser = argparse.ArgumentParser('parameters for training model')
    parser.add_argument('--model_name', type=str, default='mobilenet')
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--step_lr', type=int, default=3)
    parser.add_argument('--dataset_name', type=str, default='food101')
    args, unparsed = parser.parse_known_args()
    return args


def train_model():
    print(f'{epoch} Epoch, Training Network.....')
    model.train()

    # metric top1&top5
    top1acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=class_num, top_k=1).to(device)
    top5acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=class_num, top_k=5).to(device)

    # 训练一个Epoch
    traing_epoch_loss = 0
    for batch_index, (images, target) in enumerate(train_dataloader):
        images, target = images.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        # compute metric
        top1acc = top1acc_metric(logits, target)
        top5acc = top5acc_metric(logits, target)

        # log to tensorboard
        if batch_index % 100 == 0:
            trained_samples = batch_index * BATCH_SIZE + len(images)
            print(f"Training Epoch: {epoch}, [{trained_samples}/{total_sample}]\t "
                  f"Loss:{loss.item():.4f}, \t top1acc:{top1acc.item():.4f},\t top5acc:{top5acc.item():.4f}")
            writer.add_scalar("training batch loss", loss.item(), batch_index + epoch * len(train_dataloader))
            writer.add_scalar("training batch top1acc", top1acc, batch_index + epoch * len(train_dataloader))
            writer.add_scalar("training batch top5acc", top5acc, batch_index + epoch * len(train_dataloader))

        # log epoch loss
        traing_epoch_loss += loss.item()
    traing_epoch_loss = traing_epoch_loss/len(train_dataloader)

    top1acc_epoch = top1acc_metric.compute()
    top5acc_epoch = top5acc_metric.compute()

    top1acc_metric.reset()
    top5acc_metric.reset()

    return top1acc_epoch, top5acc_epoch, traing_epoch_loss


@torch.no_grad()
def test_model():
    print(f'{epoch} Epoch, Evaluating Network.....')
    model.eval()
    # metric top1&top5
    top1acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=class_num, top_k=1).to(device)
    top5acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=class_num, top_k=5).to(device)

    test_epoch_loss = 0
    for batch_index, (images, target) in enumerate(test_dataloader):
        print(f"Training Epoch: {epoch},[{batch_index}/{len(test_dataloader)}] has evaluated.")
        images, target = images.to(device), target.to(device)

        logits = model(images)
        loss = criterion(logits, target)
        test_epoch_loss += loss.item()

        # compute metric
        top1acc = top1acc_metric(logits, target)
        top5acc = top5acc_metric(logits, target)
    
    top1acc_epoch = top1acc_metric.compute()
    top5acc_epoch = top5acc_metric.compute()
    print(f"Evaluating Test dataset: Epoch: {epoch}, top1: {top1acc_epoch}, top5: {top5acc_epoch}")
    
    test_epoch_loss = test_epoch_loss/len(test_dataloader)

    top1acc_metric.reset()
    top5acc_metric.reset()

    return top1acc_epoch, top5acc_epoch, test_epoch_loss
    

if __name__ == "__main__":

    # 解析参数
    args = parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    lr = args.learning_rate
    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    STEP_LR = args.step_lr
    if dataset_name == "food101":
        class_num = 101
    elif dataset_name == "food500":
        class_num = 500
    elif dataset_name == "food2k":
        class_num = 2000

    # 存放日志和模型的目录    
    pathlib.Path(f"./{model_name}_{dataset_name}_pretrained_mobilenetCA").mkdir(parents=True, exist_ok=True)

    # 读取数据
    # train_food101, test_food101 = dataset.food101()
    # total_sample = len(train_food101)
    # train_dataloader = DataLoader(dataset=train_food101, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    # test_dataloader = DataLoader(dataset=test_food101, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    train_food500, test_food500 = data.load_dataset(train_dataset_path='/home/root-user/projects/food_kd/kd_food/data/isiafood500/metadata_ISIAFood_500/train_full.txt',test_dataset_path='/home/root-user/projects/food_kd/kd_food/data/isiafood500/metadata_ISIAFood_500/test_private.txt',batch_size=32)
    total_sample = len(train_food500)
    train_dataloader = DataLoader(dataset=train_food500, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(dataset=test_food500, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    # 显卡训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型
    # model = load_model(model_name, class_num=class_num, is_pretrained=False)
    # model = senet.senet154(num_classes=class_num, pretrained='imagenet')
    # model = load_model(model_name, class_num=class_num, is_pretrained=True)
    
    
    # 加载预训练的 state_dict
    state_dict = torch.load('/home/root-user/projects/food_kd/kd_food/models/mbv2_ca.pth', map_location=device)

    # 如果需要调整键的映射，可以手动创建一个新的 state_dict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k == 'classifier.weight':
            # 将 'classifier.weight' 映射到 'classifier.1.weight'
            new_state_dict['classifier.1.weight'] = v
        elif k == 'classifier.bias':
            # 将 'classifier.bias' 映射到 'classifier.1.bias'
            new_state_dict['classifier.1.bias'] = v
        else:
            # 其他层保持原样
            new_state_dict[k] = v

    model = mbv2.mbv2_ca()
    # 加载模型的 state_dict，并确保设备匹配
    model.load_state_dict(new_state_dict)

    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
    model.to(device)


    # 加载模型的 state_dict，并且确保设备匹配
    # model.load_state_dict(state_dict)
    model.to(device)

    # 将模型放到显卡上跑
    # model.to(device)

    # 配置学习率
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_LR, gamma=0.9)

    # 训练与测试
    best_top1_acc = 0
    best_top5_acc = 0

    # init tensorboard
    time_str = str(time.time())
    log_dir = f"./{model_name}_{dataset_name}_pretrained_mobilenetCA/{model_name}_{dataset_name}_{time_str}_log"
    writer = SummaryWriter(log_dir)

    best_epoch = 0
    for epoch in range(EPOCH):
        # 训练模型
        train_top1_acc, train_top5_acc, training_epoch_loss = train_model()

        # 评估并记录最优top1模型
        test_top1_acc, test_top5_acc, test_epoch_loss = test_model()

        # 画出训练集和测试集中一个epoch的loss曲线
        writer.add_scalars("epoch loss for training and test", 
        {"training epoch loss":training_epoch_loss,"test epoch loss":test_epoch_loss},
        epoch)

        # 画出训练集和测试集中一个epoch的top1 acc曲线
        writer.add_scalars("epoch top1 acc for training and test", 
        {"training epoch top1":train_top1_acc,"test epoch top1":test_top1_acc},
        epoch)

        # 画出训练集和测试集中一个epoch的top1 acc曲线
        writer.add_scalars("epoch top5 acc for training and test", 
        {"training epoch top5":train_top5_acc,"test epoch top1":test_top5_acc},
        epoch)

        if test_top1_acc > best_top1_acc:
            best_top1_acc = test_top1_acc
            best_top5_acc = test_top5_acc
            best_epoch = epoch
            
            torch.save(model.state_dict(), f'./{model_name}_{dataset_name}_pretrained_mobilenetCA/{model_name}_{epoch}_pretrained.pth')

        # 学习率优化
        scheduler.step()

    print(f"best top1 acc is {best_top1_acc} and top5 acc is {best_top5_acc} at {best_epoch} epoch.")
