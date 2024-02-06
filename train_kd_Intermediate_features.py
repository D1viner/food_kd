import torch
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch.optim as optim
from model import *
import torchmetrics
import pathlib
import time

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

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.senet154 = load_model("senet154_pretrained", class_num=101, is_pretrained=False).to(torch.float32)
        self.senet154.load_state_dict(torch.load('/home/root-user/projects/food_kd/kd_food/models/senet154_pretrained.pth'))
        self.features = nn.Sequential(*list(self.senet154.children())[:-1])  # 移除最后一个全连接层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 添加全局平均池化层
        self.regressor = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 101),
        )

    def forward(self, x):
        feats = self.features(x)
        # print(feats.shape)#torch.Size([32, 2048, 1, 1])

        feats = self.regressor(feats)
        # print(feats.shape)#torch.Size([32, 256, 1, 1])

        feats = self.global_avg_pool(feats)  # 应用全局平均池化层
        feats = torch.flatten(feats, 1)
        # print(feats.shape)#torch.Size([32, 256])
        logits = self.classifier(feats)
        return logits, feats



class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.mobilenet = load_model('mobilenet',class_num=101,is_pretrained=True)
        
        self.feature = self.mobilenet.features
        self.regressor = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 101),
        )

    def forward(self, x):
        feats = self.feature(x)
        # print(feats.shape)  # 在这里添加打印语句来检查张量的形状torch.Size([32, 1280, 7, 7])

        feats = self.regressor(feats)
        # print(feats.shape)  # 在这里添加打印语句来检查张量的形状torch.Size([32, 256, 7, 7])

        feats = self.global_avg_pool(feats)
        # print(feats.shape)  # 在这里添加打印语句来检查张量的形状torch.Size([32, 256, 1, 1])

        feats = torch.flatten(feats, 1)
        # print(feats.shape)  # 在这里添加打印语句来检查张量的形状torch.Size([32, 256])

        logits = self.classifier(feats)
        
        return logits, feats



def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def compute_kld_loss(student_feats, teacher_feats):
    student_probs = torch.nn.functional.softmax(student_feats, dim=1)
    teacher_probs = torch.nn.functional.softmax(teacher_feats, dim=1)
    kld_loss = torch.nn.functional.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
    return kld_loss

def train_student_model(teacher_model, student_model, train_dataloader, test_dataloader, num_epochs=10, lr=0.001, step_lr=3, temperature=3.0):
    criterion = nn.CrossEntropyLoss()
    criterion_KLDiv = nn.KLDivLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_lr, gamma=0.9)
    
    best_accuracy = 0.0
    best_epoch = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    teacher_model.eval()  
    student_model.train()  

    freeze_model(teacher_model)

    time_str = str(time.time())
    log_dir = f"./{model_name}_{dataset_name}_KD_intermediate_features/{model_name}_{dataset_name}_{time_str}_KD_log"
    writer = SummaryWriter(log_dir)

    class_num = len(train_dataloader.dataset.classes)

    top1acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=class_num, top_k=1).to(device)
    top5acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=class_num, top_k=5).to(device)

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        total_samples = 0
        correct = 0

        for batch_index, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_features = teacher_model(inputs)
            
            student_logits, student_features = student_model(inputs)

            # Calculate classification loss
            loss_classification = criterion(student_logits, labels)

            # Calculate distillation loss
            soft_labels = torch.nn.functional.softmax(teacher_features[0] / temperature, dim=1)
            distillation_loss = criterion_KLDiv(torch.nn.functional.log_softmax(student_logits / temperature, dim=1), soft_labels)

            # Calculate intermediate feature loss
            loss_kld = mse_loss(student_features, teacher_features[1])

            # Combine losses
            loss = distillation_loss + 0.1 * loss_kld + 0.1 * loss_classification

            loss.backward()
            optimizer.step()


            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(student_logits, 1)
            correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_dataloader.dataset)
            epoch_accuracy = correct / len(train_dataloader.dataset)

            top1acc_metric(student_logits, labels)
            top5acc_metric(student_logits, labels)
        
            if batch_index % 100 == 0:
                trained_samples = batch_index * BATCH_SIZE + len(inputs)
                total_sample = len(train_food101)

                train_top1_acc = top1acc_metric.compute()
                train_top5_acc = top5acc_metric.compute()
                print(f"Training Epoch: {epoch}, [{trained_samples}/{total_sample}]\t "
                      f"Loss:{loss:.4f}, \t top1acc:{train_top1_acc:.4f},\t top5acc:{train_top5_acc:.4f}")

                writer.add_scalar("training batch loss", loss.item(), batch_index + epoch * len(train_dataloader))
                writer.add_scalar("training batch top1acc", train_top1_acc, batch_index + epoch * len(train_dataloader))
                writer.add_scalar("training batch top5acc", train_top5_acc, batch_index + epoch * len(train_dataloader))

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {train_top1_acc:.4f}, Top5 Accuracy: {train_top5_acc:.4f}")

        scheduler.step() 

        test_top1_acc, test_top5_acc, test_epoch_loss = test_model(student_model, test_dataloader, criterion, device, class_num)
        writer.add_scalars("Epoch Loss for Training and Test", 
                            {"Training Epoch Loss": epoch_loss, "Test Epoch Loss": test_epoch_loss},
                            epoch)
        writer.add_scalars("Epoch Top1 Acc for Training and Test", 
                            {"Training Epoch Top1": train_top1_acc, "Test Epoch Top1": test_top1_acc},
                            epoch)
        writer.add_scalars("Epoch Top5 Acc for Training and Test", 
                            {"Training Epoch Top5": train_top5_acc, "Test Epoch Top5": test_top5_acc},
                            epoch)
        
        if test_top1_acc > best_accuracy:
            best_accuracy = test_top1_acc
            best_epoch = epoch
            torch.save(student_model.state_dict(), f'./{model_name}_{dataset_name}_KD_intermediate_features/{model_name}_{epoch}_pretrained_KD.pth')
        scheduler.step()
    
    print(f"Best accuracy: {best_accuracy} at epoch {best_epoch}")

    writer.close()
    return student_model



@torch.no_grad()
def test_model(model, test_dataloader, criterion, device, class_num):
    model.eval()
    top1acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=class_num, top_k=1).to(device)
    top5acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=class_num, top_k=5).to(device)

    test_epoch_loss = 0
    for batch_index, (images, target) in enumerate(test_dataloader):
        print(f"Evaluating: [{batch_index}/{len(test_dataloader)}] batches.")
        images, target = images.to(device), target.to(device)

        logits, _ = model(images)  # 取出模型输出的 logits 部分
        loss = criterion(logits, target)
        test_epoch_loss += loss.item()

        top1acc_metric(logits, target)
        top5acc_metric(logits, target)
    
    top1acc_epoch = top1acc_metric.compute()
    top5acc_epoch = top5acc_metric.compute()
    print(f"Test dataset: top1: {top1acc_epoch}, top5: {top5acc_epoch}")
    
    test_epoch_loss = test_epoch_loss / len(test_dataloader)

    top1acc_metric.reset()
    top5acc_metric.reset()

    return top1acc_epoch, top5acc_epoch, test_epoch_loss






if __name__ == "__main__":
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

    pathlib.Path(f"./{model_name}_{dataset_name}_KD_intermediate_features").mkdir(parents=True, exist_ok=True)

    train_food101, test_food101 = dataset.food101()
    total_sample = len(train_food101)
    train_dataloader = DataLoader(dataset=train_food101, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(dataset=test_food101, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    teacher_model = TeacherModel()
    # student_model = load_model("mobilenet", class_num=class_num, is_pretrained=False)
    student_model = StudentModel()

    # teacher_model.load_state_dict(torch.load('/home/root-user/projects/food_kd/kd_food/models/senet154_pretrained.pth'),)

    trained_student_model = train_student_model(
        teacher_model,
        student_model,
        train_dataloader,
        test_dataloader,
        num_epochs=200,
        temperature=3.0,
        lr=lr,
        step_lr=STEP_LR)

