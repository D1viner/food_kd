import torch
import torch.nn as nn
import dataset_tri
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch.optim as optim
from model import *
import torchmetrics
import pathlib
import time
import random
from torchvision import transforms

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
    def forward(self, x):
        representations = self.senet154(x)
        return representations

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.mobilenet = load_model('mobilenet',class_num=101,is_pretrained=True)
    def forward(self, x):
        representations = self.mobilenet(x)
        return representations
    

def compute_infoNCE_loss(student_anchor, teacher_positive, teacher_negative, temperature=1.0):
    device = student_anchor.device  # 获取当前张量所在的设备

    batch_size = student_anchor.size(0)

    positive_similarity = torch.einsum('nc,nc->n', student_anchor, teacher_positive)
    # print(positive_similarity.shape)#torch.Size([32])
    negative_similarity = torch.einsum('nc,nc->n', student_anchor, teacher_negative)

    logits = torch.cat([positive_similarity.unsqueeze(-1), negative_similarity.unsqueeze(-1)], dim=1)
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    infoNCE_loss = nn.functional.cross_entropy(logits * temperature, targets)

    return infoNCE_loss





def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


@torch.no_grad()
def test_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f"Test Accuracy: {accuracy:.4f}, Avg Loss: {avg_loss:.4f}")
    return accuracy


def train_student_model(teacher_model, student_model, train_dataloader, test_dataloader, num_epochs=10, lr=0.001, step_lr=3, temperature=3.0):
    criterion = nn.CrossEntropyLoss()
    criterion_KLDiv = nn.KLDivLoss()
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_lr, gamma=0.9)
    
    best_accuracy = 0.0
    best_epoch = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    freeze_model(teacher_model)

    time_str = str(time.time())
    log_dir = f"./{model_name}_{dataset_name}_KD_after_contrastive/{model_name}_{dataset_name}_{time_str}_KD_log"
    writer = SummaryWriter(log_dir)

    class_num = len(train_dataloader.dataset.classes)

    top1acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=class_num, top_k=1).to(device)
    top5acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=class_num, top_k=5).to(device)

    for epoch in range(num_epochs):
        student_model.train()

        running_InfoNCE_loss = 0.0
        running_kd_loss = 0.0
        total_samples = 0
        correct = 0
        for batch_index, batch in enumerate(train_dataloader):   
            anchors, positives, negatives, anchor_labels, positive_labels, negative_labels = batch  

            anchor_samples = anchors.to(device)
            positive_samples = positives.to(device)
            negative_samples = negatives.to(device)   

            labels = torch.tensor(positive_labels, dtype=torch.long).to(device)

            optimizer.zero_grad()

            teacher_logits_anchor = teacher_model(anchor_samples)
            teacher_logits_positive = teacher_model(positive_samples)
            teacher_logits_negative = teacher_model(negative_samples)

            student_logits_anchor = student_model(anchor_samples)
            student_logits_positive = student_model(positive_samples)
            student_logits_negative = student_model(negative_samples)

            InfoNCE_loss = compute_infoNCE_loss(student_logits_anchor, teacher_logits_positive, teacher_logits_negative, temperature=1.0)
            running_InfoNCE_loss += InfoNCE_loss.item() * anchor_samples.size(0)

            InfoNCE_loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            teacher_logits_anchor_kd = teacher_model(anchor_samples)
            teacher_logits_positive_kd = teacher_model(positive_samples)
            teacher_logits_negative_kd = teacher_model(negative_samples)

            student_logits_anchor_kd = student_model(anchor_samples)
            student_logits_positive_kd = student_model(positive_samples)
            student_logits_negative_kd = student_model(negative_samples)

            cross_entropy_loss_1 = criterion(student_logits_anchor_kd, torch.argmax(nn.functional.softmax(teacher_logits_anchor_kd / temperature, dim=1), dim=1))
            cross_entropy_loss_2 = criterion(student_logits_positive_kd, labels)
            alpha = 0.5
            kd_loss = alpha * cross_entropy_loss_1 + (1 - alpha) * cross_entropy_loss_2
            running_kd_loss += kd_loss.item() * anchor_samples.size(0)

            kd_loss.backward()
            optimizer.step()

            
            total_samples += labels.size(0)
            _, predicted = torch.max(student_logits_anchor, 1)
            correct += (predicted == labels).sum().item()

            top1acc_metric(student_logits_anchor, labels)
            top5acc_metric(student_logits_anchor, labels)
        
            if batch_index % 100 == 0:
                trained_samples = batch_index * BATCH_SIZE + len(anchor_samples)
                total_sample= len(train_food101)
                contrastive_epoch_loss = running_InfoNCE_loss / total_samples
                kd_epoch_loss = running_kd_loss / total_samples

                train_top1_acc = top1acc_metric.compute()
                train_top5_acc = top5acc_metric.compute()
                print(f"Training Epoch: {epoch}, [{trained_samples}/{total_sample}]\t "
                f"contrastive loss:{contrastive_epoch_loss:.4f},kd loss:{kd_epoch_loss:.4f}, \t top1acc:{train_top1_acc:.4f},\t top5acc:{train_top5_acc:.4f}")
                writer.add_scalar("training epoch contrastive loss", contrastive_epoch_loss, epoch)
                writer.add_scalar("training epoch kd loss", kd_epoch_loss, epoch)
                writer.add_scalar("training batch top1acc", train_top1_acc, batch_index + epoch * len(train_dataloader))
                writer.add_scalar("training batch top5acc", train_top5_acc, batch_index + epoch * len(train_dataloader))

        print(f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Contrastive Loss: {contrastive_epoch_loss:.4f}, KD Loss: {kd_epoch_loss:.4f},")

        scheduler.step() 

        test_top1_acc, test_top5_acc, test_epoch_loss = test_model(student_model, test_dataloader, criterion, device, class_num)
        writer.add_scalars("Epoch Loss for Training and Test", 
                            {"Training Epoch contrastive loss": contrastive_epoch_loss,"Training Epoch kd losss": kd_epoch_loss, "Test Epoch Loss": test_epoch_loss},
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
            torch.save(student_model.state_dict(), f'./{model_name}_{dataset_name}_KD_after_contrastive/{model_name}_{epoch}_pretrained_KD.pth')    
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

        logits = model(images)
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

    pathlib.Path(f"./{model_name}_{dataset_name}_KD_after_contrastive").mkdir(parents=True, exist_ok=True)
    print('dataset preparing')
    train_food101, test_food101 = dataset_tri.food101()
    print('dataset prepared')
    total_sample = len(train_food101)
    train_dataloader = DataLoader(dataset=train_food101, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(dataset=test_food101, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    teacher_model = TeacherModel()
    student_model = StudentModel()
    
    trained_student_model = train_student_model(
        teacher_model,
        student_model,
        train_dataloader,
        test_dataloader,
        num_epochs=200,
        temperature=3.0,
        lr=lr,
        step_lr=STEP_LR)

