import torch
import torch.nn as nn
import torch.optim as optim

# 虚构的数据
batch_size = 32
num_classes = 10
input_size = 100
temperature = 1.0

# 创建模型
teacher_model = nn.Linear(input_size, num_classes)
student_model = nn.Linear(input_size, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(student_model.parameters(), lr=0.01)

# 创建虚构的数据和标签
anchors = torch.randn(batch_size, input_size)
labels = torch.randint(0, num_classes, (batch_size,))

# Forward pass：获取模型预测
teacher_logits_anchor = teacher_model(anchors)
student_logits_anchor = student_model(anchors)

# 计算 KD Loss
correct_predictions = torch.argmax(teacher_logits_anchor, dim=1).eq(labels)
incorrect_predictions = ~correct_predictions
pre_kd_loss = torch.sum(criterion(student_logits_anchor, labels))
# 计算 KD Loss，正确预测的样本使用交叉熵损失，错误预测的样本 KD Loss 设置为零
kd_loss = torch.sum(correct_predictions * criterion(student_logits_anchor, torch.argmax(teacher_logits_anchor, dim=1)) +
                   incorrect_predictions * 0.0)

# 反向传播和优化
optimizer.zero_grad()
kd_loss.backward()
optimizer.step()

# 打印 KD Loss
print(f"KD Loss: {kd_loss.item()}")
print(f"pre_kd_loss: {pre_kd_loss.item()}")
