# import senet
from torchvision import models
from torchstat import stat
from thop import profile
from thop import clever_format
import torch
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1024/ 1024))
    print('-' * 90)
import mbv2
# weights = models.MobileNet_V2_Weights
# model = models.mobilenet_v2(weights = weights)
# model_structure(model)
model = mbv2.mbv2_ca()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
# print(flops, params)
macs, params = clever_format([flops, params], "%.3f")
print(macs,params)
import senet
import torch
import mbv2
# 创建 SENet-154 模型实例
# model = senet.senet154(num_classes=1000, pretrained=None)
# weights = models.MobileNet_V2_Weights
# model = models.mobilenet_v2(weights = weights)

# # 统计模型的参数数量
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# # 计算参数占用的存储空间大小（MB）
# param_size_mb = total_params * 4 / (1024 * 1024)  # 假设每个参数占用 4 个字节（float32）

# print(f"SENet-154 模型的参数占用内存大小为: {param_size_mb:.2f} MB")
# model_structure(model)