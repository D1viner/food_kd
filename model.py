import torch
from torchvision import models
import torch.nn as nn
import senet
def load_model(name, class_num, is_pretrained):

    weights = None
    if name == 'mobilenet':
        if is_pretrained:
            weights = models.MobileNet_V2_Weights
        model = models.mobilenet_v2(weights = weights)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=class_num, bias=True)
    elif name == "mobilenetV3":
        MobileNet_V3_Large_Weights=models.MobileNet_V3_Large_Weights
        model = models.mobilenet_v3_large(weights=("pretrained", MobileNet_V3_Large_Weights.IMAGENET1K_V1))
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
        model.eval()
        return model
    elif name == "resnet152":
        if is_pretrained:
            weights = models.ResNet152_Weights
        model = models.resnet152(weights = weights)
        model.fc = nn.Linear(in_features=2048, out_features=class_num, bias=True)
    elif name == "resnet18":
        if is_pretrained:
            weights = models.ResNet18_Weights
        model = models.resnet18(weights = weights)
        model.fc = nn.Linear(in_features=2048, out_features=class_num, bias=True)
    elif name == "senet154_pretrained":
        model = senet.senet154(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(in_features=2048, out_features=class_num, bias=True)
    model.eval()

    return model

if __name__ == "__main__":
    #
    # mobilenet = load_model('mobilenet', 1000, True)
    # total_param = sum([param.nelement() for param in mobilenet.parameters()])
    # print("Number of parameter: %.2fM" % (total_param/1e6))

    # resnet152 = load_model('resnet152', 101, False)
    # total_param = sum([param.nelement() for param in resnet152.parameters()])
    # print("Number of parameter: %.2fM" % (total_param/1e6))

    # resnet18 = load_model('resnet18', 101, False)
    # total_param = sum([param.nelement() for param in resnet18.parameters()])
    # print("Number of parameter: %.2fM" % (total_param/1e6))

    mobilenetV3 = load_model('mobilenetV3', 101, False)
    total_param = sum([param.nelement() for param in mobilenetV3.parameters()])
    print("Number of parameter: %.2fM" % (total_param/1e6))