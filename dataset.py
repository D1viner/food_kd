from torchvision import datasets,transforms


def load_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    N = 256
    train_transforms = transforms.Compose([
         transforms.RandomHorizontalFlip(p=0.5),  # default value is 0.5
         transforms.Resize((N, N)),
         transforms.RandomCrop((224, 224)),
         transforms.ToTensor(),
         normalize
     ])

    test_transforms = transforms.Compose([
         transforms.Resize((N, N)),
         transforms.CenterCrop((224, 224)),
         transforms.ToTensor(),
         normalize
     ])
    return train_transforms, test_transforms

def food101():
    train_transforms, test_transforms = load_transforms()
    train_dataset = datasets.Food101(root='./data', split='train', transform=train_transforms)
    test_dataset = datasets.Food101(root='./data', split='test', transform=test_transforms)
    return train_dataset, test_dataset
