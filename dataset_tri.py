from torchvision import datasets,transforms
from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader
import data

class Food101TripletDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.food101_dataset = datasets.Food101(root=root, split=split, transform=transform)
        self.labels_to_indices = self._create_label_indices()
        self.triplets = self._generate_triplets()
        self.classes = self.food101_dataset.classes

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx, anchor_label, positive_label, negative_label = self.triplets[idx]

        anchor_image, _ = self.food101_dataset[anchor_idx]
        positive_image, _ = self.food101_dataset[positive_idx]
        negative_image, _ = self.food101_dataset[negative_idx]

        return anchor_image, positive_image, negative_image, anchor_label, positive_label, negative_label

    def _create_label_indices(self):
        print("Creating label indices...")
        labels_to_indices = {}
        total_samples = len(self.food101_dataset)
        for idx, (_, label) in enumerate(self.food101_dataset):
            if label not in labels_to_indices:
                labels_to_indices[label] = []
            labels_to_indices[label].append(idx)
            if idx % 1000 == 0:
                print(f"Processed {idx}/{total_samples} samples for label indices creation.")
        print("Label indices created.")
        return labels_to_indices

    def _generate_triplets(self):
        print("Generating triplets...")
        triplets = []
        labels_to_indices = self.labels_to_indices

        label_set = list(labels_to_indices.keys())
        total_samples = len(self.food101_dataset)
        
        for anchor_idx, (_, anchor_label) in enumerate(self.food101_dataset):
            positive_indices = labels_to_indices[anchor_label]
            negative_label = anchor_label

            while negative_label == anchor_label:
                negative_label = random.choice(label_set)

            negative_indices = labels_to_indices[negative_label]

            positive_idx = random.choice(positive_indices)
            negative_idx = random.choice(negative_indices)
            
            # Get the correct labels using indices
            positive_label = self.food101_dataset[positive_idx][1]
            negative_label = self.food101_dataset[negative_idx][1]

            triplets.append((anchor_idx, positive_idx, negative_idx, anchor_label, positive_label, negative_label))
            if anchor_idx % 1000 == 0:
                print(f"Generated triplets for {anchor_idx}/{total_samples} samples.")
        print("Triplets generated.")
        return triplets


class Food500TripletDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        # self.food101_dataset =data.MyDataset('/home/root-user/projects/food_kd/kd_food/data/isiafood500/metadata_ISIAFood_500/train_full.txt','/home/root-user/projects/food_kd/kd_food/data/isiafood500/ISIA_Food500/images', transform=transform)
        self.food101_dataset, self.test_food500 = data.load_dataset(train_dataset_path='/home/root-user/projects/food_kd/kd_food/data/isiafood500/metadata_ISIAFood_500/train_full.txt',test_dataset_path='/home/root-user/projects/food_kd/kd_food/data/isiafood500/metadata_ISIAFood_500/test_private.txt',batch_size=32)

        self.labels_to_indices = self._create_label_indices()
        self.triplets = self._generate_triplets()

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx, anchor_label, positive_label, negative_label = self.triplets[idx]

        anchor_image, _ = self.food101_dataset[anchor_idx]
        positive_image, _ = self.food101_dataset[positive_idx]
        negative_image, _ = self.food101_dataset[negative_idx]

        return anchor_image, positive_image, negative_image, anchor_label, positive_label, negative_label

    def _create_label_indices(self):
        print("Creating label indices...")
        labels_to_indices = {}
        total_samples = len(self.food101_dataset)
        for idx, (_, label) in enumerate(self.food101_dataset):
            if label not in labels_to_indices:
                labels_to_indices[label] = []
            labels_to_indices[label].append(idx)
            if idx % 1000 == 0:
                print(f"Processed {idx}/{total_samples} samples for label indices creation.")
        print("Label indices created.")
        return labels_to_indices

    def _generate_triplets(self):
        print("Generating triplets...")
        triplets = []
        labels_to_indices = self.labels_to_indices

        label_set = list(labels_to_indices.keys())
        total_samples = len(self.food101_dataset)
        
        for anchor_idx, (_, anchor_label) in enumerate(self.food101_dataset):
            positive_indices = labels_to_indices[anchor_label]
            negative_label = anchor_label

            while negative_label == anchor_label:
                negative_label = random.choice(label_set)

            negative_indices = labels_to_indices[negative_label]

            positive_idx = random.choice(positive_indices)
            negative_idx = random.choice(negative_indices)
            
            # Get the correct labels using indices
            positive_label = self.food101_dataset[positive_idx][1]
            negative_label = self.food101_dataset[negative_idx][1]

            triplets.append((anchor_idx, positive_idx, negative_idx, anchor_label, positive_label, negative_label))
            if anchor_idx % 1000 == 0:
                print(f"Generated triplets for {anchor_idx}/{total_samples} samples.")
        print("Triplets generated.")
        return triplets



    
    
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

def food500():
    print("Loading transforms...")
    train_transforms, test_transforms = load_transforms()
    print("Transforms loaded.")

    print("Loading datasets...")
    train_triplet_dataset = Food500TripletDataset(root='./data', split='train', transform=train_transforms)
    test_dataset, test_food500 = data.load_dataset(train_dataset_path='/home/root-user/projects/food_kd/kd_food/data/isiafood500/metadata_ISIAFood_500/train_full.txt',test_dataset_path='/home/root-user/projects/food_kd/kd_food/data/isiafood500/metadata_ISIAFood_500/test_private.txt',batch_size=32)
    # train_triplet_dataset = Food101TripletDataset(root='./data', split='train', transform=train_transforms)
    # test_dataset = datasets.Food101(root='./data', split='test', transform=test_transforms)
    print("Datasets loaded.")

    return train_triplet_dataset, test_dataset

def food101():
    print("Loading transforms...")
    train_transforms, test_transforms = load_transforms()
    print("Transforms loaded.")

    print("Loading datasets...")
    train_triplet_dataset = Food101TripletDataset(root='./data', split='train', transform=train_transforms)
    test_dataset = datasets.Food101(root='./data', split='test', transform=test_transforms)
    print("Datasets loaded.")

    return train_triplet_dataset, test_dataset

if __name__ == "__main__":
    train_food101, test_food101 = food500()
    print('dataset prepared')
    total_sample = len(train_food101)
    train_dataloader = DataLoader(dataset=train_food101, batch_size=32, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(dataset=test_food101, batch_size=32, shuffle=True, num_workers=8)

    print('success')