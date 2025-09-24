import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np

class UnifiedDataset(torch.utils.data.Dataset):
    """
    Wrapper to unify multiple datasets with different label mappings
    """
    def __init__(self, dataset, label_offset=0, target_transform=None):
        self.dataset = dataset
        self.label_offset = label_offset
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Add label offset to create unique labels across datasets
        label = label + self.label_offset
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_multi_datasets(batch_size=64):
    """
    Download and prepare multiple datasets with 1000+ classes
    Creates unified label library with extensive data augmentation
    """
    
    # Data augmentation for training - larger images (64x64)
    train_transform = transforms.Compose([
        transforms.Resize((72, 72)),  # Resize larger first
        transforms.RandomCrop((64, 64)),  # Then random crop to target size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Grayscale to RGB transform for MNIST-like datasets
    train_transform_gray = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.RandomCrop((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform_gray = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download all datasets
    datasets_train = []
    datasets_test = []
    all_labels = []
    current_offset = 0
    
    print("Downloading MNIST...")
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform_gray)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform_gray)
    mnist_labels = [f'digit_{i}' for i in range(10)]
    all_labels.extend(mnist_labels)
    datasets_train.append(UnifiedDataset(mnist_train, label_offset=current_offset))
    datasets_test.append(UnifiedDataset(mnist_test, label_offset=current_offset))
    current_offset += 10
    
    print("Downloading Fashion-MNIST...")
    fashion_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform_gray)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform_gray)
    fashion_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']
    all_labels.extend(fashion_labels)
    datasets_train.append(UnifiedDataset(fashion_train, label_offset=current_offset))
    datasets_test.append(UnifiedDataset(fashion_test, label_offset=current_offset))
    current_offset += 10
    
    print("Downloading CIFAR-10...")
    cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    all_labels.extend(cifar10_labels)
    datasets_train.append(UnifiedDataset(cifar10_train, label_offset=current_offset))
    datasets_test.append(UnifiedDataset(cifar10_test, label_offset=current_offset))
    current_offset += 10
    
    print("Downloading CIFAR-100...")
    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    cifar100_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                       'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                       'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                       'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                       'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                       'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                       'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                       'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                       'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                       'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                       'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                       'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                       'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                       'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    all_labels.extend(cifar100_labels)
    datasets_train.append(UnifiedDataset(cifar100_train, label_offset=current_offset))
    datasets_test.append(UnifiedDataset(cifar100_test, label_offset=current_offset))
    current_offset += 100
    
    print("Downloading SVHN...")
    try:
        svhn_train = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
        svhn_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)
        svhn_labels = [f'svhn_digit_{i}' for i in range(10)]
        all_labels.extend(svhn_labels)
        datasets_train.append(UnifiedDataset(svhn_train, label_offset=current_offset))
        datasets_test.append(UnifiedDataset(svhn_test, label_offset=current_offset))
        current_offset += 10
    except:
        print("SVHN download failed, skipping...")
    
    print("Downloading STL-10...")
    try:
        stl10_train = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=train_transform)
        stl10_test = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=test_transform)
        stl10_labels = ['stl_airplane', 'stl_bird', 'stl_car', 'stl_cat', 'stl_deer', 'stl_dog', 'stl_horse', 'stl_monkey', 'stl_ship', 'stl_truck']
        all_labels.extend(stl10_labels)
        datasets_train.append(UnifiedDataset(stl10_train, label_offset=current_offset))
        datasets_test.append(UnifiedDataset(stl10_test, label_offset=current_offset))
        current_offset += 10
    except:
        print("STL-10 download failed, skipping...")
    
    print("Downloading Food-101...")
    try:
        food101_train = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=train_transform)
        food101_test = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=test_transform)
        # Food-101 has 101 food classes with real food names
        food101_labels = [
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
            'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
            'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
            'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
            'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
            'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
            'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
            'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
            'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
            'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
            'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
            'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
            'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
            'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
            'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
            'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
            'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
            'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
            'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
        ]
        all_labels.extend(food101_labels)
        datasets_train.append(UnifiedDataset(food101_train, label_offset=current_offset))
        datasets_test.append(UnifiedDataset(food101_test, label_offset=current_offset))
        current_offset += 101
    except:
        print("Food-101 download failed, skipping...")
    
    print("Downloading Flowers-102...")
    try:
        flowers_train = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=train_transform)
        flowers_test = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=test_transform)
        flowers_labels = [f'flower_{i:03d}' for i in range(102)]
        all_labels.extend(flowers_labels)
        datasets_train.append(UnifiedDataset(flowers_train, label_offset=current_offset))
        datasets_test.append(UnifiedDataset(flowers_test, label_offset=current_offset))
        current_offset += 102
    except:
        print("Flowers-102 download failed, skipping...")
    
    print("Downloading Oxford-IIIT Pet Dataset...")
    try:
        pets_train = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=train_transform)
        pets_test = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=test_transform)
        pets_labels = [f'pet_{i:02d}' for i in range(37)]
        all_labels.extend(pets_labels)
        datasets_train.append(UnifiedDataset(pets_train, label_offset=current_offset))
        datasets_test.append(UnifiedDataset(pets_test, label_offset=current_offset))
        current_offset += 37
    except:
        print("Oxford-IIIT Pet download failed, skipping...")
    
    print("Downloading Caltech-101...")
    try:
        caltech101 = torchvision.datasets.Caltech101(root='./data', download=True, transform=train_transform)
        # Split manually since Caltech101 doesn't have train/test splits
        train_size = int(0.8 * len(caltech101))
        test_size = len(caltech101) - train_size
        caltech101_train, caltech101_test = torch.utils.data.random_split(caltech101, [train_size, test_size])
        caltech101_labels = [f'caltech101_{i:03d}' for i in range(102)]  # 101 + background
        all_labels.extend(caltech101_labels)
        datasets_train.append(UnifiedDataset(caltech101_train, label_offset=current_offset))
        datasets_test.append(UnifiedDataset(caltech101_test, label_offset=current_offset))
        current_offset += 102
    except:
        print("Caltech-101 download failed, skipping...")
    
    print("Downloading Caltech-256...")
    try:
        caltech256 = torchvision.datasets.Caltech256(root='./data', download=True, transform=train_transform)
        # Split manually
        train_size = int(0.8 * len(caltech256))
        test_size = len(caltech256) - train_size
        caltech256_train, caltech256_test = torch.utils.data.random_split(caltech256, [train_size, test_size])
        caltech256_labels = [f'caltech256_{i:03d}' for i in range(257)]
        all_labels.extend(caltech256_labels)
        datasets_train.append(UnifiedDataset(caltech256_train, label_offset=current_offset))
        datasets_test.append(UnifiedDataset(caltech256_test, label_offset=current_offset))
        current_offset += 257
    except:
        print("Caltech-256 download failed, skipping...")
    
    # Combine all available datasets
    combined_train = ConcatDataset(datasets_train)
    combined_test = ConcatDataset(datasets_test)
    
    # Create data loaders
    train_loader = DataLoader(
        combined_train, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        combined_test, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    print(f"Total training samples: {len(combined_train)}")
    print(f"Total test samples: {len(combined_test)}")
    print(f"Total classes: {len(all_labels)}")
    
    return train_loader, test_loader, all_labels
