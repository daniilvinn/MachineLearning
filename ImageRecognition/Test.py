import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import sys
import os
import glob
import re
from Network import WideMultiCNN

def get_food101_labels():
    """
    Get the actual Food-101 class names in the correct alphabetical order
    This matches the order used by PyTorch's Food101 dataset
    """
    food_classes = [
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
    return food_classes

def get_label_names():
    """
    Get the same label names used during training - matches Dataset.py exactly
    """
    all_labels = []
    
    # MNIST labels
    mnist_labels = [f'digit_{i}' for i in range(10)]
    all_labels.extend(mnist_labels)
    
    # Fashion-MNIST labels  
    fashion_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']
    all_labels.extend(fashion_labels)
    
    # CIFAR-10 labels
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    all_labels.extend(cifar10_labels)
    
    # CIFAR-100 labels
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
    
    # SVHN labels
    svhn_labels = [f'svhn_digit_{i}' for i in range(10)]
    all_labels.extend(svhn_labels)
    
    # STL-10 labels
    stl10_labels = ['stl_airplane', 'stl_bird', 'stl_car', 'stl_cat', 'stl_deer', 'stl_dog', 'stl_horse', 'stl_monkey', 'stl_ship', 'stl_truck']
    all_labels.extend(stl10_labels)
    
    # Food-101 labels (real food names)
    food101_labels = get_food101_labels()
    all_labels.extend(food101_labels)
    
    # Flowers-102 labels
    flowers_labels = [f'flower_{i:03d}' for i in range(102)]
    all_labels.extend(flowers_labels)
    
    # Oxford-IIIT Pet labels
    pets_labels = [f'pet_{i:02d}' for i in range(37)]
    all_labels.extend(pets_labels)
    
    # Caltech-101 labels
    caltech101_labels = [f'caltech101_{i:03d}' for i in range(102)]
    all_labels.extend(caltech101_labels)
    
    # Caltech-256 labels
    caltech256_labels = [f'caltech256_{i:03d}' for i in range(257)]
    all_labels.extend(caltech256_labels)
    
    return all_labels

def find_latest_checkpoint(checkpoint_pattern='wide_multi_cnn_model_epoch_*.pth'):
    """
    Find the latest checkpoint file based on epoch number
    """
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        # Fall back to the original model file
        if os.path.exists('wide_multi_cnn_model.pth'):
            return 'wide_multi_cnn_model.pth'
        else:
            return None
    
    # Extract epoch numbers and find the latest
    latest_epoch = 0
    latest_file = None
    
    for file in checkpoint_files:
        # Extract epoch number from filename like "wide_multi_cnn_model_epoch_042.pth"
        match = re.search(r'epoch_(\d+)', file)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_file = file
    
    return latest_file

def get_num_classes_from_checkpoint(checkpoint_path):
    """
    Determine the number of classes from a saved checkpoint
    """
    try:
        # Load checkpoint to examine the final layer
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Look for the final classifier layer weight
        if 'classifier.6.weight' in checkpoint:
            return checkpoint['classifier.6.weight'].shape[0]
        elif 'classifier.7.weight' in checkpoint:
            return checkpoint['classifier.7.weight'].shape[0]
        else:
            # Try to find the last linear layer
            for key in checkpoint.keys():
                if 'weight' in key and len(checkpoint[key].shape) == 2:
                    # This is likely the final classification layer
                    return checkpoint[key].shape[0]
        
        print("Warning: Could not determine number of classes from checkpoint")
        return None
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")
        return None

def load_model(model_path=None):
    """
    Load the trained model - automatically finds latest checkpoint if no path specified
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find the model file to load
    if model_path is None:
        model_path = find_latest_checkpoint()
        if model_path:
            print(f"Found latest checkpoint: {model_path}")
        else:
            print("No checkpoint files found!")
            print("Please train the model first by running: python Network.py")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first by running: python Network.py")
        sys.exit(1)
    
    # Determine correct number of classes from checkpoint
    num_classes = get_num_classes_from_checkpoint(model_path)
    if num_classes is None:
        print("Could not determine number of classes from checkpoint, falling back to full label set")
        label_names = get_label_names()
        num_classes = len(label_names)
    
    print(f"Creating model with {num_classes} classes")
    model = WideMultiCNN(num_classes=num_classes)
    
    try:
        if device.type == 'cuda':
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Model loaded successfully from {model_path}")
        print(f"Model supports {num_classes} classes")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to a mismatch in model architecture.")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    return model, device, num_classes

def preprocess_image(image_path):
    """
    Preprocess the input image to match training format (64x64)
    """
    try:
        # Load original image (keep full resolution for display)
        original_image = Image.open(image_path)
        
        # Convert to RGB if needed
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Create resized version for model prediction (64x64 to match training)
        resized_image = original_image.resize((64, 64), Image.LANCZOS)
        
        # Convert resized image to numpy array
        img_array = np.array(resized_image, dtype=np.float32)
        
        # Normalize to [-1, 1] range (same as training)
        img_array = img_array / 255.0
        img_array = (img_array - 0.5) / 0.5
        
        # Convert to PyTorch tensor and add batch dimension
        img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor, original_image
        
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

def predict_image(model, img_tensor, device):
    """
    Make prediction on the preprocessed image
    """
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
        
    return predicted.item(), confidence.item(), top5_indices[0].cpu().numpy(), top5_probs[0].cpu().numpy()

def plot_result(original_image, predicted_label, confidence, top5_labels, top5_probs, label_names):
    """
    Plot the full resolution image with prediction results
    """
    # Get image dimensions for better figure sizing
    img_width, img_height = original_image.size
    aspect_ratio = img_height / img_width
    
    # Adjust figure size based on image aspect ratio
    if aspect_ratio > 1:  # Tall image
        fig_width = 15
        fig_height = max(8, 8 * aspect_ratio * 0.6)
    else:  # Wide image
        fig_width = 15
        fig_height = 8
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Plot original full-resolution image
    ax1.imshow(original_image)
    ax1.set_title(f'Input Image (Full Resolution: {img_width}x{img_height})\nPredicted: {predicted_label}\nConfidence: {confidence:.1f}%', 
                  fontsize=12, pad=20)
    ax1.axis('off')
    
    # Plot top 5 predictions
    labels = [label_names[idx] for idx in top5_labels]
    probs = top5_probs * 100
    
    bars = ax2.barh(range(len(labels)), probs)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Top 5 Predictions', fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3)
    
    # Color the highest prediction differently
    bars[0].set_color('green')
    for i in range(1, len(bars)):
        bars[i].set_color('lightblue')
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test image recognition with trained CNN model')
    parser.add_argument('filename', help='Path to the image file to recognize')
    parser.add_argument('--model', default=None, 
                       help='Path to the trained model file (default: auto-detect latest checkpoint)')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.filename):
        print(f"Error: Image file '{args.filename}' not found!")
        sys.exit(1)
    
    print(f"Loading model...")
    model, device, num_classes = load_model(args.model)
    
    # Get the appropriate label names for the number of classes in the model
    all_label_names = get_label_names()
    label_names = all_label_names[:num_classes]  # Only use the first num_classes labels
    
    print(f"Processing image: {args.filename}")
    img_tensor, original_image = preprocess_image(args.filename)
    
    print("Making prediction...")
    predicted_idx, confidence, top5_indices, top5_probs = predict_image(model, img_tensor, device)
    
    predicted_label = label_names[predicted_idx]
    
    print(f"\nPrediction Results:")
    print(f"Top prediction: {predicted_label}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"\nTop 5 predictions:")
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
        print(f"{i+1}. {label_names[idx]}: {prob*100:.1f}%")
    
    # Plot results
    plot_result(original_image, predicted_label, confidence*100, 
                top5_indices, top5_probs, label_names)

if __name__ == "__main__":
    main()
