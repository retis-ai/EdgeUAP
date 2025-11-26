from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import random
import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, WeightedRandomSampler


'''
TODO
- Refactor the codebase 
- add function documentation.
'''




CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
####################################### UTILS Function ######################################################
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------


#-------------------------------------------------------
# Attack info
#-------------------------------------------------------
class AttackInfo:
    def __init__(self):
        self.model_results = {
            'model_name': None, 
            'depth': [],
            'res': [],
            'attack_info': {
                'uap': [],              # Store UAP model instances directly
                'target_class': None,
                'epsilon': None,
                'samples_mlp_target': None,
                'samples_mlp_others': None,
                'samples_attack_dataset': None,
                'attack_type': None
            }
        }

    def update_attack_info(self, model_name=None, depth=None, res=None, uap_models=None, target_class=None,
                           epsilon=None, samples_mlp_target=None, samples_mlp_others=None,
                           samples_attack_dataset=None, attack_type=None):
        if model_name is not None: 
            self.model_results['model_name'] = model_name
        if depth is not None:
            self.model_results['depth'].append(depth)
        if res is not None:
            self.model_results['res'].append(res)

        attack_info = self.model_results['attack_info']
        if uap_models is not None:
            attack_info['uap'].append(uap_models.to('cpu'))  # Store UAP models directly in the dictionary
        if target_class is not None:
            attack_info['target_class'] = target_class
        if epsilon is not None:
            attack_info['epsilon'] = epsilon
        if samples_mlp_target is not None:
            attack_info['samples_mlp_target'] = samples_mlp_target
        if samples_mlp_others is not None:
            attack_info['samples_mlp_others'] = samples_mlp_others
        if samples_attack_dataset is not None:
            attack_info['samples_attack_dataset'] = samples_attack_dataset
        if attack_type is not None:
            attack_info['attack_type'] = attack_type

    def save_as_pickle(self, filename="attack_info.pkl"):
        # Save the entire AttackInfo object as a pickle file
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Attack information saved to {filename}")

    @staticmethod
    def load_from_pickle(filename="attack_info.pkl"):
        # Load the AttackInfo object from a pickle file
        with open(filename, "rb") as f:
            loaded_attack_info = pickle.load(f)
        print(f"Attack information loaded from {filename}")
        return loaded_attack_info




#-----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
def split_dataset_by_class(dataset, target_class, samples_no_classes, samples_classes):
    if isinstance(target_class, int):
        target_class = [target_class]

    if isinstance(dataset, Subset):
        labels = torch.tensor(dataset.dataset.targets)
    elif isinstance(dataset, Dataset):
        labels = torch.tensor(dataset.targets)
        
    mask_class = torch.isin(labels, torch.tensor(target_class))
    indices_class = torch.nonzero(mask_class).squeeze().tolist()
    indices_other = torch.nonzero(~mask_class).squeeze().tolist()

    indices_class = np.random.choice(indices_class, samples_classes, replace=False)
    indices_other = np.random.choice(indices_other, samples_no_classes, replace=False)
        
    subset_class = Subset(dataset, indices_class)
    subset_other = Subset(dataset, indices_other)
    return subset_class, subset_other

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def create_balanced_dataloader(dataset, labels, batch_size):
    """
    Creates a DataLoader with weighted sampling to handle class imbalance.
    Args:
        dataset (torch.utils.data.Dataset): The dataset.
        labels (list or torch.Tensor): The list of labels corresponding to each sample in the dataset.
        batch_size (int): The batch size for the DataLoader.
    Returns:
        DataLoader: A DataLoader that samples in a balanced way based on class weights.
    """
    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    dataloader = DataLoader(dataset,  batch_size=batch_size, num_workers=2, sampler=sampler)
    
    return dataloader



#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def map_subset_indices_to_original(subset):
    """
    Maps the indices of a Subset back to the original dataset's indices.
    Args:
        subset (torch.utils.data.Subset): The input subset, which may be a subset of a subset.
    Returns:
        list: A list of indices that map back to the original dataset.
    """
    # Recursively retrieve the dataset and indices from nested subsets
    original_dataset = subset.dataset
    original_indices = subset.indices
    
    # If the original dataset is also a Subset, map its indices recursively
    if isinstance(original_dataset, Subset):
        mapped_indices = [i for i in original_indices]
        return map_subset_indices_to_original(Subset(original_dataset.dataset, mapped_indices))
    
    # If the original dataset is not a Subset, return the indices as they map to it
    return original_indices



#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def concat_subsets(subset1, subset2):
    """
    Concatenates two subsets into one new subset by combining their indices.
    Args:
        subset1 (torch.utils.data.Subset): The first subset.
        subset2 (torch.utils.data.Subset): The second subset.
    Returns:
        Subset: A new subset containing samples from both subsets.
    """
    # Map the indices of both subsets to the original dataset
    mapped_indices1 = map_subset_indices_to_original(subset1)
    mapped_indices2 = map_subset_indices_to_original(subset2)
    
    # Combine the indices from both subsets
    combined_indices = mapped_indices1 + mapped_indices2
    #combined_indices = subset1.indices + subset2.indices
    
    # Create a new subset using the original dataset and the combined indices
    original_dataset = subset1.dataset if not isinstance(subset1.dataset, Subset) else subset1.dataset.dataset
    new_subset = Subset(original_dataset, combined_indices)
    
    return new_subset

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def split_dataset(dataset, split_ratio):
    if isinstance(dataset, Subset):
        targets = np.array(dataset.dataset.targets)
    elif isinstance(dataset, Dataset):
        targets = np.array(dataset.targets)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio[1], train_size=split_ratio[0], random_state=42)
    train_indices, test_indices = next(sss.split(np.zeros(len(targets)), targets))
    subset_1 = Subset(dataset, train_indices)
    subset_2 = Subset(dataset, test_indices)
    return subset_1, subset_2

# Define a function to load Imagenet-10 data
def load_imagenet_data():
    # TODO capire come mette uguali le resize e RandomResize
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # Crop the image to 224x224 pixels randomly.
        #transforms.RandomHorizontalFlip(),  # Horizontally flip the image randomly with a given probability (default=0.5).
        transforms.ToTensor(),
    ])
    train_dataset =  datasets.ImageFolder(root='/home/datasets/Imagenet/train', transform=transform)
    test_dataset =  datasets.ImageFolder(root='/home/datasets/Imagenet/val', transform=transform)
    return train_dataset, test_dataset


def load_cifar10_data():
    from torchvision.datasets import CIFAR10
    # TODO capire come mette uguali le resize e RandomResize
    transform = transforms.Compose([
        #transforms.Resize((32,224)),  # Crop the image to 224x224 pixels randomly.
        #transforms.RandomHorizontalFlip(),  # Horizontally flip the image randomly with a given probability (default=0.5).
        transforms.ToTensor(),
    ])
    train_dataset =  CIFAR10(root="./data", train=True, download=True)
    test_dataset =  CIFAR10(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def get_model_and_info(MODEL_NAME):
    normalize = Normalizer(mean = IMGNET_MEAN, std = IMGNET_STD)
    
    if MODEL_NAME == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model = nn.Sequential(normalize, model)
        layers_info = [
        (model[1].features[4], 32, 'features-4') , 
        (model[1].features[8], 64, 'features-8' ) , 
        (model[1].features[12], 96, 'features-12') , 
        (model[1].features[18], 1280, 'features-18') ,
        (model[1].classifier, 1000, 'cls')  ]
    
    
    elif MODEL_NAME == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model = nn.Sequential(normalize, model)
        layers_info = [
        (model[1].features[9], 128, 'features-9' ) , 
        (model[1].features[16], 256, 'features-16' ) , 
        (model[1].features[23], 512, 'features-23') , 
        (model[1].features[30], 512, 'features-30') ,
        (model[1].classifier, 1000, 'cls') ]
    
    
    elif MODEL_NAME == 'vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained=True)
        model = nn.Sequential(normalize, model)
        layers_info = [
        (model[1].features[13], 128, 'features-13' ) , 
        (model[1].features[23], 256, 'features-23' ) , 
        (model[1].features[33], 512, 'features-33') , 
        (model[1].features[43], 512, 'features-43') ,
        (model[1].classifier, 1000, 'cls')  ]
        
    
    
    elif MODEL_NAME == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model = nn.Sequential(normalize, model)
        layers_info = [
        (model[1].layer1, 256, 'features-256' ) , 
        (model[1].layer2, 512, 'features-512' ) , 
        (model[1].layer3, 1024, 'features-1024') , 
        (model[1].layer4, 2048, 'features-2048') ,
        (model[1].fc, 1000, 'cls') ]

    elif MODEL_NAME == 'wide_resnet101':
        model = torchvision.models.wide_resnet101_2(pretrained=True)
        model = nn.Sequential(normalize, model)
        layers_info = [
        (model[1].layer1, 256, 'features-256' ) , 
        (model[1].layer2, 512, 'features-512' ) , 
        (model[1].layer3, 1024, 'features-1024') , 
        (model[1].layer4, 2048, 'features-2048') ,
        (model[1].fc, 1000, 'cls') ]


    elif MODEL_NAME == 'vit_b':
        # Load pretrained ViT model
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model = nn.Sequential(normalize, model)
        layers_info = [
        (model[1].blocks[1], 768, 'features-256' ) , 
        (model[1].blocks[4], 768, 'features-512' ) , 
        (model[1].blocks[6], 768, 'features-1024') , 
        (model[1].blocks[10], 768, 'features-2048') ]


    elif MODEL_NAME == 'vit_s':
        # Load pretrained ViT model
        model = timm.create_model('vit_small_patch16_224', pretrained=True)
        model = nn.Sequential(normalize, model)
        layers_info = [
        (model[1].blocks[1], 384, 'features-256' ) , 
        (model[1].blocks[4], 384, 'features-512' ) , 
        (model[1].blocks[6], 384, 'features-1024') , 
        (model[1].blocks[10], 384, 'features-2048') ]

    return model, layers_info




def get_model_and_info_cifar10(MODEL_NAME, model_path):
    #normalize = Normalizer(mean = IMGNET_MEAN, std = IMGNET_STD)
    num_classes = 10
    
    if MODEL_NAME == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(1280, num_classes)
        model.load_state_dict(torch.load(model_path))
        model = nn.Sequential(nn.Identity(), model)
        layers_info = [
        (model[1].features[4], 32, 'features-4') , 
        (model[1].features[8], 64, 'features-8' ) , 
        (model[1].features[12], 96, 'features-12') , 
        (model[1].features[18], 1280, 'features-18') ,
        (model[1].classifier, 10, 'cls')  ]
    
    
    elif MODEL_NAME == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
        model.load_state_dict(torch.load(model_path))
        model = nn.Sequential(nn.Identity(), model)
        layers_info = [
        (model[1].features[9], 128, 'features-9' ) , 
        (model[1].features[16], 256, 'features-16' ) , 
        (model[1].features[23], 512, 'features-23') , 
        (model[1].features[30], 512, 'features-30') ,
        (model[1].classifier, 10, 'cls') ]
    
    
    
    elif MODEL_NAME == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
        model.load_state_dict(torch.load(model_path))
        model = nn.Sequential(nn.Identity(), model)
        layers_info = [
        (model[1].layer1, 256, 'features-256' ) , 
        (model[1].layer2, 512, 'features-512' ) , 
        (model[1].layer3, 1024, 'features-1024') , 
        (model[1].layer4, 2048, 'features-2048') ,
        (model[1].fc, 10, 'cls') ]
    
    return model, layers_info

    

# Guardare da se...
# vitb_16_224_model_layers = [
#     (model[1].blocks[2], 128 ) , 
#     (model[1].features[5], 256 ) , 
#     (model[1].features[8], 512) , 
#     (model[1].features[11], 512) ,
#     (model[1].head, 1000) ]






#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)
    
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
        
    
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)
normalize = Normalizer(mean = IMGNET_MEAN, std = IMGNET_STD)



#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def get_activation(name, transformers = False, use_class_token=False):
    def hook(model, input, output):
        if transformers: 
            batch_size, num_tokens, embedding_size = output.shape
            if use_class_token: 
                patch_tokens = output[:, 0, :]  
                output = patch_tokens.view(batch_size, -1)
            else:
                output = output[:, 1:, :]  
                h = w = int(num_tokens ** 0.5)
                output = output.view(batch_size, h, w, embedding_size)
                output = output.permute(0, 3, 1, 2)
        activation[name] = output#.detach()
    return hook


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
# dynamic_transform = transforms.Compose([
#         transforms.Resize(224),
#         transforms.RandomRotation(20),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomCrop(224, padding=10),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
#     ])


dynamic_transform = transforms.Compose([
        transforms.RandomCrop(224, padding=5),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    ])

dynamic_transform_cifar10 = transforms.Compose([
        transforms.RandomCrop(32, padding=5),
        #transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    ])

class EnsembleLinears(nn.Module):
    def __init__(self, num_neurons, num_classes, num_linears, num_classifier_conv=1):
        super(EnsembleLinears, self).__init__()
        if num_classifier_conv == 1:
            self.backbone = nn.Sequential(
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1),      
                    nn.AdaptiveAvgPool2d(1), 
                    nn.Flatten())
        elif num_classifier_conv == 2:
            self.backbone = nn.Sequential(
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1), 
                    nn.ReLU(),
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1),      
                    nn.AdaptiveAvgPool2d(1), 
                    nn.Flatten())
        elif num_classifier_conv == 4:
            self.backbone = nn.Sequential(
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1), 
                    nn.ReLU(),
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1), 
                    nn.ReLU(),
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1), 
                    nn.ReLU(),
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1),      
                    nn.AdaptiveAvgPool2d(1), 
                    nn.Flatten())
        elif num_classifier_conv == 6:
            self.backbone = nn.Sequential(
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1), 
                    nn.ReLU(),
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1), 
                    nn.ReLU(),
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1), 
                    nn.ReLU(),
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1), 
                    nn.ReLU(),
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1), 
                    nn.ReLU(),
                    nn.Conv2d(num_neurons, num_neurons, 3, padding=1, stride=1),      
                    nn.AdaptiveAvgPool2d(1), 
                    nn.Flatten())
            
        self.linear_cls = nn.Sequential(
                nn.Linear(num_neurons, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes))
    def forward(self, x):
        out = x
        if len(x.shape) > 2: 
            out = self.backbone(x)
        out = self.linear_cls(out)
        return out

class UniversalPerturbation(nn.Module):
    def __init__(self, shape, epsilon):
        super(UniversalPerturbation, self).__init__()
        self.perturbation = nn.Parameter(torch.ones(shape) * (epsilon/2), requires_grad=True) 

    def forward(self, x):
        if self.training:
            perturbed = x + self.perturbation
        else:
            perturbed = torch.clamp(x + self.perturbation, 0, 1)
        return perturbed

    def update_grad_inf(self):
        self.perturbation.grad.data = self.perturbation.grad.data.sign()


    def update_grad_l2(self):
        grad = self.perturbation.grad.data
        grad_norm = torch.norm(grad, p=2)
        grad_normalized = grad / (grad_norm +  1e-10)
        self.perturbation.grad.data = grad_normalized


def project_perturbation_inf(perturbation, epsilon):
    with torch.no_grad():
        # Project the perturbation within the epsilon-ball (L-infinity norm)
        perturbation.clamp_(-epsilon, epsilon)



def project_perturbation_l2(perturbation, epsilon):
    """
    Projects a universal perturbation onto the L2 epsilon-ball.

    Args:
        perturbation (torch.Tensor): The universal perturbation tensor.
        epsilon (float): The L2 norm constraint.

    Returns:
        torch.Tensor: The projected perturbation within the L2 norm constraint.
    """
    # Calculate the L2 norm of the perturbation
    perturbation_norm = torch.norm(perturbation.view(-1), p=2)

    # If the perturbation norm exceeds epsilon, scale it back to the epsilon-ball
    if perturbation_norm > epsilon:
        perturbation = perturbation * (epsilon / (perturbation_norm + 1e-10))
    
    return perturbation



#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

activation = {}

def reset_activation():
    global activation 
    activation = {}

def evaluate_uap(test_loader, model, perturbation_model, target_label, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    attacked_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Filter out samples with the target label
            if target_label is not None: 
                mask = labels != target_label
                images, labels = images[mask], labels[mask]

            if images.size(0) == 0:  # Skip if no images remain after filtering
                continue

            # Apply the universal perturbation to the test images
            perturbed_images = perturbation_model(images)
            
            # Get model predictions
            outputs = model(perturbed_images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Count correctly predicted labels
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if target_label is not None: 
                attacked_correct += (predicted == target_label).sum().item()
    
    # Calculate and return accuracy
    accuracy = 100 * correct / total if total > 0 else 0  # Handle case of zero samples
    target_accuracy = 100 * attacked_correct / total if total > 0 else 0 
    return accuracy, target_accuracy


def plot_multiple_perturbations(images, perturbation_models, device, model_names):
    """
    Applies multiple perturbation models to a set of images, plots the original images,
    perturbed images for each model, and the perturbations alone in a grid.
    
    Args:
        images (list of torch.Tensor): List of images in tensor format.
        perturbation_models (list of torch.nn.Module): List of perturbation models.
        device (torch.device): The device on which to perform computations.
        model_names (list of str): List of names for each perturbation model (one name per model).
    """
    # Ensure images are on the correct device
    images = [img.to(device) for img in images]
    num_models = len(perturbation_models)
    num_images = len(images)

    # Set up the figure grid
    fig, axs = plt.subplots(num_models, 2 * num_images + 1, figsize=(4 * (2 * num_images + 1), 4 * num_models))
    
    # Iterate over each perturbation model
    for model_idx, (perturbation_model, model_name) in enumerate(zip(perturbation_models, model_names)):
        # Apply the perturbation model to each image and calculate the perturbation
        perturbed_images = [perturbation_model(img.unsqueeze(0)).squeeze(0) for img in images]
        perturbations = [perturbed_img - img for img, perturbed_img in zip(images, perturbed_images)]
        
        # Plot each original, perturbed, and perturbation image for this model
        for i in range(num_images):
            # Extract the original, perturbed, and perturbation images
            original = images[i].detach().cpu().permute(1, 2, 0).numpy()
            perturbed = perturbed_images[i].detach().cpu().permute(1, 2, 0).numpy()
            perturbation = perturbations[i].detach().cpu().permute(1, 2, 0).numpy()
            
            # Normalize perturbation for better visualization
            perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
            
            # Plot original and perturbed images in adjacent columns
            axs[model_idx, 2 * i].imshow(original)
            axs[model_idx, 2 * i].axis('off')
            axs[model_idx, 2 * i + 1].imshow(perturbed)
            axs[model_idx, 2 * i + 1].axis('off')
        
        # Plot only the perturbation in the last column
        axs[model_idx, 2 * num_images].imshow(perturbation, cmap='viridis')
        axs[model_idx, 2 * num_images].axis('off')
        
        # Set row labels for each model
        axs[model_idx, 0].set_ylabel(model_name, fontsize=14, fontweight='bold')
    
    # Set column titles for the first row only
    for i in range(num_images):
        axs[0, 2 * i].set_title(f'Original Image {i+1}', fontsize=14)
        axs[0, 2 * i + 1].set_title(f'Perturbed Image {i+1}', fontsize=14)
    axs[0, 2 * num_images].set_title('Perturbation Alone', fontsize=14)
    
    plt.tight_layout()
    plt.show()



def evaluate_uap_full(test_loader, model, perturbation_model, target_label, device, remove_target_samples = False):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    attacked_correct = 0
    total = 0
    top5_correct = 0
    target_in_top5 = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Filter out samples with the target label
            if remove_target_samples:
                mask = labels != target_label
                images, labels = images[mask], labels[mask]

            if images.size(0) == 0:  # Skip if no images remain after filtering
                continue

            # Apply the universal perturbation to the test images
            perturbed_images = perturbation_model(images)
            
            # Get model predictions
            outputs = model(perturbed_images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Top-5 predictions
            top5_pred = torch.topk(outputs, 5, dim=1).indices
            
            # Count correctly predicted labels
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            attacked_correct += (predicted == target_label).sum().item()
            
            # Check for top-5 accuracy
            top5_correct += (top5_pred == labels.view(-1, 1)).sum().item()
            
            # Check if the target label is in the top-5 predictions
            target_in_top5 += (top5_pred == target_label).sum().item()
    
    # Calculate accuracies and the target-in-top-5 score
    accuracy = 100 * correct / total if total > 0 else 0
    target_accuracy = 100 * attacked_correct / total if total > 0 else 0
    top5_accuracy = 100 * top5_correct / total if total > 0 else 0
    target_in_top5_score = 100 * target_in_top5 / total if total > 0 else 0
    
    return accuracy, target_accuracy, top5_accuracy, target_in_top5_score
        


def get_cifar10_subset(train=True, subset_size=1000):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Rescale to 224x224
        transforms.ToTensor(),         # Convert to tensor
    ])

    dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    # Randomly select a subset of the dataset
    indices = random.sample(range(len(dataset)), subset_size)
    subset = Subset(dataset, indices)
    return subset



#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
####################################### ATTACK CODE ######################################################
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def target_mlp_optimization(model, 
                            layer_channels, 
                            target_class, 
                            target_loader, 
                            test_loader, 
                            layer_name, 
                            device, 
                            num_training_epochs=50, 
                            print_info = True, 
                            num_conv_blocks_mlp = 1, 
                            num_classifier_conv = 1):
    channels = layer_channels
    target_mlp =  EnsembleLinears(channels, 2, 1, num_conv_blocks_mlp)
    model.eval()
    target_mlp.train()
    target_mlp = target_mlp.to(device)
    optimizer = optim.Adam(target_mlp.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()  

    count_iter = 0

    print("train of the target-classifier")

    for epoch in range(num_training_epochs):
        target_mlp.train()  # Ensure the auxiliary head is in training mode
        running_loss = 0.0

        for images, labels in target_loader:  # Assuming train_loader provides labels
            count_iter += 1
            images, labels = images.to(device), labels.to(device)
    
            labels = (labels == target_class).long()

            if images.shape[2] == 32: 
                images = torch.stack([dynamic_transform_cifar10(input_img) for input_img in images])
            else:    
                images = torch.stack([dynamic_transform(input_img) for input_img in images])
    
            _ = model(images)
            activations = activation[layer_name].detach()
    
            pooled_activation = activations
            output = target_mlp(pooled_activation)
            loss = criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            running_loss += loss.item()


            if count_iter > 500:
                break
                

        if epoch % 20 == 0 and print_info == True:
            print(f'Epoch [{epoch + 1}/{num_training_epochs}], Loss: {running_loss / len(target_loader):.4f}')
            target_mlp.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in target_loader:
                    images, labels = images.to(device), labels.to(device)
                    _ = model(images)
                    labels = (labels == target_class).long()
                    activations = activation[layer_name].detach()
                    pooled_activation = activations
                    output = target_mlp(pooled_activation)
                    _, predicted = torch.max(output, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                accuracy = 100 * correct / total
                print(f'Test Accuracy MLP Classifier - after Epoch {epoch + 1}: {accuracy:.2f}%')

        if count_iter > 500:
            target_mlp.eval()
            break
    
    print()
    return target_mlp.eval()
    
        



#---------------------------------------------------------------------------------------------------------
def run_attacks_inf(model, 
                    layer_shape,
                    input_shape,
                    target_class, 
                    target_loader, 
                    opt_loader, 
                    test_loader, 
                    device,
                    num_mlp_training_epochs=50, 
                    num_attack_epochs =50, 
                    epsilon =16/255,
                    alpha = 2/255,
                    iter_stop = True, 
                    print_info = True):
    

    attack_success_array = []

    perturbation_model = UniversalPerturbation((1, 3, input_shape, input_shape), epsilon).to(device)
    optimizer = optim.SGD(perturbation_model.parameters(), lr=alpha)

    target_mlp = target_mlp_optimization(model, 
                                         layer_shape, 
                                         target_class, 
                                         target_loader, 
                                         test_loader, 
                                         device, 
                                         num_training_epochs=num_mlp_training_epochs, 
                                         print_info = print_info)

    target_mlp.to(device)
    target_mlp.eval()

    criterion = nn.CrossEntropyLoss()  

    test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
    print(f'Test Accuracy: {test_accuracy:.2f}% | {attack_accuracy:.2f}%')
    attack_success_array.append((test_accuracy, attack_accuracy))

    count = 0
    for epoch in range(num_attack_epochs):
        for images, _ in opt_loader:
            perturbation_model.train()  # Set to training mode during UAP crafting
            count += 1
            images = images.to(device)
            perturbed_images = perturbation_model(images)
            _ = model(perturbed_images)            
            activations = activation['act']  
            pooled_activation = activations    
            output = target_mlp(pooled_activation)
    
            # Compute the loss to maximize the probability of the target class index (which is 1 in the model)
            target_class_labels = torch.full((images.size(0),), 1, dtype=torch.long).to(device)
            loss = criterion(output, target_class_labels)  # Use CrossEntropyLoss to push towards target class
    
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            perturbation_model.update_grad_inf()
            optimizer.step()
            project_perturbation_inf(perturbation_model.perturbation, epsilon)

    
            if count % 10 == 0:
                perturbation_model.eval()
                test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
                if print_info: 
                    print(f'Epoch [{epoch+1}/{num_attack_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy:.2f} | [{attack_accuracy:.2f}] %')
                attack_success_array.append((test_accuracy, attack_accuracy))
                
    perturbation_model.eval()
    test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
    print(f'[END] Epoch [{epoch+1}/{num_attack_epochs}], Loss: {loss.item():.4f}, Test Accuracy: [{test_accuracy:.2f}] | [{attack_accuracy:.2f}] %')
    attack_success_array.append((test_accuracy, attack_accuracy))
    
    print("Universal perturbation crafted and evaluated successfully.")

    return perturbation_model, attack_success_array



def run_attacks_inf_multi_layer(model, 
                    layer_shape,
                    input_shape,
                    target_class, 
                    target_loader, 
                    opt_loader, 
                    test_loader, 
                    layer_name,
                    device,
                    num_mlp_training_epochs=50, 
                    num_attack_epochs =50, 
                    epsilon =16/255,
                    alpha = 2/255,
                    iter_stop = False, 
                    print_info = True, 
                    list_mlp_layers = None, 
                    attack_type = 'inf',
                    normalize_gradient = True, 
                    multiple_layers = True,
                    num_conv_blocks_mlp = 1):
    

    attack_success_array = []

    perturbation_model = UniversalPerturbation((1, 3, input_shape, input_shape), epsilon).to(device)
    optimizer = optim.SGD(perturbation_model.parameters(), lr=alpha)

    target_mlp = target_mlp_optimization(model, 
                                         layer_shape, 
                                         target_class, 
                                         target_loader, 
                                         test_loader,
                                         layer_name,
                                         device, 
                                         num_training_epochs=num_mlp_training_epochs, 
                                         print_info = print_info, 
                                         num_conv_blocks_mlp = num_conv_blocks_mlp
                                        )

    target_mlp.to(device)
    target_mlp.eval()

    #if len(list_mlp_layers) > 0:
    list_mlp_layers.append(target_mlp)
        

    criterion = nn.CrossEntropyLoss()  

    test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
    print(f'Test Accuracy: {test_accuracy:.2f}% | {attack_accuracy:.2f}%')
    attack_success_array.append((test_accuracy, attack_accuracy))

    count = 0
    for epoch in range(num_attack_epochs):
    
        for images, _ in opt_loader:
            perturbation_model.train()  # Set to training mode during UAP crafting
            count += 1
            images = images.to(device)
            perturbed_images = perturbation_model(images)
            _ = model(perturbed_images)   

            # optimize accross all the available hidden layers
            grad_value = None
            for i, key in enumerate(activation.keys()):
                activations_ = activation[key]  
                if multiple_layers is False and (i < len(activation.keys())-1):
                    continue
                
                pooled_activation = activations_    
                output = list_mlp_layers[i](pooled_activation)
    
                # Compute the loss to maximize the probability of the target class index (which is 1 in the model)
                target_class_labels = torch.full((images.size(0),), 1, dtype=torch.long).to(device)
                loss = criterion(output, target_class_labels)  # Use CrossEntropyLoss to push towards target class
    
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                if grad_value is None: 
                    if normalize_gradient: 
                        grad_value = perturbation_model.perturbation.grad.data / (perturbation_model.perturbation.grad.data.norm(p=2) + 1e-8)
                    else:    
                        grad_value = perturbation_model.perturbation.grad.data
                else:
                    if normalize_gradient: 
                        grad_value += perturbation_model.perturbation.grad.data / (perturbation_model.perturbation.grad.data.norm(p=2) + 1e-8)
                    else:
                        grad_value += perturbation_model.perturbation.grad.data


            perturbation_model.perturbation.grad.data = grad_value
            if attack_type == 'l2':
                #perturbation_model.update_grad_l2()
                None
            else:
                perturbation_model.update_grad_inf()

                
            optimizer.step()

            if attack_type == 'l2':
                project_perturbation_l2(perturbation_model.perturbation, epsilon)
                
            else:
                project_perturbation_inf(perturbation_model.perturbation, epsilon)
                
    perturbation_model.eval()  # Set to training mode during UAP crafting
    test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
    print(f'[END] Epoch [{epoch+1}/{num_attack_epochs}], Loss: {loss.item():.4f}, Test Accuracy: [{test_accuracy:.2f}] | [{attack_accuracy:.2f}] %')
    attack_success_array.append((test_accuracy, attack_accuracy))
    
    print("Universal perturbation crafted and evaluated successfully.")

    return perturbation_model, attack_success_array, list_mlp_layers


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------




def run_baseline_inf_multi_layer(model,
                    input_shape,
                    target_class, 
                    opt_loader, 
                    test_loader, 
                    device,
                    num_attack_epochs =50, 
                    epsilon =16/255,
                    alpha = 2/255,
                    iter_stop = False, 
                    print_info = True, 
                    attack_type = 'inf'):
    

    attack_success_array = []

    perturbation_model = UniversalPerturbation((1, 3, input_shape, input_shape), epsilon).to(device)
    optimizer = optim.SGD(perturbation_model.parameters(), lr=alpha)

    criterion = nn.CrossEntropyLoss()  

    test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
    print(f'Test Accuracy: {test_accuracy:.2f}% | {attack_accuracy:.2f}%')
    attack_success_array.append((test_accuracy, attack_accuracy))

    count = 0
    for epoch in range(num_attack_epochs):    
        for (images, labels) in opt_loader:
            perturbation_model.train()  # Set to training mode during UAP crafting
            count += 1
            images = images.to(device)
            labels = labels.to(device)
            perturbed_images = perturbation_model(images)
            output = model(perturbed_images)  

            if target_class >= 0:
                # Loss for targeted attacks: aim to push the model's output towards the target class
                target_labels = torch.full((images.size(0),), target_class, device=device, dtype=torch.long)
                loss = criterion(output, target_labels)
            else:
                # Loss for untargeted attacks: maximize misclassification by minimizing correct class scores
                loss = -criterion(output, labels) 

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if attack_type == 'l2':
                #perturbation_model.update_grad_l2()
                None
            else:
                perturbation_model.update_grad_inf()
            optimizer.step()

            if attack_type == 'l2':
                project_perturbation_l2(perturbation_model.perturbation, epsilon) 
            else:
                project_perturbation_inf(perturbation_model.perturbation, epsilon)

    perturbation_model.eval()  # Set to training mode during UAP crafting
    test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
    print(f'[END] Epoch [{epoch+1}/{num_attack_epochs}], Loss: {loss.item():.4f}, Test Accuracy: [{test_accuracy:.2f}] | [{attack_accuracy:.2f}] %')
    attack_success_array.append((test_accuracy, attack_accuracy))
    
    print("Baseline Universal perturbation crafted and evaluated successfully.")

    return perturbation_model, attack_success_array



#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------




def run_baseline_inf_multi_layer_cosine(model,
                    input_shape,
                    target_class, 
                    opt_loader, 
                    test_loader, 
                    device,
                    num_attack_epochs =50, 
                    epsilon =16/255,
                    alpha = 2/255,
                    iter_stop = False, 
                    print_info = True, 
                    attack_type = 'inf'):
    

    attack_success_array = []

    perturbation_model = UniversalPerturbation((1, 3, input_shape, input_shape), epsilon).to(device)
    optimizer = optim.SGD(perturbation_model.parameters(), lr=alpha)

    criterion = nn.CosineSimilarity(dim=1, eps=1e-08)  

    test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
    print(f'Test Accuracy: {test_accuracy:.2f}% | {attack_accuracy:.2f}%')
    attack_success_array.append((test_accuracy, attack_accuracy))

    count = 0
    for epoch in range(num_attack_epochs):    
        for (images, labels) in opt_loader:
            perturbation_model.train()  # Set to training mode during UAP crafting
            count += 1
            images = images.to(device)
            labels = labels.to(device)

            ref_output =  model(images).detach()
            perturbed_images = perturbation_model(images)
            output = model(perturbed_images)  

            loss = criterion(output, ref_output).mean()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if attack_type == 'l2':
                #perturbation_model.update_grad_l2()
                None
            else:
                perturbation_model.update_grad_inf()
            optimizer.step()

            if attack_type == 'l2':
                project_perturbation_l2(perturbation_model.perturbation, epsilon) 
            else:
                project_perturbation_inf(perturbation_model.perturbation, epsilon)

    perturbation_model.eval()  # Set to training mode during UAP crafting
    test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
    print(f'[END] Epoch [{epoch+1}/{num_attack_epochs}], Loss: {loss.item():.4f}, Test Accuracy: [{test_accuracy:.2f}] | [{attack_accuracy:.2f}] %')
    attack_success_array.append((test_accuracy, attack_accuracy))
    
    print("Baseline Universal perturbation crafted and evaluated successfully.")

    return perturbation_model, attack_success_array




def run_attacks_inf_multi_layer_cosine(model, 
                    layer_shape,
                    input_shape,
                    target_class, 
                    opt_loader, 
                    test_loader, 
                    layer_name,
                    device,
                    num_attack_epochs =50, 
                    epsilon =16/255,
                    alpha = 2/255,
                    iter_stop = False, 
                    print_info = True, 
                    attack_type = 'inf',
                    normalize_gradient = True, 
                    multiple_layers = True,):
    
    attack_success_array = []
    perturbation_model = UniversalPerturbation((1, 3, input_shape, input_shape), epsilon).to(device)
    optimizer = optim.SGD(perturbation_model.parameters(), lr=alpha)

   
    criterion = nn.CosineSimilarity(dim=1, eps=1e-08) 
    test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
    print(f'Test Accuracy: {test_accuracy:.2f}% | {attack_accuracy:.2f}%')
    attack_success_array.append((test_accuracy, attack_accuracy))

    count = 0
    for epoch in range(num_attack_epochs):
    
        for images, _ in opt_loader:

            images = images.to(device)
            _ = model(images) 
            activations_clean = []
            for i, key in enumerate(activation.keys()):
                activations_clean.append(activation[key].detach())
                if multiple_layers is False and (i < len(activation.keys())-1):
                    continue
            
            perturbation_model.train()  # Set to training mode during UAP crafting
            count += 1
            perturbed_images = perturbation_model(images)
            _ = model(perturbed_images)   

            # optimize accross all the available hidden layers
            grad_value = None
            for i, key in enumerate(activation.keys()):
                activations_ = activation[key]  
                if multiple_layers is False and (i < len(activation.keys())-1):
                    continue

                reference_act = activations_clean[i]
                loss = criterion(activations_, reference_act).mean()
    
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                if grad_value is None: 
                    if normalize_gradient: 
                        grad_value = perturbation_model.perturbation.grad.data / (perturbation_model.perturbation.grad.data.norm(p=2) + 1e-8)
                    else:    
                        grad_value = perturbation_model.perturbation.grad.data
                else:
                    if normalize_gradient: 
                        grad_value += perturbation_model.perturbation.grad.data / (perturbation_model.perturbation.grad.data.norm(p=2) + 1e-8)
                    else:
                        grad_value += perturbation_model.perturbation.grad.data


            perturbation_model.perturbation.grad.data = grad_value
            if attack_type == 'l2':
                #perturbation_model.update_grad_l2()
                None
            else:
                perturbation_model.update_grad_inf()

                
            optimizer.step()

            if attack_type == 'l2':
                project_perturbation_l2(perturbation_model.perturbation, epsilon)
                
            else:
                project_perturbation_inf(perturbation_model.perturbation, epsilon)
                
    perturbation_model.eval()  # Set to training mode during UAP crafting
    test_accuracy, attack_accuracy = evaluate_uap(test_loader, model, perturbation_model, target_class, device)
    print(f'[END] Epoch [{epoch+1}/{num_attack_epochs}], Loss: {loss.item():.4f}, Test Accuracy: [{test_accuracy:.2f}] | [{attack_accuracy:.2f}] %')
    attack_success_array.append((test_accuracy, attack_accuracy))
    
    print("Universal perturbation crafted and evaluated successfully.")

    return perturbation_model, attack_success_array