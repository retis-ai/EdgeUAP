import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import random
import argparse
sys.path.append(os.path.realpath('..'))
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import utils_uap

# Function to load configuration from YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Argument parser to specify YAML config file from command line
parser = argparse.ArgumentParser(description="Run model attacks with specified configuration.")
parser.add_argument("--config", type=str, default="config.yml", help="Path to the configuration YAML file.")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
SEED = config['seed']
DEVICE_NUM = config['device_num']
torch.cuda.set_device(DEVICE_NUM)
device = torch.device(f"cuda:{DEVICE_NUM}")

# Set random seeds for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Model and attack parameters from config
MODEL_NAMES = config['model_names']
batch_size = config['batch_size']
batch_size_test = config['batch_size_test']
target_class = config['target_class']
epsilon = config['epsilon']
alpha = config['alpha']
samples_mlp_target = config['samples_mlp_target']
samples_mlp_others = config['samples_mlp_others']
samples_attack_dataset = config['samples_attack_dataset']
num_max_mlp_training_epochs = config['num_max_mlp_training_epochs']
num_max_attack_epochs = config['num_max_attack_epochs']
attack_type = config['attack_type']
data_split_ratios = config['data_split_ratios']
use_only_first_layer = config.get('use_only_first_layer', False)

baseline_optimization = config.get('baseline_optimization', False)
normalize_gradient = config.get('normalize_gradient', True)
multiple_layers = config.get('multiple_layers', True)
limit_layers_analysis = config.get('limit_layers_analysis', -1)



output_dir = config['output_dir']

# Load datasets and transform
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
_, val_dataset = utils_uap.load_imagenet_data()

# Split dataset based on configuration
uap_opt_dataset, test_dataset = utils_uap.split_dataset(val_dataset, data_split_ratios)
if target_class >= 0 and baseline_optimization is False:
    class_specific_samples, noclass_specific_samples = utils_uap.split_dataset_by_class(
        uap_opt_dataset, target_class,
        samples_no_classes=samples_mlp_others,
        samples_classes=samples_mlp_target
    )
    mlp_dataset = utils_uap.concat_subsets(noclass_specific_samples, class_specific_samples)
    print("MLP DATASET LEN: " + str(len(mlp_dataset)))
    
print("TEST DATASET LEN: " + str(len(test_dataset)))


# Creating loaders for MLP dataset and test dataset
if target_class >= 0 and baseline_optimization is False:
    labels = [0] * samples_mlp_others + [1] * samples_mlp_target
    mlp_loader = utils_uap.create_balanced_dataloader(mlp_dataset, labels, batch_size)


test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=2)

# Further split for UAP optimization
uap_opt_dataset, _ = torch.utils.data.random_split(
    uap_opt_dataset, [samples_attack_dataset, len(uap_opt_dataset) - samples_attack_dataset]
)
print("UAP OPT DATASET LEN: " + str(len(uap_opt_dataset)))
uap_opt_loader = DataLoader(uap_opt_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Set random experiment seed for unique result files
random_exp_seed = random.randint(1, 99999)

# Ensure results directory exists
os.makedirs("./results", exist_ok=True)

# Main loop to iterate through each model specified in MODEL_NAMES
for NAME in MODEL_NAMES:
    model, layers_info = utils_uap.get_model_and_info(NAME)
    model = model.to(device)
    
    if use_only_first_layer: 
        layers_info = [layers_info[0]]
        
    print("Starting analysis of " + str(NAME))

    # Initialize attack info object
    attack_info = utils_uap.AttackInfo()
    attack_info.update_attack_info(model_name=NAME, target_class=target_class, epsilon=epsilon,
                                   samples_mlp_target=samples_mlp_target, 
                                   samples_mlp_others=samples_mlp_others, 
                                   samples_attack_dataset=samples_attack_dataset,
                                   attack_type=attack_type)

    # Iterate through each layer in the model for analysis
    list_mlp_layers = []
    list_handlers = []
    for info in layers_info:
        
        # Run the attack for this specific layer
        if baseline_optimization is True:
            layer_name = 'all'
            uap, attack_results = utils_uap.run_baseline_inf_multi_layer(
                model,
                input_shape=224,
                target_class=target_class, 
                opt_loader=uap_opt_loader, 
                test_loader=test_loader, 
                device=device,
                num_attack_epochs=num_max_attack_epochs, 
                epsilon=epsilon, 
                alpha=alpha,
                print_info=True,
                attack_type=attack_type)
            
        else:
            
            model_layer, layer_channels, layer_name = info
            print("-->Analysis of layer: " + str(layer_name))
            # Register a forward hook to capture activations
            handler = model_layer.register_forward_hook(utils_uap.get_activation(layer_name))
            list_handlers.append(handler)
            uap, attack_results, list_mlp_layers = utils_uap.run_attacks_inf_multi_layer(
                model,
                layer_shape=layer_channels,
                input_shape=224,
                target_class=target_class, 
                target_loader=mlp_loader, 
                opt_loader=uap_opt_loader, 
                test_loader=test_loader, 
                layer_name = layer_name,
                device=device,
                num_mlp_training_epochs=num_max_mlp_training_epochs, 
                num_attack_epochs=num_max_attack_epochs, 
                epsilon=epsilon, 
                alpha=alpha,
                print_info=True,
                list_mlp_layers = list_mlp_layers, 
                attack_type=attack_type, 
                normalize_gradient = normalize_gradient, 
                multiple_layers = multiple_layers
            )


        
        # Store the UAP on CPU and remove the hook
        uap = uap.to('cpu')
        
        #handle.remove()
        
        # Update attack information
        attack_info.update_attack_info(depth=layer_name, res=attack_results, uap_models=uap)

        if limit_layers_analysis > 0:
            if len(list_mlp_layers) == limit_layers_analysis: 
                break
        if baseline_optimization is True:
            break

    # Save attack information for the model as a pickle file
    attack_info.save_as_pickle(f"{output_dir}/{NAME}_last_{random_exp_seed}.pkl")
    print(attack_info)
    print("\n--------------------------------------------------\n")

    # Clean up for the next model
    for handler in list_handlers:
        handler.remove()
    del model 
    torch.cuda.empty_cache()
    utils_uap.reset_activation()
