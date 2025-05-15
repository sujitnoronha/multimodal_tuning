import torch
from torch.utils.data import Dataset, DataLoader
from transformers.image_utils import load_image
import os 
import numpy as np
import json
from PIL import Image
import logging 
from datasets import load_dataset
from model import MarkVLMConfig, MarkVLM
from Mark1_VLM.processor import MarkVLMProcessor
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_dir_empty(folder_path):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print("Folder doesn't exist")
        return False
    
    # List contents of the folder
    contents = os.listdir(folder_path)
    
    if len(contents) == 0:
        print("Folder is empty")
        return True
    
    print("Files found in folder:")
    for item in contents:
        print(f"- {item}")
    return False


def load_adapter_weights(model, checkpoint_path):
    """
    Load adapter weights from a checkpoint file.

    Args:
        model: The Model containing the adapter
        checkpoint_path (str): Path to the checkpoint file
    """

    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        model.adapter.load_state_dict(checkpoint['adapter_state_dict'])

        logger.info(f"Successfully loaded checkpoints form step {checkpoint['step']}")
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        # Verify adapter parameters are loaded
        trainable_params = sum(p.numel() for p in model.adapter.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Adapter parameters loaded: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        
        return model
    

    except Exception as e:
        logger.error(f"Error loading checkpoints: {str(e)}")
        return False
    

class VLMDataset(Dataset):
    def __init__self(self, data_dir, processor):
        self.processor = processor
        with open(data_dir, 'r') as f:
            self.data = json.load(f)[:40000]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        
        return image_path


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )



def main(): 
    output_dir = "model_training_checkpoints"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    config = MarkVLMConfig()
    model = MarkVLM(config)

    print_trainable_parameters(model)
    
    for name, param in model.named_parameters():
        logger.info(f'{name}: {param.shape}')

    

    



if __name__ == "__main__":
    main()

