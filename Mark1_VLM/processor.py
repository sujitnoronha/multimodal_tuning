from typing import Optional, Tuple, List, Dict, Any, Union
from PIL import Image
import torch
from transformers import ProcessorMixin, AutoTokenizer
import json
from transformers import CLIPImageProcessor
import os
import numpy as np


class MarkVLMProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.current_processor = self

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the processor to a directory.
        Args:
            save_directory (str): Directory to save the processor.
            **kwargs: Additional keyword arguments.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory)
        self.image_processor.save_pretrained(save_directory)

        config = {
            "tokenizer_name": "meta-llama/Llama-3.2-1B-Instruct",
            "image_processor_name": "openai/clip-vit-large-patch14"
        }

        config_path = os.path.join(save_directory, "processor_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)


    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """ Create config from a dictionary. """
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)



    def __call__(self, 
                text: Optional[Union[str, List[str]]] = None,
                images: Optional[Union[Image.Image, List[Image.Image]]] = None,
                return_tensors: Optional[str] = 'pt',
                **kwargs
                 ) -> Dict[str, torch.Tensor]:
        """
        Process images and text 

        Args:
            text (str or list of str): Text to be processed.
            images (PIL Image or list of PIL Images): Images to be processed.
            return_tensors (str): The type of tensors to return. Default is 'pt' for PyTorch tensors.
            **kwargs: Additional keyword arguments.
        
        Raises:
            ValueError: If both images and text are provided but their lengths don't match
        """
        
        if images is None and text is None:
            raise ValueError("Please provide either images or text to process.")
        
        encoding  = {}
        special_image_token = None
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
                special_image_symbol = "※"
                special_image_token = self.tokenizer(special_image_symbol, 
                                                    return_tensors=return_tensors,
                                                    add_special_tokens=False).input_ids
                
                #logging.info(f"Special image token: {special_image_token}")

                
                pixel_values = self.image_processor(images, return_tensors=return_tensors)['pixel_values']
                encoding["pixel_values"] = pixel_values
            
        if text is not None:
            if isinstance(text, list) and any(isinstance(i, dict) for i in text):
                text = self.tokenizer.decode(self.tokenizer.apply_chat_template(text, add_generation_prompt=True))

            if isinstance(text, list) and any(isinstance(item, list) for item in text):
                text = [self.tokenizer.decode(self.tokenizer.apply_chat_template(item, add_generation_prompt=True)) for item in text]
            

            if isinstance(text, str):
                text = [text]
            
            if images is not None and len(images) != len(text):
                raise ValueError(
                        f"Number of images ({len(images)}) must match number of text entries ({len(text)}) "
                        "when both are provided."
                    )
        
            text_encoding = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding="max_length",
                max_length=249,
                truncation=True,
                **kwargs
            )

            if special_image_token:
                
                special_tokens_column = torch.full(
                    (text_encoding["input_ids"].size(0), 1),
                    float(special_image_token[0]),
                    dtype=text_encoding["input_ids"].dtype,
                    device=text_encoding["input_ids"].device
                )

                # example
                # special_tokens_column would now be:
                # tensor([[49152],
                #         [49152]])


                attention_mask = text_encoding[
                    "attention_mask"
                ]

                encoding["input_ids"] = torch.cat([
                    special_tokens_column,
                    text_encoding["input_ids"]], dim=1
                )

                attention_mask_special = torch.zeros(attention_mask.size()[0],
                                                     attention_mask.size()[1] + 1)

                seq_lengths = attention_mask.sum(dim=1)

                for i in range(len(seq_lengths)):
                    attention_mask_special[i, :seq_lengths[i] + 1] = 1

                encoding["attention_mask"] = attention_mask_special
                 
            else:
                encoding["input_ids"] = text_encoding["input_ids"]
                encoding["attention_mask"] = text_encoding["attention_mask"]
            
        return encoding
    
    def batch_decode(self, *args, **kwargs):
        """
        Decode a batch of input IDs to text.
        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        Returns:
            Decoded text.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        Decode input IDs to text.
        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        Returns:
            Decoded text.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Returns the model input names.
        Returns:
            List of model input names.
        """
        return ["input_ids", "attention_mask", "pixel_values"]            
    

    def get_chat_template(self, messages: Optional[List[List[str]]]):
        """
        Apply chat template to the messages.
        """
        return self.tokenizer.apply_chat_template(messages)