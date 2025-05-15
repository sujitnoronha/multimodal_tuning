from typing import Optional, Union, Dict, List
from PIL import Image
import torch 
from torch.nn.modules import padding
from transformers import PretrainedConfig, ProcessorMixin, AutoTokenizer
from torchvision import transforms
import json 
from transformers import CLIPImageProcessor
import os
from dotenv import load_dotenv
from transformers import PreTrainedModel, AutoModelForCausalLM, CLIPVisionModel
load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_layer_dims = 500

class MarkVLMAdapter(torch.nn.Module):
    def __init__(self, lang_embed_dim, clip_dim, hidden_layer_dims=500):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.layer1 = torch.nn.Linear(clip_dim, 500)
        self.layer2 = torch.nn.Linear(500, 500)
        self.layer3 = torch.nn.Linear(500, lang_embed_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        output = self.activation(x)
        return output



class MarkVLMConfig(PretrainedConfig):
    model = 'MarkVLM'

    def __init__(self, 
                 lang_embed_dim=2048,
                 clip_dim=1024,
                 hidden_size=500,
                 vocab_size=32000,):
        self.lang_embed_dim = lang_embed_dim
        self.clip_dim = clip_dim
        self.hidden_size = hidden_size,
        self.vocab_size = vocab_size

        self.return_dict = True
        self.torchscript = False
        self.use_cache = True
        self.is_encoder_decoder = False

        self.hidden_size = 768
        self.num_attention_heads = 12
        self.num_hidden_layers = 12

    

    def to_dict(self):
        """ Convert config to dict format"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """ create config from a dictionary"""

        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)

        return config
    

class MarkVLM(PreTrainedModel):
    config_class = MarkVLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.vit = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        self.adapter = MarkVLMAdapter(config.lang_embed_dim, config.clip_dim, config.hidden_size).to(device)
        self.language_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",device_map='auto')

    def __extend_attention_mask(self, atten_mask, atten_to_img=True, num_added_tokens=257):
        batch_size, original_seq_length = atten_mask.shape

        if atten_to_img:
            extended_mask = torch.ones(
                batch_size,
                original_seq_length + num_added_tokens,
                dtype=atten_mask.dtype,
                device=atten_mask.device
            )
        else:
            extended_mask = torch.zeros(
                batch_size,
                original_seq_length + num_added_tokens,
                dtype=atten_mask.dtype,
                device=atten_mask.device
            )
        

        extended_mask[:, -original_seq_length:] = atten_mask

        return extended_mask
    

    def process_inputs(self, input_ids, attention_mask, pixel_values, attend_to_img_tokens=True):
        if input_ids is not None:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            final_embeddings = self.language_model.model.embed_tokens(input_ids).to(device)

            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
                vision_outputs = self.vit(pixel_values)

                image_embeddings = vision_outputs.last_hidden_state

                adapted_image_embeddings = self.adapter(image_embeddings).to(device)

                final_embeddings = torch.concat([adapted_image_embeddings, final_embeddings], axis=1).to(device)
                attention_mask = self.__extend_attention_mask(attention_mask, atten_to_img=attend_to_img_tokens).to(device)

            return final_embeddings, attention_mask


    def forward(self, inputs_ids=None, attention_mask=None, pixel_values=None, attend_to_img_tokens=True, labels=None, **kwargs):
        input_ids = kwargs.get('input_ids', None) or inputs_ids
        attention_mask = kwargs.get('attention_mask', None) or attention_mask
        pixel_values = kwargs.get('pixel_values', None) or pixel_values
        labels = kwargs.get('labels', None) or labels

        final_embeddings, attention_mask = self.process_inputs(input_ids, attention_mask, pixel_values, attend_to_img_tokens)

        if labels is not None:
            pred = self.language_model(input_embeds=final_embeddings, attention_mask=attention_mask, labels=labels)
        else:
            pred = self.language_model(input_embeds=final_embeddings, attention_mask=attention_mask)
        
        return pred

    def generate(self, input_ids=None, attention_mask=None, pixel_values=None, attend_to_img_tokens=True, max_new_tokens=50, temperature=0.3, top_p=0.9, **kwargs):
        input_ids = kwargs.pop('input_ids', None) or input_ids
        attention_mask = kwargs.pop('attention_mask', None) or attention_mask
        pixel_values = kwargs.pop('pixel_values', None) or pixel_values

        final_embeddings, attention_mask = self.process_inputs(input_ids, attention_mask,pixel_values, attend_to_img_tokens)
        return self.language_model.generate(input_embeds=final_embeddings, attention_mask=attention_mask, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    
