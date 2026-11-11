#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import datasets  
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split


# In[63]:


ROOT = "data"
RANDOM_STATE = 42


# # Data Pre Processing

# In[68]:


train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ViT expects 224x224, normalized with mean of 0.5 and std of 0.5


# In[69]:


class PetDataset(Dataset):
    def __init__(self, df, transform):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.data[i]
        transformed_image = self.transform(image['image'].convert("RGB"))
        return transformed_image, image['label'], image['image_id'], image['label_cat_dog']


# In[ ]:


def get_data(batch_size=64,random_state=RANDOM_STATE):
    hf = load_dataset("timm/oxford-iiit-pet", cache_dir=ROOT)
    train_data = hf['train']
    test_data = hf['test']
    
    labels = train_data['label']

    train_indices, val_indices = train_test_split(range(len(labels)),test_size=735, stratify=labels, random_state=random_state)

    train_ds = PetDataset(df=train_data.select(train_indices), transform=train_tf)
    val_ds = PetDataset(df=train_data.select(val_indices), transform=eval_tf)
    test_ds = PetDataset(df=test_data, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

def get_breads_names():
    hf = load_dataset("timm/oxford-iiit-pet", cache_dir=ROOT)
    return hf['train'].features["label"].names