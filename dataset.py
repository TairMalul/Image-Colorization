import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from skimage.color import rgb2lab


class MyDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[0, :, :]
        ab = img_lab[[1, 2], :, :]
        L = torch.reshape(L, [1, 256, 256])
        return L, ab
