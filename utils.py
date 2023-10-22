from typing import List

import numpy as np
import torch
from skimage.color import lab2rgb
from torchvision import transforms
import config
from torchvision.utils import save_image


def connect_x_y(l, ab):
    x = np.zeros(256 * 256 * 3)
    l = l.cpu()
    ab = ab.cpu()
    x.resize([256, 256, 3])
    x[:, :, 0] = l
    x[:, :, 1] = ab[0, :, :]
    x[:, :, 2] = ab[1, :, :]
    x = lab2rgb(x)
    x = transforms.ToTensor()(x)
    return x


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = connect_x_y(x, y_fake[0, :, :, :])
        save_image(y_fake, folder + f"/gen_09052023_{epoch}.png")

        if epoch == 0:
            real = connect_x_y(x, y[0, :, :, :])
            save_image(real, folder + f"/input_09052023_{epoch}.png")

    gen.train()


def save_checkpoint(model, optimaizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimaizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=>loading checkpoint ")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

import numpy as np
def rotate(self, matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    matrix=np.ones_like(matrix).dot(matrix)