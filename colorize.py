import torch
from skimage.color import rgb2lab

from utils import load_checkpoint, connect_x_y
import torch.optim as optim
import config
from generator import Generator
from torchvision import transforms
import numpy as np
import cv2

Transforms = transforms.Compose([
    transforms.Resize((256, 256))
])


def getImageChannels(image):
    x = image[0, :, :]
    y = image[[1, 2], :, :]
    return x, y


# def rescale(image):
#     img = cv2.imread("sean.jpg")
#     #We will be factoring down images using the already scaled.
#     lwr1 = cv2.pyrDown(img)
#     lwr2 = cv2.pyrDown(lwr1)
#     lwr3 = cv2.pyrDown(lwr2)
#     lwr4 = cv2.pyrDown(lwr3)
#     # We will be Increasing the resolution of already scaled down image that is lwr4.
#     hir1 = cv2.pyrUp(lwr3)
#     cv2.imshow("Original image", img)
#     cv2.imshow("First Scaled Down Image", lwr1)
#     cv2.imshow("Second Scaled Down Image", lwr2)
#     cv2.imshow("Third Scaled Down Image", lwr3)
#     cv2.imshow("Fourth Scaled Down Image", lwr4)
#     cv2.imshow("First Scaled Up Image", hir1)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def pix2pix(image):
    # transform = transforms.Compose([transforms.Resize((256, 256))])
    # img = transform(image)
    # img = np.array(img)
    # img_lab = transforms.ToTensor()(img)
    # l = img_lab[0, :, :]
    # l = torch.reshape(l, [1, 256, 256])
    # l = l.to(config.DEVICE)
    img = image.convert("RGB")
    img = Transforms(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)
    x, y = getImageChannels(img_lab)
    x = torch.reshape(x, [1, 1, 256, 256])
    # x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen = Generator(in_channels=1).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(r"C:/Users/ASUS/PycharmProjects/ImageColorization/checkPoints/09052023_104/09052023_gen.pth.tar", gen,
                    opt_gen, config.LEARNING_RATE)
    # gen.eval()
    # with torch.no_grad():
    #     y_fake = gen(x)
    #     y_fake = connect_x_y(x, y_fake[0, :, :, :])
    #     save_image(y_fake, f"results/1.png")
    # return y_fake
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = connect_x_y(x, y_fake[0, :, :, :])
    gen.train()
    return y_fake
