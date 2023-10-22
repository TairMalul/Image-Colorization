import torch
from utils import load_checkpoint, save_some_examples, save_checkpoint
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MyDataset
from generator import Generator
from discrimnator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(disc, gen, loader, opt_disc, opt_gen, l1, BCE, g_scalar, d_scalar):
    loop = tqdm(loader, leave=True)
    for index, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            d_real = disc(x, y)
            d_fake = disc(x, y_fake.detach())
            d_real_loss = BCE(d_real, torch.ones_like(d_real))
            d_fake_loss = BCE(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2

        disc.zero_grad()
        d_scalar.scale(d_loss).backward()
        d_scalar.step(opt_disc)
        d_scalar.update()

        with torch.cuda.amp.autocast():
            d_fake = disc(x, y_fake)
            g_fake_loss = BCE(d_fake, torch.ones_like(d_fake))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            g_loss = g_fake_loss + L1

        opt_gen.zero_grad()
        g_scalar.scale(g_loss).backward()
        g_scalar.step(opt_gen)
        g_scalar.update()


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=1).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1 = nn.L1Loss()
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = MyDataset(root_dir=r"train2017A")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS)
    # val_dataset = MyDataset(root_dir=r"val2017")
    val_dataset = MyDataset(root_dir=r"abc")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)
    g_scalar = torch.cuda.amp.GradScaler()
    d_scalar = torch.cuda.amp.GradScaler()

    # for epoch in range(config.NUM_EPOCHS):
    #     train(disc, gen, train_loader, opt_disc, opt_gen, L1, BCE, g_scalar, d_scalar)
    #     if config.SAVE_MODEL and epoch % 1 == 0:
    #         save_checkpoint(disc, opt_disc, config.CHECKPOINT_DISC)
    #         save_checkpoint(gen, opt_gen, config.CHECKPOINT_GEN)
    #     save_some_examples(gen, val_loader, epoch+112,
    #    folder=r"C:\Users\ASUS\PycharmProjects\ImageColorization\netProgress")


    lopp =tqdm(range(1), leave=True)
    for x in lopp:
        save_some_examples(gen, val_loader, x,folder=r"C:\Users\ASUS\PycharmProjects\ImageColorization\try")

if __name__ == '__main__':
    main()
