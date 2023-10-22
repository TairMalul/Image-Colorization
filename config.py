import torch.cuda

DEVICE ="cuda:0" if torch.cuda.is_available() else "cpu"
# LEARNING_RATE = 0.0002
LEARNING_RATE = 2e-4
BATCH_SIZE = 12
IMAGE_SIZE = 256
CHANNELS_IMAGE = 3
L1_LAMBDA = 100
NUM_EPOCHS = 4
LOAD_MODEL = True
SAVE_MODEL = True
# CHECKPOINT_DISC = r"C:\Users\ASUS\PycharmProjects\ImageColorization\09052023_104\09052023_disc.pth.tar"
# CHECKPOINT_GEN = r"C:\Users\ASUS\PycharmProjects\ImageColorization\checkPoints\09052023_104\09052023_gen.pth.tar"
CHECKPOINT_DISC = r"C:\Users\ASUS\PycharmProjects\ImageColorization\checkPoints\22062023\22062023_disc.pth.tar"
CHECKPOINT_GEN = r"C:\Users\ASUS\PycharmProjects\ImageColorization\checkPoints\22062023\22062023_gen.pth.tar"
NUM_WORKERS = 3

