import torch
import pytorch_lightning as pl
import os
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch.nn.functional as F
batch_size = 256
random_seed = 42
torch.manual_seed(random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_workers = int(os.cpu_count()/2)

class data(pl.LightningDataModule):
    def __init__(self, data = "./data", batch_size = batch_size, num_workers = num_workers):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    def prepare_data(self):
        MNIST(self.data, train=True, download = True)
        MNIST(self.data, train=False, download = True)
    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            mmnist = MNIST(self.data, train=True, transform = self.transform)
            self.mnist_train, self.mnist_val = random_split(mmnist, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data, train = False, transform = self.transform)
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size = self.batch_size, num_workers = self.num_workers, persistent_workers = True)
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size = self.batch_size, num_workers = self.num_workers, persistent_workers = True)
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)




class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 7 * 7 * 64)
        self.bn2 = nn.BatchNorm1d(7*7*64)
        self.ct1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.ct2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv = nn.Conv2d(16, 1, kernel_size=7, padding=3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.linear1(x)))
        x = F.leaky_relu(self.bn2(self.linear2(x)))
        x = x.view(-1, 64, 7, 7)
        x = F.leaky_relu(self.bn3(self.ct1(x)))
        x = F.leaky_relu(self.bn4(self.ct2(x)))
        x = torch.tanh(self.conv(x))
        return x


class GAN(pl.LightningModule):
    def __init__(self, latent_dim = 100, lr = 0.0002):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim = self.hparams.latent_dim)
        self.discriminator = Discriminator()
        self.automatic_optimization = False
    def forward(self, noise):
        return self.generator(noise)
    def loss_function(self, output, target):
        return F.binary_cross_entropy(output, target)
    def training_step(self, batch, batch_idx):
        images, _ = batch
        noise = torch.randn(images.size(0), self.hparams.latent_dim).type_as(images)
        gopt, dopt = self.optimizers()
        gopt.zero_grad()
        fake_images = self(noise)
        y_pred_fake = self.discriminator(fake_images)
        g_loss = self.loss_function(y_pred_fake, torch.ones_like(y_pred_fake))
        self.manual_backward(g_loss)
        gopt.step()

        dopt.zero_grad()
        y_pred_real = self.discriminator(images)
        y_pred_fake = self.discriminator(fake_images.detach())

        real_loss = self.loss_function(y_pred_real, torch.ones_like(y_pred_real))
        fake_loss = (self.loss_function(y_pred_fake, torch.zeros_like(y_pred_fake)))

        d_loss = (real_loss + fake_loss)/2
        self.manual_backward(d_loss)
        dopt.step()
        self.log_dict({"d_loss" : d_loss, "g_loss": g_loss})
    def validation_step(self, batch, batch_idx):
        images, _ = batch
        noise = torch.randn(images.size(0), self.hparams.latent_dim).type_as(images)
        fake_images = self(noise)
        y_pred_fake = self.discriminator(fake_images)
        g_loss = F.binary_cross_entropy(y_pred_fake, torch.ones_like(y_pred_fake))

        y_pred_real = self.discriminator(images)
        real_loss = F.binary_cross_entropy(y_pred_real, torch.ones_like(y_pred_real))

        y_pred_fake = self.discriminator(fake_images.detach())
        fake_loss = F.binary_cross_entropy(y_pred_fake, torch.zeros_like(y_pred_fake))

        d_loss = (real_loss + fake_loss) / 2
        self.log_dict({"val_d_loss": d_loss, "val_g_loss": g_loss}, prog_bar=True)
    def configure_optimizers(self):
        genoptim = torch.optim.Adam(self.generator.parameters(), lr = self.hparams.lr, betas = (0.5, 0.999))
        discoptim = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [genoptim, discoptim]

    def plot(self):
        size = 6
        noise = torch.randn(size, self.hparams.latent_dim).type_as(self.generator.linear1.weight)
        imgs = self(noise).cpu()
        fig = plt.figure()
        for i in range(imgs.size(0)):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            imgs_plot = imgs.detach().cpu()[i, 0, :, :]
            plt.imshow(imgs_plot, cmap='gray_r', interpolation='none')
            plt.title("generated")
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        plt.show()

    def on_train_epoch_end(self):
        self.plot()

if __name__ == "__main__":
    dm = data()
    model = GAN()
    model.plot()

    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, dm)
