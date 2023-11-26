import math
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import tqdm
import tqdm as tqdm_module
import os
from scipy.stats import entropy
import numpy as np
import torchvision.utils as vutils
from torchvision.models import inception_v3
!pip install scipy
from scipy.linalg import sqrtm

torch.cuda.empty_cache()







transform = transforms.Compose([
    transforms.ToTensor(),])

dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
BATCH_SIZE = 64
IMAGE_SIZE = 32  # A CIFAR-10 képek mérete mindkét dimenzióban 32 pixel
CHANNELS = 3     # RGB színkód miatt 3
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


#kép mentéshez
def save_generated_images(images, folder_path, epoch):
    os.makedirs(folder_path, exist_ok=True)
    for i, image in enumerate(images):
        image_path = os.path.join(folder_path, f'generated_image_{epoch}_{i}.png')
        try:
            save_image(image, image_path)
        except Exception as e:
            print(f"Hiba a kép mentésekor {i}. számú képnél az epoch {epoch}. alkalommal: {e}")


#fid számoláshoz
cifar10_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
reference_images = torch.stack([cifar10_dataset[i][0] for i in range(1000)])  # Példaként az első 1000 képet használjuk
inception_model = inception_v3(pretrained=True, transform_input=False).to('cuda').eval()

def get_inception_features(images, model):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    images = torch.stack([transform(img) for img in images])
    features = model(images)
    return features.cpu().detach().numpy()



def calculate_fid(real_images, generated_images):
    # Képek helyes alakja
    if len(real_images.shape) != 4 or len(generated_images.shape) != 4:
        raise ValueError("A bemeneti képeknek (batch, channels, height, width) alakúnak kell lenniük")

    # Másolja a tenzorokat a host memóriába
    real_images = np.transpose(real_images.cpu().numpy(), (0, 2, 3, 1))
    generated_images = np.transpose(generated_images.cpu().numpy(), (0, 2, 3, 1))

    # Átlag és kovariancia valós
    mu_real = np.mean(real_images, axis=(0, 1, 2))
    cov_real = np.cov(real_images.reshape((-1, real_images.shape[-1])), rowvar=False)

    # Átlag és kovariancia generált
    mu_generated = np.mean(generated_images, axis=(0, 1, 2))
    cov_generated = np.cov(generated_images.reshape((-1, generated_images.shape[-1])), rowvar=False)

    # FID
    diff = mu_real - mu_generated
    cov_sqrt = sqrtm(cov_real @ cov_generated)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = np.sum(diff**2) + np.trace(cov_real + cov_generated - 2 * cov_sqrt)

    return fid



class TimeEmbedding(nn.Module):
    def __init__(self, feature_map_size):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.lin1 = nn.Linear(self.feature_map_size // 4, self.feature_map_size)
        self.lin2 = nn.Linear(self.feature_map_size, self.feature_map_size)

    def swish_activation(self, x):
        return x * torch.sigmoid(x)

    def forward(self, t):
        half_dim = self.feature_map_size // 8
        embedding = math.log(10_000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=t.device) * -embedding)
        embedding = t[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=1)
        embedding = self.swish_activation(self.lin1(embedding))
        embedding = self.lin2(embedding)
        return embedding

class Attention(nn.Module):

    def __init__(
        self, 
        feature_map_size, 
        groups_number,
        heads_number,
      ):
        super().__init__()
        self.head_size = feature_map_size
        self.heads_number = heads_number
        
        self.norm = nn.GroupNorm(groups_number, feature_map_size)
        self.projection = nn.Linear(feature_map_size, heads_number * self.head_size * 3)
        self.output = nn.Linear(heads_number * self.head_size, feature_map_size)
        self.scale = self.head_size ** -0.5
      
    def forward(
        self, 
        x, 
        t=None
    ):

        batch_size, feature_map_size, height, width = x.shape

        x = x.view(batch_size, feature_map_size, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.heads_number, 3 * self.head_size)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        attention = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attention = attention.softmax(dim=2)

        res = torch.einsum('bijh,bjhd->bihd', attention, v)
        res = res.view(batch_size, -1, self.heads_number * self.head_size)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, feature_map_size, height, width)
        return res

class Residual(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_channels,
        groups_number, 
        dropout_rate=0.1
    ):
        super().__init__()

        self.normalisation_1 = nn.GroupNorm(groups_number, in_channels)
        self.convolution_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.normalisation_2 = nn.GroupNorm(groups_number, out_channels)
        self.convolution_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.skip = nn.Identity()

        self.time_embedding = nn.Linear(time_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
    def swish_activation(
        self,
        x
    ):
        return x * torch.sigmoid(x)
    

    def forward(
        self, 
        x, 
        t
    ):

        y = self.convolution_1(self.swish_activation(self.normalisation_1(x)))
        y += self.time_embedding(self.swish_activation(t))[:, :, None, None]
        y = self.convolution_2(self.dropout(self.swish_activation(self.normalisation_2(y))))
        return y + self.skip(x)

class DownBlock(nn.Module):
    
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_channels, 
        has_attention,
        groups_number,
        heads_number
    ):  
        super().__init__()
        self.res = Residual(
            in_channels, 
            out_channels, 
            time_channels, 
            groups_number
        )
        
        if has_attention:
            self.attention = Attention(
                out_channels, 
                groups_number, 
                heads_number
            )
        else:
            self.attention = nn.Identity()

    def forward(
        self, 
        x, 
        t
    ):
        x = self.res(x, t)
        x = self.attention(x)
        return x


class UpBlock(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_channels, 
        has_attention,
        groups_number,
        heads_number
    ):
        super().__init__()
        self.res = Residual(
            in_channels + out_channels, 
            out_channels, 
            time_channels, 
            groups_number
        )
        
        if has_attention:
            self.attention = Attention(
                out_channels, 
                groups_number, 
                heads_number
            )
        else:
            self.attention = nn.Identity()

    def forward(
        self, 
        x, 
        t
    ):
        x = self.res(x, t)
        x = self.attention(x)
        return x


class Downsample(nn.Module):
    
    def __init__(
        self, 
        feature_map_size
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            feature_map_size, 
            feature_map_size, 
            (3, 3), 
            (2, 2), 
            (1, 1)
        )

    def forward(
        self, 
        x, 
        t
    ):

        return self.conv(x)


class Upsample(nn.Module):

    def __init__(
        self, 
        feature_map_size
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            feature_map_size, 
            feature_map_size, 
            (4, 4), 
            (2, 2), 
            (1, 1)
        )

    def forward(
        self, 
        x, 
        t
    ):

        return self.conv(x)
    

class Bottleneck(nn.Module):

    def __init__(
        self, 
        feature_map_size, 
        time_channels,
        groups_number,
        heads_number
    ):
        super().__init__()
        
        self.res1 = Residual(feature_map_size, feature_map_size, time_channels, groups_number)
        self.attn = Attention(feature_map_size, groups_number, heads_number)
        self.res2 = Residual(feature_map_size, feature_map_size, time_channels, groups_number)

    def forward(
        self, 
        x, 
        t
    ):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class EpsilonTheta(nn.Module):
    def __init__(self, channels, feature_map_size, groups_number, heads_number, blocks_number):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.groups_number = groups_number
        self.heads_number = heads_number
        self.has_attention = [False, False, False, True]

        self.image_proj = nn.Conv2d(channels, feature_map_size, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb = TimeEmbedding(feature_map_size * 4)  # Fix here

        multipliers = [1, 2, 2, 4]
        n_resolutions = len(multipliers)

        # DOWNSAMPLING (ENCODER)
        down = []
        out_channels = in_channels = feature_map_size
        for i in range(n_resolutions):
            out_channels = in_channels * multipliers[i]
            for _ in range(blocks_number):
                down.append(DownBlock(
                    in_channels,
                    out_channels,
                    feature_map_size * 4,
                    has_attention=self.has_attention[i],
                    groups_number=self.groups_number,
                    heads_number=self.heads_number)
                )
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        # BOTTLENECK
        self.bottleneck = Bottleneck(
            out_channels,
            feature_map_size * 4,
            groups_number=self.groups_number,
            heads_number=self.heads_number,
        )

        # UPSAMPLING (DECODER)
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(blocks_number):
                up.append(UpBlock(
                    in_channels,
                    out_channels,
                    feature_map_size * 4,
                    has_attention=self.has_attention[i],
                    groups_number=self.groups_number,
                    heads_number=self.heads_number)
                )
            out_channels = in_channels // multipliers[i]
            up.append(UpBlock(
                in_channels,
                out_channels,
                feature_map_size * 4,
                has_attention=self.has_attention[i],
                groups_number=self.groups_number,
                heads_number=self.heads_number)
            )
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        self.normalisation = nn.GroupNorm(8, feature_map_size)
        self.final = nn.Conv2d(in_channels, channels, kernel_size=(3, 3), padding=(1, 1))

    def swish_activation(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x, t):
        t = self.time_emb(t)
        x = self.image_proj(x)

        h = [x]

        # DOWNSAMPLING LAYER (ENCODER)
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # BOTTLENECK LAYER
        x = self.bottleneck(x, t)

        # UPSAMPLING LAYER (DECODER)
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.final(self.swish_activation(self.normalisation(x)))

    def display_images(self, images, epoch):
      # Áthelyezés CPU-ra a NumPy tömb létrehozásához
      grid = vutils.make_grid(images.cpu(), normalize=True, scale_each=True)
      plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
      plt.axis('off')
      plt.savefig(f'generált kép_{epoch}-ban.png')
      plt.show()





class DenoisingDiffusion:
    def __init__(self, epsilon_theta_model, beta_initial, beta_final, T, device):
        super().__init__()

        self.T = T
        self.epsilon_theta_model = epsilon_theta_model
        self.beta = torch.linspace(beta_initial, beta_final, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

    def forward_diffusion(self, x_0, t, epsilon=None):
        if epsilon is None:
            epsilon = torch.randn_like(x_0)

        mean = self.gather(self.alpha_bar, t) ** 0.5 * x_0
        variance = 1 - self.gather(self.alpha_bar, t)
        return mean + (variance ** 0.5) * epsilon

    def reverse_diffusion(self, x_t, t):
        epsilon_theta = self.epsilon_theta_model(x_t, t)
        alpha = self.gather(self.alpha, t)
        alpha_bar = self.gather(self.alpha_bar, t)
        epsilon_coefficient = (1 - alpha) / (1 - alpha_bar) ** .5

        mean = 1 / (alpha ** 0.5) * (x_t - epsilon_coefficient * epsilon_theta)
        variance = self.gather(self.sigma2, t)
        epsilon = torch.randn(x_t.shape, device=x_t.device)
        return mean + (variance ** .5) * epsilon

    def gather(self, consts, t):
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1, 1)

    def display_images(self, x):
        x = torch.clamp(x, 0.0, 1.0)
        plt.rcParams["figure.dpi"] = 175
        plt.imshow(cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
        plt.axis("off")
        plt.grid(False)
        plt.show()

def get_data(dataset_name, batch_size):
    
    if dataset_name == "cifar10":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        
        class_names = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        dataset = torchvision.datasets.CIFAR10("../data/cifar10", train=True, download=True, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    elif dataset_name == "ffhq_96":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(96),
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.ImageFolder("../data/ffhq/thumbnails128x128/", transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        
    elif dataset_name == "ffhq_128":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.ImageFolder("../data/ffhq/thumbnails128x128/", transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)
 
    else:
        raise NotImplementedError()

    return data_loader


SEED=45
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = ["cifar10", "ffhq_96", "ffhq_128"]
DATASET_NAME = DATASETS[1]

BATCH_SIZE = 128       # 64-el jól fut
IMAGE_SIZE = 32  # A CIFAR-10 képek mérete mindkét dimenzióban 32 pixel
CHANNELS = 3     # A CIFAR-10 képek RGB színmélységgel rendelkeznek (3 csatorna)

BETA_INITIAL = 1e-4
BETA_FINAL = 1e-2

FEATURE_MAP_SIZE = 64    # 64-el jól fut
GROUPS_NUMBER = 32
HEADS_NUMBER = 1
BLOCKS_NUMBER = 2    # 1-el jól fut

LEARNING_RATE = 2e-5     # 2e-5 -el jól fut
EPOCHS = 20
T = 2000     # 1000-el jól fut

CHECKPOINT_FREQUENCY = 10
CHECKPOINT_FILE = None # Otherwise, for example: "checpoints/cifar10_checkpoint_epoch_100.pt"


def get_data(dataset_name, batch_size):
    if dataset_name == "cifar10":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        
        class_names = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        dataset = torchvision.datasets.CIFAR10("../data/cifar10", train=True, download=True, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        return data_loader
    else:
        raise NotImplementedError()



torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
#data_loader = dataset.get_data(DATASET_NAME, BATCH_SIZE)

model_instance = EpsilonTheta(
    channels=CHANNELS,
    feature_map_size=FEATURE_MAP_SIZE,
    groups_number=GROUPS_NUMBER,
    heads_number=HEADS_NUMBER,
    blocks_number=BLOCKS_NUMBER,).to(DEVICE)

diffusion = DenoisingDiffusion(
    epsilon_theta_model=model_instance,
    beta_initial=BETA_INITIAL,
    beta_final=BETA_FINAL,
    T=T,
    device=DEVICE)

# Az optimizáló inicializálása, checkpoint betöltése (ha van ilyen)
optimiser = torch.optim.Adam(model_instance.parameters(), lr=LEARNING_RATE)
if CHECKPOINT_FILE is not None:
    checkpoint = torch.load(CHECKPOINT_FILE)
    epoch_start = checkpoint["epoch"]
    loss = checkpoint["loss"]
    losses = checkpoint["losses"]
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
else:
    epoch_start = 1
    losses = []


    
if epoch_start > EPOCHS:
    raise ValueError("Invalid number of epochs. Please choose a number greater than the number of epochs already trained.")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

for epoch in range(epoch_start, EPOCHS + 1):
    print(f"EPOCH {epoch}/{EPOCHS}")
    epoch_loss = 0.0
    
    for batch in tqdm.tqdm(data_loader, desc="Batch feldolgozás"):
        x_0 = batch[0].to(DEVICE)
        t = torch.randint(0, T, (BATCH_SIZE,), device=DEVICE, dtype=torch.long)
        epsilon = torch.randn_like(x_0)
        x_t = diffusion.forward_diffusion(x_0, t, epsilon)
        epsilon_theta = diffusion.epsilon_theta_model(x_t, t)
        loss = torch.functional.F.mse_loss(epsilon, epsilon_theta)
        epoch_loss += loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    epoch_loss /= len(data_loader)
    losses.append(epoch_loss)
    
    # Generálás és megjelenítés
    with torch.no_grad():
        x_T = torch.randn([64, CHANNELS, IMAGE_SIZE, IMAGE_SIZE], device=DEVICE)
        for t_ in tqdm_module.tqdm(range(0, T), desc="Zajszűrő lépések"):
            t = T - t_ - 1
            x_T = diffusion.reverse_diffusion(x_T, x_T.new_full((64, ), t, dtype=torch.long))
        model_instance.display_images(x_T, epoch)

        #kép mentése
        save_generated_images(x_T, '/content/drive/MyDrive/cifar_2', epoch)
        print("Kép mentve")
    
    #loss + fid kiírása
    print(f"{epoch}. epoch loss: {epoch_loss}")
    fid_value = calculate_fid(reference_images, x_T)
    print(f"{epoch}. fid érték: {fid_value}")
    print("-" * 100)



    # Checkpoint mentése
    if epoch % CHECKPOINT_FREQUENCY == 0:
        torch.save({
            "epoch": epoch + 1,
            "loss": loss,
            "losses": losses,
            "model_state_dict": model_instance.state_dict(),
            "optimiser_state_dict": optimiser.state_dict()
        }, f"ddpm_{DATASET_NAME}_checkpoint_epoch_{epoch}.pt")
