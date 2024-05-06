import time
import torch

torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mse_loss = torch.nn.MSELoss()


def recover_image(img):
    return (
            (
                    img *
                    np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
                    np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
            ).transpose(0, 2, 3, 1) *
            255.
    ).clip(0, 255).astype(np.uint8)


#Formula for Gram Matrix
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


#save the image that the variational autoencoder gives, the input image and the output image
def save_debug_image(tensor_orig, tensor_transformed, filename):
    assert tensor_orig.size() == tensor_transformed.size()
    result = Image.fromarray(recover_image(tensor_transformed.cpu().numpy())[0])
    orig = Image.fromarray(recover_image(tensor_orig.cpu().numpy())[0])
    save_image = Image.new('RGB', (result.size[0] * 2 + 5, result.size[1]))
    save_image.paste(orig, (0, 0))
    save_image.paste(result, (result.size[0] + 5, 0))
    save_image.save(filename)


# The Loss Network to determine perceptual loss
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # only return stuff from relu 1 to 4
        self.vgg_layers = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    # Forward Pass
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output


# Style picture, can be changed if needed.
style_image = "../Style/candy.jpg"
style_img = Image.open(style_image).convert('RGB')
vgg = VGGPerceptualLoss().to(device)


# Normalize the inputs
def tensor_normalizer():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


# Add the Style picture as a tensor and get its gram matrix to compare with the output picture for style loss
with torch.no_grad():
    style_img_tensor = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        tensor_normalizer()]
    )(style_img).unsqueeze(0)

    style_img_tensor = style_img_tensor.to(device)
    style_features = vgg(style_img_tensor)
    gram_style = [gram_matrix(y) for y in style_features.values()]


# run the program
def run():
    vae2 = VAE().to(device)
    transform = transforms.Compose([transforms.ToTensor(), tensor_normalizer()])
    data = torch.utils.data.DataLoader(
        torchvision.datasets.STL10('./data',
                                   transform=transform,
                                   download=True),
        batch_size=50,
        shuffle=True)

    vae2 = train(vae2, data)


#autoencoder
class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=512):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )

        # Outputting the mean and variance
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
            nn.Sigmoid()  # Sigmoid to get pixel values between 0 and 1
        )

    # Reparametrize the program
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Forward Pass
    def forward(self, x):
        # Encode, get x, mu and logvar
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar


#Determine KL loss
def loss_function(mu, logvar):
    # Compute KL divergence loss
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return kl_divergence


# Weights
CONTENT_WEIGHT = 1
REGULARIZATION = 1e-7
STYLE_WEIGHTS = np.array([1e-1, 1, 1e1, 5]) * 5e1
LOG_INTERVAL = 10


# Train the autoencoder
def train(autoencoder, data, epochs=60):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)
    agg_epoch_content_loss = 0.
    agg_epoch_style_loss = 0.
    agg_content_loss = 0.
    agg_style_loss = 0.
    agg_reg_loss = 0.
    content_loss_values = []
    style_loss_values = []
    for epoch in range(epochs):
        count = 0
        for x, _ in data:

            #Train Autoencoder
            count += 1
            opt.zero_grad()
            x = x.to(device)  # GPU

            e, mu, logvar = autoencoder(x)

            with torch.no_grad():
                xc = x.detach()

            # Calculate Feature Loss
            features_x = vgg(xc)
            with torch.no_grad():
                xc_3 = features_x["relu2_2"].detach()
            y = vgg(e)
            feature_loss = mse_loss(xc_3, y["relu2_2"]) * CONTENT_WEIGHT
            # Calculate Regular Loss
            reg_loss = REGULARIZATION * (
                    torch.sum(torch.abs(e[:, :, :, :-1] - e[:, :, :, 1:])) +
                    torch.sum(torch.abs(e[:, :, :-1, :] - e[:, :, 1:, :])))

            # Calculate Style Loss
            style_loss = 0.
            dict2 = {0: "relu1_2",
                     1: "relu2_2",
                     2: "relu3_3",
                     3: "relu4_3"}
            for l, weight in enumerate(STYLE_WEIGHTS):
                gram_s = gram_style[l]
                gram_y = gram_matrix(y[dict2[l]])
                style_loss += float(weight) * mse_loss(gram_y, gram_s.expand_as(gram_y))

            # Combine the loss and do backward pass.
            overall_loss = feature_loss + reg_loss + style_loss
            overall_loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 2)
            opt.step()

            # Aggregate the losses
            agg_content_loss += feature_loss
            agg_epoch_content_loss += feature_loss
            agg_style_loss += style_loss
            agg_epoch_style_loss += style_loss

            agg_reg_loss += reg_loss

            # Log message every 10 steps
            if count % LOG_INTERVAL == 0:
                mesg = "{} [Epoch: {} Step: {}] content: {:.2f} regular:{:.2f} style:{:.2f} total:{:.6f}".format(
                    time.ctime(), epoch, count,
                    agg_content_loss / LOG_INTERVAL,
                    agg_reg_loss / LOG_INTERVAL,
                    agg_style_loss / LOG_INTERVAL,
                    (agg_content_loss + agg_reg_loss + agg_style_loss) / LOG_INTERVAL
                )
                print(mesg)
                agg_content_loss = 0.
                agg_style_loss = 0.
                agg_reg_loss = 0.
                y, mu, logvar = autoencoder(x)
                save_debug_image(x, y.detach(), "../debug/{}-{}.png".format(epoch, count))

        # Get loss for each epoch for graph
        agg_epoch_style_loss_g = (agg_epoch_style_loss / count).detach()
        agg_epoch_content_loss_g = (agg_epoch_content_loss / count).detach()
        content_loss_values.append(agg_epoch_content_loss_g)
        style_loss_values.append(agg_epoch_style_loss_g)
        agg_epoch_content_loss = 0.
        agg_epoch_style_loss = 0.

    # Create graph of loss
    np_content_loss = np.array([x.detach().cpu().numpy() for x in content_loss_values])
    np_style_loss = np.array([x.detach().cpu().numpy() for x in style_loss_values])
    plt.plot(np_content_loss, color="r", label="Content Loss")
    plt.plot(np_style_loss, color="b", label="Style Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    return autoencoder


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
