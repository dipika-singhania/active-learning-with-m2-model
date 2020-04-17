import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn.functional as F


class PrintLayer(nn.Module):
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.name, ":", x.shape)
        return x

def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.
    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.
    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """
    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparametrize(mu, log_var), mu, log_var


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class LinearEncoder(nn.Module):
    def __init__(self, dims, sample_layer=GaussianSample):
        """
        Inference network
        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).
        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(LinearEncoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.sample(x)


class FashionMnistEncoder(nn.Module):
    def __init__(self, dims, sample_layer=GaussianSample):
        """
        Inference network
        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).
        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(FashionMnistEncoder, self).__init__()

        [x_dim, h_dim, z_dim, y_dim] = dims
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=x_dim[0], out_channels=h_dim[0], kernel_size=4, stride=2, padding=1, bias=False),
            # Input : bs * x_dim[0] * x_dim[1] * x_dim[2], Output: bs * 64 * x_dim[1]/2 * x_dim[2]/2
            nn.BatchNorm2d(h_dim[0]),
            nn.ReLU(True),
            nn.Conv2d(in_channels=h_dim[0], out_channels=h_dim[1], kernel_size=4, stride=2, padding=1, bias=False),
            # Input : bs * 64 * x_dim[1]/2 * x_dim[2]/2, Output: bs * 32 * x_dim[1]/4 * x_dim[2]/4
            nn.BatchNorm2d(h_dim[1]),
            nn.ReLU(True),
            View((-1, h_dim[1] * x_dim[1]//4 * x_dim[2]//4)),
            nn.Linear(h_dim[1] * x_dim[1]//4 * x_dim[2]//4, h_dim[-1]),
            nn.BatchNorm1d(h_dim[-1]),
            nn.ReLU(True),
        )

        # self.sample = sample_layer(h_dim[-1] + y_dim, z_dim)
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, x, y):
        x = self.encoder(x)
        # return self.sample(torch.cat([x, y], dim=-1))
        return self.sample(x)


class LinearDecoder(nn.Module):
    def __init__(self, dims):
        """
        Generative network
        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).
        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(LinearDecoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims

        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)

        self.reconstruction = nn.Linear(h_dim[-1], x_dim)

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output_activation(self.reconstruction(x))


class FashionMnistDecoder(nn.Module):
    def __init__(self, dims):
        """
        Generative network
        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).
        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(FashionMnistDecoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        self.up_sample = nn.Linear(z_dim, h_dim[-1])
        self.decoder = nn.Sequential(
            nn.Linear(h_dim[-1], h_dim[1] * x_dim[1] // 4 * x_dim[2] // 4),
            nn.BatchNorm1d(h_dim[1] * x_dim[1] // 4 * x_dim[2] // 4),
            nn.ReLU(True),
            View((-1, h_dim[1], x_dim[1] // 4,  x_dim[2] // 4)),
            nn.ConvTranspose2d(in_channels=h_dim[1], out_channels=h_dim[0], kernel_size=4, stride=2, padding=1, bias=False),
            # Input : bs * 32 * x_dim[1]/4 * x_dim[2]/4 Output: bs * 64 * x_dim[1]/2 * x_dim[2]/2
            nn.BatchNorm2d(h_dim[0]),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=h_dim[0], out_channels=x_dim[0], kernel_size=4, stride=2, padding=1, bias=False),
            # Input : bs * 64 * x_dim[1]/2 * x_dim[2]/2, Output: bs * x_dim[0] * x_dim[1] * x_dim[2],
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        upscale_x = self.up_sample(x)
        recon_x = self.decoder(upscale_x)
        return self.output_activation(recon_x)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, dims, dataset):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].
        :param dims: x, z and hidden dimensions of the networks
        """
        super(VariationalAutoEncoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.flow = None
        if dataset is 'MNIST':
            self.encoder = LinearEncoder([x_dim, h_dim, z_dim])
            self.decoder = LinearDecoder([z_dim, list(reversed(h_dim)), x_dim])
        elif dataset == 'FashionMNIST':
            self.encoder = FashionMnistEncoder([x_dim, h_dim, z_dim, self.y_dim])
            self.decoder = FashionMnistDecoder([x_dim, h_dim, z_dim])
            self.classifier = FashionMnistClassifier([x_dim, h_dim, self.y_dim])

        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.
        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]
        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz

        return kl

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.
        :param x: input data
        :return: reconstructed input
        """
        z, z_mu, z_log_var = self.encoder(x)

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        x_mu = self.decoder(z)

        return x_mu

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)


class FashionMnistClassifier(nn.Module):
    def __init__(self, dims):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(FashionMnistClassifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=x_dim[0], out_channels=h_dim[0], kernel_size=4, stride=2, padding=1, bias=False),
            # Input : bs * x_dim[0] * x_dim[1] * x_dim[2], Output: bs * 64 * x_dim[1]/2 * x_dim[2]/2
            nn.BatchNorm2d(h_dim[0]),
            nn.ReLU(True),
            nn.Conv2d(in_channels=h_dim[0], out_channels=h_dim[1], kernel_size=4, stride=2, padding=1, bias=False),
            # Input : bs * 64 * x_dim[1]/2 * x_dim[2]/2, Output: bs * 32 * x_dim[1]/4 * x_dim[2]/4
            nn.BatchNorm2d(h_dim[1]),
            nn.ReLU(True),
            View((-1, h_dim[1] * x_dim[1] // 4 * x_dim[2] // 4)),
            nn.Linear(h_dim[1] * x_dim[1] // 4 * x_dim[2] // 4, h_dim[-1]),
            nn.BatchNorm1d(h_dim[-1]),
            nn.ReLU(True),
        )
        self.final_layer = nn.Linear(h_dim[-1], y_dim)
        PrintLayer("FinalLater"),

    def forward(self, x):
        x = self.encoder(x)
        x = F.softmax(self.final_layer(x), dim=-1)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, dims):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(LinearClassifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x


class DeepGenerativeModel(VariationalAutoEncoder):
    def __init__(self, dims, dataset):
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.
        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.
        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim], dataset)

        self.dataset = dataset
        if dataset == 'MNIST':
            self.flatten = View((-1, x_dim))
            self.encoder = LinearEncoder([x_dim + self.y_dim, h_dim, z_dim])
            self.decoder = LinearDecoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
            self.classifier = LinearClassifier([x_dim, h_dim[0], self.y_dim])
        elif dataset == 'FashionMNIST':
            self.encoder = FashionMnistEncoder([x_dim, h_dim, z_dim, self.y_dim])
            self.decoder = FashionMnistDecoder([x_dim, h_dim, z_dim + self.y_dim])
            self.classifier = FashionMnistClassifier([x_dim, h_dim, self.y_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Add label and data and generate latent variable
        if self.dataset == 'MNIST':
            x = self.flatten(x)
            z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))
        else:
            z, z_mu, z_log_var = self.encoder(x, y)

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))
        return x_mu, z, z_mu, z_log_var

    def classify(self, x):
        if self.dataset == 'MNIST':
            x = self.flatten(x)
        logits = self.classifier(x)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10, class_probs=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim + class_probs
        # self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z, pred_probability):
        z_new = torch.cat((z, pred_probability), dim=1)
        return self.net(z_new).squeeze()


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()



class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, class_probs=10):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.class_probs = class_probs
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),  # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*2*2)),                       # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + self.class_probs, 1024*4*4),                           # B, 1024*8*8
            View((-1, 1024, 4, 4)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()
        # self.fc_logvar.weight.data.uniform_()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x, pred_probability):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)

        z = self.reparameterize(mu, logvar)
        z_new = torch.cat((z, pred_probability), dim=1)
        x_recon = self._decode(z_new)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

