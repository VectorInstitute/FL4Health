import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, num_class ,z_dim):
        super().__init__()
        self.z_dim = 2  
        self.num_class = 10
        # encoder part
        self.fc1 = nn.Linear(x_dim+num_class, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc_mu = nn.Linear(h_dim2, z_dim)
        self.fc_logvar = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim+num_class, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x, y):
        x = torch.cat((x, y.float()), dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z, y):
        z = torch.cat((z, y.float()), dim=-1)
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x, y):
        # One-hot encoding
        zeros_t = torch.zeros(y.size(0), self.num_class).to(y.device)
        y_index = y.view(y.size(0), 1)
        y = zeros_t.scatter_(1, y_index, 1)
        # Feeding the input and the label vector to the encoder
        mu, log_var = self.encoder(x, y)
        z = self.sampling(mu, log_var)
        # Feeding the laten vector and label vector to the decoder
        return self.decoder(z, y), mu, log_var


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super().__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


class ConvVae(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2),   #12*12                                                                                                                                                                  
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2),  #4*4
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
                      
        # Decoder
        self.fc2 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 16 * 4 * 4),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 5, stride=2), #11*11
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 5, stride=2), # 14*14
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 4, stride=1), # 28*28

        )
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc2(z)
        z = self.fc3(z)
        z = z.view(-1, 16, 4, 4)
        z = self.deconv(z)
        return self.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2),   #12*12                                                                                                                                                                  
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2),  #4*4
            nn.ReLU(),
        )
        # Decoder
        self.convT = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 5, stride=2), #11*11
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 5, stride=2), # 14*14
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 4, stride=1), # 28*28

        )
        self.sigmoid = nn.Sigmoid()
        
    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convT(x)
        x = self.sigmoid(x)
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    



    