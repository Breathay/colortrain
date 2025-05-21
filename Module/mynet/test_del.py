# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributions as dist

pi = torch.rand((10, 3))
mu = 2 * torch.rand((10, 3 * 4)) - 1
sd = torch.rand((10, 3 * 4))

pi = F.softmax(pi, dim=1)
mu = mu.reshape(10, 3, 4)
sd = sd.reshape(10, 3, 4)

x = torch.linspace(-2, 2, 200)

y = 0

y1 = pi[0, 0] * torch.exp(-torch.square(x - mu[0, 0, 0]) / (2 * torch.square(sd[0, 0, 0])))
y2 = pi[0, 1] * torch.exp(-torch.square(x - mu[0, 1, 0]) / (2 * torch.square(sd[0, 1, 0])))
y3 = pi[0, 2] * torch.exp(-torch.square(x - mu[0, 2, 0]) / (2 * torch.square(sd[0, 2, 0])))

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y1 + y2 + y3)
plt.show()

def reparametrization(mu, pi, sd, n_samples):
    batch_size, num_gaussians, output_dim = mu.shape
    
    indices = torch.multinomial(pi, n_samples, replacement=True)  
    indices = indices.unsqueeze(-1).expand(-1, -1, output_dim)
    
    selected_mu = torch.gather(mu, 1, indices)  
    selected_sd = torch.gather(sd, 1, indices)  
    
    normal_dist = dist.Normal(selected_mu, selected_sd)
    samples = normal_dist.rsample()  # 使用 rsample 支持反向传播
    
    return samples

samples = reparametrization(mu, pi, sd, 100)
plt.hist(samples[0, :, 0])



