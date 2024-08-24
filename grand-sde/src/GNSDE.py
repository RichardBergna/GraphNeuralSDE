import torch
from torch import nn, distributions
import torch.nn.functional as F
from base_classes import BaseGNSDE
from model_configurations import set_block, set_function
import torchsde



# Todo Make GNSDE, with Deterministic diffusion
class GNSDE(BaseGNSDE):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNSDE, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device) # Try to use this for our GNSDE
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)

  def forward(self, x, pos_encoding=None):
    batch_size =  x.shape[0]
    # Encode each node based on its feature.
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]

    if self.opt['beltrami']:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
      p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
      p = self.mp(p)
      x = torch.cat([x, p], dim=1)
    else:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.m1(x)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper

    if self.opt['use_labels']:
      print("self.opt['use_labels']", self.opt['use_labels'])
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      print("self.opt['batch_norm']", self.opt['batch_norm'])
      x = self.bn_in(x)
    
    
    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      print("self.opt['batch_norm']", self.opt['batch_norm'])
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)
    
    self.odeblock.set_x0(x)

    kl_loss_diffusion = 0
    if self.training  and self.odeblock.nreg > 0:
      if self.opt['use_stochastic_grand']:
        z, self.reg_states, kl_loss_diffusion = self.odeblock(x) 
      else:
        z, self.reg_states  = self.odeblock(x)
    else:
      if self.opt['use_stochastic_grand']:
        z, kl_loss_diffusion = self.odeblock(x)
      else:
        z = self.odeblock(x)

    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    
    """ Latent Neural SDE Layer"""
    qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
    py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
    logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)

    aug_y0 = torch.cat([z, torch.zeros(z.shape[0], 1).to(x)], dim=1)
    bm = self._init_brownian_motion(batch_size, aug_y0)
    

    aug_ys = torchsde.sdeint(
        sde=self,
        y0=aug_y0,
        ts=torch.tensor([self.t0, self.t1]).float().to(self.device),
        method=self.method,
        bm=bm,
        adaptive=self.adaptive,
        rtol=self.rtol,
        atol=self.atol,
        names={'drift': 'f_aug', 'diffusion': 'g_aug'}
    )
    z, logqp_path = aug_ys[:, :, 0:self.sde_output_dim], aug_ys[-1, :, self.sde_output_dim]
    """ End Latent Neural SDE Layer"""
     
    z = self.m2(z[-1])
    logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
    
    total_kl_loss = logqp + kl_loss_diffusion
    return z, total_kl_loss

