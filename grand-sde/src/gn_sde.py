
import math
import torch
import torchsde
from torchdiffeq import odeint
import torch
from torch import distributions, nn
from utils import Meter
from torch_geometric.nn import GCNConv
from bayesian_gcn import BayesianGCNConv



def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b

class LatentGraphSDE(torchsde.SDEIto):

    def __init__(self, in_net, drift_net, out_net, theta=1.0, mu=0.0, sigma=0.1, adaptive=True, method="srk", rtol =0.005, atol = 0.005, sde_output_dim=64, t0=0, t1=1.0, device=None, opt=None):
        super(LatentGraphSDE, self).__init__(noise_type="diagonal")
        
        
        sigma = opt['sigma'] if opt is not None else sigma
        
        logvar = math.log(sigma ** 2 / (2. * theta))
        self.sde_output_dim = sde_output_dim
        # Prior drift
        self.register_buffer("theta", torch.full((1, sde_output_dim), theta))
        self.register_buffer("mu", torch.full((1, sde_output_dim), mu))
        self.register_buffer("sigma", torch.full((1, sde_output_dim), sigma))

        # p(y0)
        self.register_buffer("py0_mean", torch.full((1, sde_output_dim), mu))
        self.register_buffer("py0_logvar", torch.full((1, sde_output_dim), logvar))

        # Approximate posterior drift
        self.net = drift_net

        # q(y0)
        self.qy0_mean = nn.Parameter(torch.full((1, sde_output_dim), mu), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.full((1, sde_output_dim), logvar), requires_grad=True)

        self.in_net = in_net
        self.projection_net = out_net
        
        # Arguments
        if device == None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else: 
            self.device = device
        self.adaptive = adaptive
        self.method = method
 
        if opt is not None:
                self.rtol = opt['rtol']
                self.atol = opt['rtol']
                t1 = opt['sde_t']
                self.ts_vis = torch.tensor([t0, t1]).float().to(self.device)    
                self.t0 = t0
                self.t1 = t1
        else:
            self.rtol = rtol
            self.atol = atol
            self.ts_vis = torch.tensor([t0, t1]).float().to(self.device)    
            self.t0 = t0
            self.t1 = t1
        
        self.opt = opt
        self.fm = Meter()
        self.bm = Meter()
        print('using file gn_sde.py')
        print('sigma', sigma)
        print("rtol", opt['rtol'])
        print("t1", t1)


    def f(self, t, y):  # Approximate posterior drift.
        if t.dim() == 0:
            t = t.unsqueeze(0).expand_as(y[:, :1])  # Expand t to match y's batch size, but keep it as a single feature
        # Positional encoding in transformers for time-inhomogeneous posterior.
        return self.net(torch.cat((t, y), dim=-1))

    def g(self, t, y):  # Shared diffusion.
        return self.sigma.repeat(y.size(0), 1)

    def h(self, t, y):  # Prior drift.
        prioir_drift = self.theta * (self.mu - y)
        return prioir_drift

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, 0:self.sde_output_dim]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:self.sde_output_dim]
        g = self.g(t, y)
        g_logqp = torch.zeros(y.shape[0], 1).to(y)
        return torch.cat([g, g_logqp], dim=1)
    
    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)
    
    def _init_brownian_motion(self, batch_size, aug_y0):
        # We need space-time Levy area to use the SRK solver
        bm =  torchsde.BrownianInterval(
            t0=self.ts_vis[0],
            t1=self.ts_vis[-1],
            size=(batch_size, aug_y0.shape[1]),
            device=self.device,
            levy_area_approximation='space-time'
        )
        return bm

    
    def forward(self, ts):
        batch_size = ts.shape[0]
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)

        y0 = self.in_net(ts)
        aug_y0 = torch.cat([y0, torch.zeros(y0.shape[0], 1).to(ts)], dim=1)
        
        bm = self._init_brownian_motion(batch_size, aug_y0)
        
        # ts_fixed = torch.linspace(self.t0, self.t1, 10).to(self.device)         
        aug_ys = torchsde.sdeint(
            sde=self,
            y0=aug_y0,
            ts=torch.tensor([self.t0, self.t1]).float().to(self.device),
            # ts=ts_fixed,
            method=self.method,
            bm=bm,
            adaptive=self.adaptive,
            rtol=self.rtol,
            atol=self.atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        ys, logqp_path = aug_ys[:, :, 0:self.sde_output_dim], aug_ys[-1, :, self.sde_output_dim]
        ys = self.projection_net(ys[-1])
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp



class GraphNeuralODE(torch.nn.Module):
    def __init__(self, ode_func, in_net=None, out_net=None, method='rk4', atol=1e-3, rtol=1e-4, t0=0, t1=1, device=None, opt=None):
        super(GraphNeuralODE, self).__init__()
        self.in_net = in_net
        self.projection_net = out_net

        self.ode_func = ode_func
        self.method = method
        
        self.atol = atol
        self.rtol = rtol
        self.t0 = t0
        self.t1 =t1
        self.opt=opt
        self.fm = Meter()
        self.bm = Meter()
        
        # Arguments
        if device == None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else: 
            self.device = device
        
        
    def f(self, t, y):
        if t.dim() == 0:
            t = t.unsqueeze(0).expand_as(y[:, :1])
        # print('y:', y.shape, y)
        # print('torch.cat((t, y), dim=-1)', torch.cat((t, y), dim=-1).shape, torch.cat((t, y), dim=-1))
        z = self.ode_func(torch.cat((t, y), dim=-1))
        return z

    def forward(self, x):
        # x = data.x
        x = self.in_net(x).to(self.device)
        # t = torch.tensor([0, 1]).float().to(device)  # Define the time steps
        t_span = torch.linspace(self.t0, self.t1, 101).to(self.device)

        # t_span = torch.linspace(t[0], t[1], steps=5)  # Adjust the number of steps as needed
        ys = odeint(self.f, x.to(self.device), t_span.to(self.device) , method=self.method, atol=self.atol, rtol=self.rtol)
        ode_output_time_1 = ys[-1].to(self.device)
        
        # The Prob. of the Classes, using just the last output of the ODE       
        y = self.projection_net(ode_output_time_1)
        # y = F.softmax(y, dim=1)  
        # y last layer output, ys all time steps outputs form the ODE
        return y, ys
    
    
    
    
class MyGCN2(torch.nn.Module):
      def __init__(self, edge_index, in_feats=32, out_feats=7, n_dimension=64, activation=None, dropout=0,  opt=None):
          super(MyGCN2, self).__init__()
          self.edge_index = edge_index
          self.dropout = dropout
          self.activation = activation
          self.conv1 = GCNConv(in_feats, n_dimension)
        #   self.conv2 = GCNConv(n_dimension, n_dimension)
          self.conv4 = GCNConv(n_dimension, out_feats)
          self.opt = opt
          self.fm = Meter()
          self.bm = Meter()

      def forward(self, x):
          x = self.conv1(x, self.edge_index)
          x = self.conv4(x, self.edge_index)
          return x, 0 # two outputs just to keep it consistent with the other models
      
      
      
class BayesianGCN(torch.nn.Module):
    def __init__(self, edge_index, in_feats=32, n_dimension=64, out_feats=7, activation=None, dropout=0, opt=None):
        super(BayesianGCN, self).__init__()
        """
        Create a Bayesian Graph Convolutional Network with two layers.
        
        Parameters:
            g: Graph structure (e.g., from DGL or PyTorch Geometric)
            num_features: Number of input features
            hidden_units: Number of units in the hidden layer
            num_classes: Number of output classes (or units in the output layer)
        """
        self.edge_index = edge_index
        self.dropout = dropout
        self.activation = activation
        self.bayes_conv1 = BayesianGCNConv(in_feats, n_dimension, prior_mean = 0.0, prior_variance = 1.0) 
        # self.bayes_conv2 = BayesianGCNConv(n_dimension, n_dimension, prior_mean = 0.0, prior_variance = 1.0) 
        self.bayes_conv3 = BayesianGCNConv(n_dimension, out_feats, prior_mean = 0.0, prior_variance = 1.0)
        self.opt = opt
        self.fm = Meter()
        self.bm = Meter()

    def forward(self, x, return_kl=True):
        x, kl1 = self.bayes_conv1(x, self.edge_index, return_kl=return_kl)
        x = self.activation(x)
        x, kl3 = self.bayes_conv3(x, self.edge_index, return_kl=return_kl)
        total_kl = kl1 + kl3
        return x, total_kl
        