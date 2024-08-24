import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from utils import Meter
from regularized_ODE_function import RegularizedODEfunc
import regularized_ODE_function as reg_lib
import six
import math
import torchsde
import torch.nn.functional as F



REGULARIZATION_FNS = {
    "kinetic_energy": reg_lib.quadratic_cost,
    "jacobian_norm2": reg_lib.jacobian_frobenius_regularization_fn,
    "total_deriv": reg_lib.total_derivative,
    "directional_penalty": reg_lib.directional_derivative
}


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if args[arg_key] is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(args[arg_key])

    regularization_fns = regularization_fns
    regularization_coeffs = regularization_coeffs
    return regularization_fns, regularization_coeffs

class ODEblock(nn.Module):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t):
    super(ODEblock, self).__init__()
    self.opt = opt
    self.t = t
    
    self.aug_dim = 2 if opt['augment'] else 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    
    self.nreg = len(regularization_fns)
    self.reg_odefunc = RegularizedODEfunc(self.odefunc, regularization_fns)

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = None
    self.set_tol()

  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()
    self.reg_odefunc.odefunc.x0 = x0.clone().detach()

  def set_tol(self):
    self.atol = self.opt['tol_scale'] * 1e-7
    self.rtol = self.opt['tol_scale'] * 1e-9
    if self.opt['adjoint']:
      self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
      self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

  def reset_tol(self):
    self.atol = 1e-7
    self.rtol = 1e-9
    self.atol_adjoint = 1e-7
    self.rtol_adjoint = 1e-9

  def set_time(self, time):
    self.t = torch.tensor([0, time]).to(self.device)

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
           
           
# TODO The SDE version ODEblock for the diffusion message passing: We dont need this
class SDEblock(torchsde.SDEIto, nn.Module):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t, theta=1.0, mu=0.0, sigma=0.10, adaptive=False, method="srk"): # TODO Make this opt
    torchsde.SDEIto.__init__(self, noise_type="diagonal")
    nn.Module.__init__(self)
    self.opt = opt
    self.t = t
    self.aug_dim = 2 if opt['augment'] else 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    
    self.nreg = len(regularization_fns)
    self.reg_odefunc = RegularizedODEfunc(self.odefunc, regularization_fns)

    if opt['adjoint']:
      from torchsde import sdeint_adjoint as sdeint
    else:
      from torchsde import sdeint
    self.train_integrator = sdeint
    self.test_integrator = None
    self.set_tol()
    
    
    """ Beginning of SDE Init """  
    logvar = math.log(sigma ** 2 / (2. * theta))
    self.sde_output_dim = self.aug_dim * opt['hidden_dim']
    # Prior drift
    self.register_buffer("theta", torch.full((1, self.aug_dim * opt['hidden_dim']), theta))
    self.register_buffer("mu", torch.full((1, self.aug_dim * opt['hidden_dim']), mu))
    self.register_buffer("sigma", torch.full((1, self.aug_dim * opt['hidden_dim']), sigma))

    # p(y0)
    self.register_buffer("py0_mean", torch.full((1, self.aug_dim * opt['hidden_dim']), mu))
    self.register_buffer("py0_logvar", torch.full((1, self.aug_dim * opt['hidden_dim']), logvar))
    
    self.qy0_mean = nn.Parameter(torch.full((1, self.aug_dim *opt['hidden_dim']), mu), requires_grad=True)
    self.qy0_logvar = nn.Parameter(torch.full((1, self.aug_dim *opt['hidden_dim']), logvar), requires_grad=True)
    self.adaptive = adaptive
    self.method = method
    # self.rtol = rtol
    # self.atol = atol
    # self.ts_vis = torch.tensor([t0, t1]).float().to(self.device)    
    # self.t0 = t0
    # self.t1 = t1
    """ END of GNSDE Init """

  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()
    self.reg_odefunc.odefunc.x0 = x0.clone().detach()

  def set_tol(self):
    self.atol = self.opt['tol_scale'] * 1e-7
    self.rtol = self.opt['tol_scale'] * 1e-9
    if self.opt['adjoint']:
      self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
      self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

  def reset_tol(self):
    self.atol = 1e-7
    self.rtol = 1e-9
    self.atol_adjoint = 1e-7
    self.rtol_adjoint = 1e-9

  def set_time(self, time):
    self.t = torch.tensor([0, time]).to(self.device)

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
  """ Beginning of the SDE functions """
  def _stable_division(self, a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    # b_safe = torch.where(b.abs() > epsilon, b, epsilon * b.sign())
    return a / b
  
  def f_drift_mp(self, t, y):  # Approximate posterior drift.
    if t.dim() == 0:
        t = torch.full_like(y, fill_value=t)
    
    # Positional encoding in transformers for time-inhomogeneous posterior.
    return self.drift_network_mp(y, t)

  def g_diffusion_mp(self, t, y):  # Shared diffusion.
      return self.sigma.repeat(y.size(0), 1)

  def h_prior_mp(self, t, y):  # Prior drift.
      prioir_drift = self.theta * (self.mu - y)
      return prioir_drift

  def f_aug_mp(self, t, y):  # Drift for augmented dynamics with logqp term.
      y = y[:, 0:self.sde_output_dim]
      # f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
      f =  self.f_drift_mp(t, y)
      g = self.g_diffusion_mp(t, y)
      h = self.h_prior_mp(t, y)
      u = self._stable_division(f-h, g)
      f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
      return torch.cat([f, f_logqp], dim=1)

  def g_aug_mp(self, t, y):  # Diffusion for augmented dynamics with logqp term.
      y = y[:, 0:self.sde_output_dim]
      g = self.g_diffusion_mp(t, y)
      g_logqp = torch.zeros(y.shape[0], 1).to(y)
      return torch.cat([g, g_logqp], dim=1)
  
  def _init_brownian_motion_mp(self, batch_size, aug_y0):

      # We need space-time Levy area to use the SRK solver
      bm =  torchsde.BrownianInterval(
          t0=self.ts_vis[0],
          t1=self.ts_vis[-1],
          size=(batch_size, aug_y0.shape[1]),
          device=self.device,
          levy_area_approximation='space-time'
      )
      return bm
    
  @property
  def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

  @property
  def qy0_std(self):
      return torch.exp(.5 * self.qy0_logvar)
  """ End of the SDE functions """
           





class ODEFunc(MessagePassing):

  # currently requires in_features = out_features
  def __init__(self, opt, data, device):
    super(ODEFunc, self).__init__()
    self.opt = opt
    self.device = device
    self.edge_index = None
    self.edge_weight = None
    self.attention_weights = None
    self.alpha_train = nn.Parameter(torch.tensor(0.0))
    self.beta_train = nn.Parameter(torch.tensor(0.0))
    self.x0 = None
    self.nfe = 0
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

  def __repr__(self):
    return self.__class__.__name__


class BaseGNN(MessagePassing):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(BaseGNN, self).__init__()
    self.opt = opt
    self.T = opt['time']
    self.num_classes = dataset.num_classes
    self.num_features = dataset.data.num_features
    self.num_nodes = dataset.data.num_nodes
    self.device = device
    self.fm = Meter()
    self.bm = Meter()

    if opt['beltrami']:
      self.mx = nn.Linear(self.num_features, opt['feat_hidden_dim'])
      self.mp = nn.Linear(opt['pos_enc_dim'], opt['pos_enc_hidden_dim'])
      opt['hidden_dim'] = opt['feat_hidden_dim'] + opt['pos_enc_hidden_dim']
    else:
      # "This can be the embedder"
      self.m1 = nn.Linear(self.num_features, opt['hidden_dim'])

    # All of these can be the Neural SDE, but we layer min.
    if self.opt['use_mlp']:
      self.m11 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
      self.m12 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    if opt['use_labels']:
      # todo - fastest way to propagate this everywhere, but error prone - refactor later
      opt['hidden_dim'] = opt['hidden_dim'] + dataset.num_classes
    else:
      self.hidden_dim = opt['hidden_dim']
    if opt['fc_out']:
      self.fc = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
      
    # This is the output later
    self.m2 = nn.Linear(opt['hidden_dim'], dataset.num_classes)
    if self.opt['batch_norm']:
      self.bn_in = torch.nn.BatchNorm1d(opt['hidden_dim'])
      self.bn_out = torch.nn.BatchNorm1d(opt['hidden_dim'])

    self.regularization_fns, self.regularization_coeffs = create_regularization_fns(self.opt)

  def getNFE(self):
    return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe

  def resetNFE(self):
    self.odeblock.odefunc.nfe = 0
    self.odeblock.reg_odefunc.odefunc.nfe = 0

  def reset(self):
    self.m1.reset_parameters()
    self.m2.reset_parameters()

  def __repr__(self):
    return self.__class__.__name__
  
  
  


# Define the Drift Layer for the Neural SDE model.
class DriftLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, dropout=0):
        super(DriftLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.layer = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        x = self.layer(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.training:  # Apply dropout only during training
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
      
class BaseGNSDE(torchsde.SDEIto, MessagePassing):
  def __init__(self, opt, dataset, device=torch.device('cpu'), theta=1.0, mu=0.0, sigma=0.1, adaptive=True, method="srk", rtol =0.005, atol = 0.005, t0=0, t1=1.0):
    torchsde.SDEIto.__init__(self, noise_type="diagonal")
    MessagePassing.__init__(self)
    self.opt = opt
    opt['time'] = opt['time']
    self.T = opt['time']
    sigma = opt['sigma']
    print("sigma", sigma)
    self.num_classes = dataset.num_classes
    self.num_features = dataset.data.num_features
    self.num_nodes = dataset.data.num_nodes
    self.device = device
    self.fm = Meter()
    self.bm = Meter()

    if opt['beltrami']:
      self.mx = nn.Linear(self.num_features, opt['feat_hidden_dim'])
      self.mp = nn.Linear(opt['pos_enc_dim'], opt['pos_enc_hidden_dim'])
      opt['hidden_dim'] = opt['feat_hidden_dim'] + opt['pos_enc_hidden_dim']
    else:
      # "This can be the embedder"
      self.m1 = nn.Linear(self.num_features, opt['hidden_dim'])

    # All of these can be the Neural SDE, but we layer min.
    if self.opt['use_mlp']:
      self.m11 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
      self.m12 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    if opt['use_labels']:
      # todo - fastest way to propagate this everywhere, but error prone - refactor later
      opt['hidden_dim'] = opt['hidden_dim'] + dataset.num_classes
    else:
      self.hidden_dim = opt['hidden_dim']
    if opt['fc_out']:
      self.fc = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
      
    # This is the output later
    self.m2 = nn.Linear(opt['hidden_dim'], dataset.num_classes)
    if self.opt['batch_norm']:
      self.bn_in = torch.nn.BatchNorm1d(opt['hidden_dim'])
      self.bn_out = torch.nn.BatchNorm1d(opt['hidden_dim'])

    self.regularization_fns, self.regularization_coeffs = create_regularization_fns(self.opt)
    
    
    """ BEginning of GNSDE Init """  
    logvar = math.log(sigma ** 2 / (2. * theta))
    self.sde_output_dim = opt['hidden_dim']
    # Prior drift
    self.register_buffer("theta", torch.full((1, opt['hidden_dim']), theta))
    self.register_buffer("mu", torch.full((1, opt['hidden_dim']), mu))
    self.register_buffer("sigma", torch.full((1, opt['hidden_dim']), sigma))

    # p(y0)
    self.register_buffer("py0_mean", torch.full((1, opt['hidden_dim']), mu))
    self.register_buffer("py0_logvar", torch.full((1, opt['hidden_dim']), logvar))

    # Approximate posterior drift
    self.gnsde_drift_net = nn.Sequential(
      DriftLayer(in_feats=opt['hidden_dim'] + 1, out_feats=opt['hidden_dim'], activation=nn.Softplus(), dropout=self.opt['dropout']), # The +1 is for the time
      DriftLayer(in_feats=opt['hidden_dim'], out_feats=opt['hidden_dim'], activation=None, dropout=self.opt['dropout'])
      ).to(device)
    
    self.qy0_mean = nn.Parameter(torch.full((1, opt['hidden_dim']), mu), requires_grad=True)
    self.qy0_logvar = nn.Parameter(torch.full((1, opt['hidden_dim']), logvar), requires_grad=True)
    self.adaptive = adaptive
    self.method = method
    self.rtol = opt['rtol']
    print("rtol", opt['rtol'])
    self.atol = opt['rtol']
    t1 = opt['sde_t']
    print("t1", t1)
    self.ts_vis = torch.tensor([t0, t1]).float().to(self.device)    
    self.t0 = t0
    self.t1 = t1
    """ END of GNSDE Init """
    
  def getNFE(self):
    return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe

  def resetNFE(self):
    self.odeblock.odefunc.nfe = 0
    self.odeblock.reg_odefunc.odefunc.nfe = 0

  def reset(self):
    self.m1.reset_parameters()
    self.m2.reset_parameters()

  def __repr__(self):
    return self.__class__.__name__
  
  """ Beginning of the GNSDE functions """
  def _stable_division(self, a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    # b_safe = torch.where(b.abs() > epsilon, b, epsilon * b.sign())
    return a / b
  
  def f_drift(self, t, y):  # Approximate posterior drift.
    if t.dim() == 0:
        t = t.unsqueeze(0).expand_as(y[:, :1])  # Expand t to match y's batch size, but keep it as a single feature
    return self.gnsde_drift_net(torch.cat((t, y), dim=-1))

  def g_diffusion(self, t, y):  # Shared diffusion.
      return self.sigma.repeat(y.size(0), 1)

  def h_prior(self, t, y):  # Prior drift.
      prioir_drift = self.theta * (self.mu - y)
      return prioir_drift

  def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
      y = y[:, 0:self.sde_output_dim]
      # f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
      f =  self.f_drift(t, y)
      g = self.g_diffusion(t, y)
      h = self.h_prior(t, y)
      u = self._stable_division(f-h, g)
      f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
      return torch.cat([f, f_logqp], dim=1)

  def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
      y = y[:, 0:self.sde_output_dim]
      g = self.g_diffusion(t, y)
      g_logqp = torch.zeros(y.shape[0], 1).to(y)
      return torch.cat([g, g_logqp], dim=1)
  
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
  @property
  def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

  @property
  def qy0_std(self):
      return torch.exp(.5 * self.qy0_logvar)
  """ End of the GNSDE functions """