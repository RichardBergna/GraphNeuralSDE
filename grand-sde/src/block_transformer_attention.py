import torch
import torchsde
from function_transformer_attention import SpGraphTransAttentionLayer
from base_classes import ODEblock, SDEblock
from torch import distributions
from utils import get_rw_adj


class AttODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 1]), gamma=0.5):
    super(AttODEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t)

    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    # self.odefunc.edge_index, self.odefunc.edge_weight = data.edge_index, edge_weight=data.edge_attr
    edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                         fill_value=opt['self_loop_weight'],
                                         num_nodes=data.num_nodes,
                                         dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    # parameter trading off between attention and the Laplacian
    self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], opt['hidden_dim'], opt,
                                                          device, edge_weights=self.odefunc.edge_weight).to(device)

  def get_attention_weights(self, x):
    attention, values = self.multihead_att_layer(x, self.odefunc.edge_index)
    return attention

  def forward(self, x):
    t = self.t.type_as(x)
    self.odefunc.attention_weights = self.get_attention_weights(x)
    self.reg_odefunc.odefunc.attention_weights = self.odefunc.attention_weights
    integrator = self.train_integrator if self.training else self.test_integrator

    reg_states = tuple(torch.zeros(x.size(0)).to(x) for i in range(self.nreg))

    func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    state = (x,) + reg_states if self.training and self.nreg > 0 else x
    # TODO change this to SDE compatible.
    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options={'step_size': self.opt['adjoint_step_size']},
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        atol=self.atol,
        rtol=self.rtol)

    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple(st[1] for st in state_dt[1:])
      return z, reg_states
    else:
      z = state_dt[1]
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"






class AttSDEblock(SDEblock):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 0.01]), gamma=0.5):
    super(AttSDEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t)

    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    # self.odefunc.edge_index, self.odefunc.edge_weight = data.edge_index, edge_weight=data.edge_attr
    edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                         fill_value=opt['self_loop_weight'],
                                         num_nodes=data.num_nodes,
                                         dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      # See if adjoint really is important for sde.
      from torchsde import sdeint_adjoint as sdeint
    else:
      from torchsde import sdeint
      
    self.train_integrator = sdeint
    self.test_integrator = sdeint
    self.set_tol()
    # parameter trading off between attention and the Laplacian
    self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], opt['hidden_dim'], opt,
                                                          device, edge_weights=self.odefunc.edge_weight).to(device)
    # For the SDE brownian motion
    self.ts_vis = t.float()
    self.device = device
    
    
  def get_attention_weights(self, x):
    attention, values = self.multihead_att_layer(x, self.odefunc.edge_index)
    return attention
  
  def drift_function(self):
    return self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
  
  def forward(self, x):
    batch_size =  x.shape[0]
    t = self.t.type_as(x)
    self.odefunc.attention_weights = self.get_attention_weights(x)
    self.reg_odefunc.odefunc.attention_weights = self.odefunc.attention_weights
    integrator = self.train_integrator if self.training else self.test_integrator
    
    reg_states = tuple(torch.zeros(x.size(0)).to(x) for i in range(self.nreg))

    func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    self.drift_network_mp = func # This is the drift of the SDE
    state = (x,) + reg_states if self.training and self.nreg > 0 else x
    
    
    """ Latent Neural SDE Layer"""
    qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
    py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
    logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)

    # aug_y0 = state
    aug_y0_mp = torch.cat([state, torch.zeros(state.shape[0], 1).to(x)], dim=1)
    
    bm_mp = self._init_brownian_motion_mp(batch_size, aug_y0_mp)
    
    state_dt = integrator(
        sde=self,
        y0=aug_y0_mp,
        ts=torch.tensor([0, t[1]/10]).float(),
        method=self.method,
        bm=bm_mp,
        adaptive=self.adaptive,
        rtol=self.rtol,
        atol=self.atol,
        names={'drift': 'f_aug_mp', 'diffusion': 'g_aug_mp'}
    )
    z, logqp_path = state_dt[:, :, 0:self.sde_output_dim], state_dt[-1, :, self.sde_output_dim]
    state_dt = z
    logqp_mp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
    """ End Latent Neural SDE Layer"""
    
    
    
    # TODO change this to SDE compatible.
    # if self.opt["adjoint"] and self.training:
    #   state_dt = integrator(
    #     func, state, t,
    #     method=self.opt['method'],
    #     options={'step_size': self.opt['step_size']}, # Not sure if I can do this.
    #     adjoint_method=self.opt['adjoint_method'],
    #     adjoint_options={'step_size': self.opt['adjoint_step_size']},
    #     atol=self.atol,
    #     rtol=self.rtol,
    #     adjoint_atol=self.atol_adjoint,
    #     adjoint_rtol=self.rtol_adjoint)
    # else:
    #   state_dt = integrator(
    #     func, state, t,
    #     method=self.opt['method'],
    #     options={'step_size': self.opt['step_size']},
    #     atol=self.atol,
    #     rtol=self.rtol)

    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple(st[1] for st in state_dt[1:])
      return z, reg_states, logqp_mp
    else:
      z = state_dt[1]
      return z, logqp_mp 

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"