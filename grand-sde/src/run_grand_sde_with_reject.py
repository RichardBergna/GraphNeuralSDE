import argparse
import time
import os
import json  
import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
import gc  # Garbage Collector
import copy

from GNN import GNN
from GNSDE import GNSDE
from GNN_early import GNNEarly
from GNSDE_early import GNSDEEarly

from GNN_KNN import GNN_KNN
from GNN_KNN_early import GNNKNNEarly
from data import get_dataset, set_train_val_test_split
from graph_rewiring import apply_KNN, apply_beltrami
from best_params import best_params_dict
from utils import ROOT_DIR


def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
  onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
  if idx.dtype == torch.bool:
    idx = torch.where(idx)[0]  # convert mask to linear index
  onehot[idx, labels.squeeze()[idx]] = 1

  return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
  """
  when using labels as features need to split training nodes into training and prediction
  """
  if data.train_mask.dtype == torch.bool:
    idx = torch.where(data.train_mask)[0]
  else:
    idx = data.train_mask
  mask = torch.rand(idx.shape) < mask_rate
  train_label_idx = idx[mask]
  train_pred_idx = idx[~mask]
  return train_label_idx, train_pred_idx

# This is for the KL loss for GNSDE
class LinearScheduler(object):
    def __init__(self, iters, maxval=1):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val

kl_scheduler = LinearScheduler(iters=100)

# Make this for GNSDE
def train(model, optimizer, data, pos_encoding=None):
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  out, kl_div = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
  if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
    reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
    regularization_coeffs = model.regularization_coeffs

    reg_loss = sum(
      reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
    )
    loss = loss + reg_loss

  total_loss = loss + kl_div*kl_scheduler.val
  model.fm.update(model.getNFE())
  model.resetNFE()
  total_loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()


def train_OGB(model, optimizer, data, pos_encoding=None):
  model.train()
  # print("1",  mp)
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  # print("1",  mp)
  # pos_encoding = mp(pos_encoding).to(model.device)
  # print("2", pos_encoding)
  out, _ = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
  if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
    reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
    regularization_coeffs = model.regularization_coeffs

    reg_loss = sum(
      reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
    )
    loss = loss + reg_loss

  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()


""" Begin: Evaluate with Entropy """
def entropy_fn(prediction_probabilities):
    eps = 1e-9
    p_clipped = torch.clamp(prediction_probabilities, eps, 1 - eps)
    entropy = -(p_clipped * torch.log(p_clipped)).sum(dim=1)
    return entropy
        
        
def mc_sample_stats(model, feat, n_samples=1, pos_encoding=None):
    samples = []
    samples_entropy = []
    for _ in range(n_samples):
        model.odeblock.odefunc.nfe = 0
        pred, _ = model(feat, pos_encoding)
        output_probability = F.softmax(pred, dim=-1)
        sample = output_probability
        samples.append(sample)
        sample_entropy = entropy_fn(sample)
        samples_entropy.append(sample_entropy)
    
    samples_tensor = torch.stack(samples) # to keep it consistant we keep the samples mean and var as torch tensors.
    var = torch.var(samples_tensor, dim=0, unbiased=True).sum(dim=1)  # Sum across features for each sample
    mean = torch.mean(samples_tensor, dim=0)
    mean_entropy = entropy_fn(mean)
    return var, mean, mean_entropy
    

def accuracy_with_entropy_threshold(prediction_probabilities, entropy, labels, entropy_threshold, mask_data):
    # Only predictions if the entropy is less than the given threshold
    if len(labels.shape) >1:
      labels = labels.flatten()
    predictions_probabilities = prediction_probabilities[mask_data]
    labels = labels[mask_data]
    entropy = entropy_fn(prediction_probabilities)
    entropy = entropy[mask_data]
    mask = (entropy <= entropy_threshold) # True if the entropy of the prediction is less or equal to the threshold
    predictions_bellow_entropy = predictions_probabilities[mask].to('cpu')
    labels = labels[mask].to('cpu') 
    _, pred = predictions_bellow_entropy.max(dim=1)
    correct = float(pred.eq(labels).sum().item())
    num_of_predictions = labels.size(0)
    if num_of_predictions > 0:
        acc = correct / num_of_predictions
    else:
        acc = None  # to avoid division by zero
    return acc, num_of_predictions

def accuracy_with_rejection_percentage(prediction_probabilities, labels, rejection_percentage, mask_data):
    if len(labels.shape) > 1:
        labels = labels.flatten()
    predictions_probabilities = prediction_probabilities[mask_data]
    labels = labels[mask_data]

    # Calculate entropy for each prediction
    entropy = entropy_fn(predictions_probabilities)

    # Calculate the number of predictions to reject and accept based on the rejection percentage
    num_to_reject = int(len(entropy) * rejection_percentage)
    num_to_accept = len(entropy) - num_to_reject

    # Sort predictions by entropy and select the subset based on the rejection percentage
    _, indices = torch.sort(entropy)
    accepted_indices = indices[:num_to_accept]

    # Filter predictions and labels based on the accepted indices
    accepted_predictions = predictions_probabilities[accepted_indices]
    accepted_labels = labels[accepted_indices]

    _, pred = accepted_predictions.max(dim=1)
    correct = pred.eq(accepted_labels).sum().item()
    num_of_accepted_predictions = accepted_labels.size(0)

    if num_of_accepted_predictions > 0:
        acc = correct / num_of_accepted_predictions
    else:
        acc = None  # to avoid division by zero due to high rejection rate

    return acc, num_of_accepted_predictions

# def evaluate_stochastic_model(model, data, feat, n_samples=1, entropy_threshold=np.inf, pos_encoding=None):
#     model.eval()
#     with torch.no_grad():
#       _, samples_mean, samples_entropy = mc_sample_stats(model, feat, n_samples=n_samples, pos_encoding=pos_encoding)
#       accs = []
#       for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         acc, _ = accuracy_with_entropy_threshold(samples_mean, samples_entropy, data.y, entropy_threshold=entropy_threshold, mask_data=mask)
#         accs.append(acc)
#     return accs

def evaluate_stochastic_model_with_reject(model, data, feat, n_samples=1, rejection_percentage=0, pos_encoding=None):
    model.eval()
    with torch.no_grad():
      _, samples_mean, _ = mc_sample_stats(model, feat, n_samples=n_samples, pos_encoding=pos_encoding)
      accs = []
      for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc, _ = accuracy_with_rejection_percentage(samples_mean, data.y, rejection_percentage=rejection_percentage, mask_data=mask)
        accs.append(acc)
    return accs
  
""" End: Evaluate with Entropy """

  

# @torch.no_grad()
# def test(model, data, pos_encoding=None, opt=None, entropy_threshold=np.inf, n_samples=1):  # opt required for runtime polymorphism
#   model.eval()
#   feat = data.x
#   if model.opt['use_labels']:
#     feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    
#   accs = evaluate_stochastic_model(model, data, feat, n_samples=n_samples, entropy_threshold=entropy_threshold, pos_encoding=pos_encoding)
#   return accs

@torch.no_grad()
def test_with_reject(model, data, pos_encoding=None, opt=None, rejection_percentage=0, n_samples=1):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    
  accs = evaluate_stochastic_model_with_reject(model, data, feat, n_samples=n_samples, rejection_percentage=rejection_percentage, pos_encoding=pos_encoding)
  return accs



def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


# @torch.no_grad()
# def test_OGB(model, data, pos_encoding, opt, entropy_threshold=np.inf, n_samples=1):
#   if opt['dataset'] == 'ogbn-arxiv':
#     name = 'ogbn-arxiv'

#   feat = data.x
#   if model.opt['use_labels']:
#     feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)

#   train_acc, valid_acc, test_acc = evaluate_stochastic_model(model, data, feat, n_samples=n_samples, entropy_threshold=entropy_threshold, pos_encoding=pos_encoding)

#   return train_acc, valid_acc, test_acc


def merge_cmd_args(cmd_opt, opt):
  if cmd_opt['beltrami']:
    opt['beltrami'] = True
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['attention_type'] != 'scaled_dot':
    opt['attention_type'] = cmd_opt['attention_type']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] != 1:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] != 100:
    opt['epoch'] = cmd_opt['epoch']
  if not cmd_opt['not_lcc']:
    opt['not_lcc'] = False
  if cmd_opt['num_splits'] != 1:
    opt['num_splits'] = cmd_opt['num_splits']

# New
def release_model_memory(model):
    """
    Releases the memory occupied by a PyTorch model.

    Parameters:
    model (torch.nn.Module): The PyTorch model to be released from memory.
    """
    # Move the model to CPU
    model.to('cpu')
    
    # Delete the model
    del model

    # Clear any cached memory if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Explicitly run garbage collection
    gc.collect()
    
# New
def create_results_folder(data_name):
    folder_name = f'results/{data_name.lower()}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder created: {folder_name}")
    else:
        print(f"Folder already exists: {folder_name}")

# New
def save_results(all_accuracies, model_name, results_path, task_index):
    filename = f'{results_path}/accuracies_{model_name}_task_index{task_index}.json'

    # Check if the file already exists
    counter = 1
    while os.path.exists(filename):
        filename = f'{results_path}/accuracies_{model_name}_task_index{task_index}_(%d).json' % counter
        counter += 1

    with open(filename, 'w') as fp:
        json.dump(all_accuracies, fp, indent=4)
        
        
def main(cmd_opt):
  try:
    best_opt = best_params_dict[cmd_opt['dataset']]
    opt = {**cmd_opt, **best_opt}
    merge_cmd_args(cmd_opt, opt)
  except KeyError:
    opt = cmd_opt

  dataset = get_dataset(opt, f'{ROOT_DIR}/data', opt['not_lcc'])
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if opt['beltrami']:
    pos_encoding = apply_beltrami(dataset.data, opt).to(device)
    opt['pos_enc_dim'] = pos_encoding.shape[1]
  else:
    pos_encoding = None

  model = GNSDE(opt, dataset, device).to(device) if opt["no_early"] else GNSDEEarly(opt, dataset, device).to(device)

  if not opt['planetoid_split'] and opt['dataset'] in ['Cora','Citeseer','Pubmed']:
    dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data, num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

  data = dataset.data.to(device)

  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_time = best_epoch = train_acc = val_acc = test_acc = 0

  this_test = test_with_reject if opt['dataset'] == 'ogbn-arxiv' else test_with_reject
  this_train = train if opt['dataset'] == 'ogbn-arxiv' else train


  
  best_model = None
  for epoch in range(1, opt['epoch']):
    start_time = time.time()

    if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
      ei = apply_KNN(data, pos_encoding, model, opt)
      model.odeblock.odefunc.edge_index = ei

    loss = this_train(model, optimizer, data, pos_encoding)
    tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)

    best_time = opt['time']
    if tmp_val_acc > val_acc:
      best_epoch = epoch
      train_acc = tmp_train_acc
      val_acc = tmp_val_acc
      test_acc = tmp_test_acc
      best_time = opt['time']
      if best_model != None:
        release_model_memory(best_model)
      best_model = copy.deepcopy(model)


                     
    if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc:
      best_epoch = epoch
      val_acc = model.odeblock.test_integrator.solver.best_val
      test_acc = model.odeblock.test_integrator.solver.best_test
      train_acc = model.odeblock.test_integrator.solver.best_train
      best_time = model.odeblock.test_integrator.solver.best_time
      if best_model != None:
        release_model_memory(best_model)
      best_model = copy.deepcopy(model)

    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'

    print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, val_acc, test_acc, best_time))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(val_acc, test_acc,
                                                                                                     best_epoch,
                                                                                                     best_time))
  release_model_memory(model)
  # Entropy Threshold Evaluation
  best_model.odeblock.t[1] = best_time
  print('best_model.odeblock.t', best_model.odeblock.t)
  for rejection_percentage in rejection_percentages:
    # print(
    #   '~' * 10 + f' Model: GNSDE-GRAND, Entropy Threshold: {entropy_threshold}' + '~' * 10)
    _, _, test_acc = this_test(best_model, data, pos_encoding, opt, rejection_percentage, n_samples=50)
    all_accuracies[str(rejection_percentage)].append(test_acc)
    print(f'rejection_percentage: {rejection_percentage}', 'Test accuracy:', test_acc)                
  

  return all_accuracies


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--use_80_10_10_split', action='store_true',
                    help='Uses a training-val-test split of 80, 10, 10, respectively.')
    
  parser.add_argument('--number_of_experiments', type=int, default=1, help='Number experiments to run')
  parser.add_argument('--save_results', action='store_true', help='Saves the results of the experiments')
   
  parser.add_argument('--use_stochastic_grand', action='store_true',
                    help='Enable the use of a Latent SDE for diffusion message passing instead of a traditional Neural ODE in the GRAND model.')
  
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  # data args
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                      help='use the 10 fixed splits from '
                           'https://arxiv.org/abs/2002.05287')
  parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                      help='the number of splits to repeat the results on')
  parser.add_argument('--label_rate', type=float, default=0.5,
                      help='% of training labels to use when --use_labels is set.')
  parser.add_argument('--planetoid_split', action='store_true',
                      help='use planetoid splits for Cora/Citeseer/Pubmed')
  # GNN args
  parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
  parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                      help='Add a fully connected layer to the decoder.')
  parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
  parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
  parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
  parser.add_argument('--epoch', type=int, default=2, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                      help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
  parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
  parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                      help='Add a fully connected layer to the encoder.')
  parser.add_argument('--add_source', dest='add_source', action='store_true',
                      help='If try get rid of alpha param and the beta*x0 source term')
  parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')

  # ODE args
  parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--step_size', type=float, default=1,
                      help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
  parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                      help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
  parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                      help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--adjoint_step_size', type=float, default=1,
                      help='fixed step size when using fixed step adjoint solvers e.g. rk4')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument("--max_nfe", type=int, default=1000,
                      help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
  parser.add_argument("--no_early", action="store_true",
                      help="Whether or not to use early stopping of the ODE integrator when testing.")
  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
  parser.add_argument("--max_test_steps", type=int, default=100,
                      help="Maximum number steps for the dopri5Early test integrator. "
                           "used if getting OOM errors at test time")

  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                      help="multiply attention scores by edge weights before softmax")
  parser.add_argument('--attention_type', type=str, default="scaled_dot",
                      help="scaled_dot,cosine_sim,pearson, exp_kernel")
  parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

  # regularisation args
  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

  # rewiring args
  parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
  parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
  parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
  parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                      help="obove this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
  parser.add_argument('--att_samp_pct', type=float, default=1,
                      help="float in [0,1). The percentage of edges to retain based on attention scores")
  parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                      help='incorporate the feature grad in attention based edge dropout')
  parser.add_argument("--exact", action="store_true",
                      help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
  parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
  parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
  parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
  parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
  parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
  parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
  parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
  parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
  parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
  # beltrami args
  parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
  parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
  parser.add_argument('--pos_enc_type', type=str, default="DW64",
                      help='positional encoder either GDC, DW64, DW128, DW256')
  parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
  parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
  parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
  parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
  parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
  parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                      help="random, ,anchored, importance, degree")
  parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
  parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--edge_sampling_space', type=str, default="attention",
                      help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
  parser.add_argument('--symmetric_attention', action='store_true',
                      help='maks the attention symmetric for rewring in QK space')

  parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
  parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
  parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

  parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")
  parser.add_argument('--task_index', type=str, default=0, help="run index for parrallel runs")


  args = parser.parse_args()

  opt = vars(args)
  
  
  best_model = None
  rejection_percentages = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
  all_accuracies = {str(entropy_threshold): [] for entropy_threshold in rejection_percentages}
  
  model_sde_name = 'GRAND-SDE' 
  if opt['planetoid_split'] or opt['use_80_10_10_split']:
    model_sde_name = 'GNSDE-SDE_planetoid_split'
    print("model_sde_name", model_sde_name)
  dataset_name = opt['dataset']
  task_index = opt['task_index']
  entropy_results_path = f"results/{dataset_name.lower()}" 
  create_results_folder(dataset_name)
  
  # for number of experiments: 
  num_experiments = 1
  for i in range(num_experiments):
    print("experiment:", i)
    all_accuracies = main(opt)
  save_results(all_accuracies, model_sde_name, entropy_results_path, task_index)   
