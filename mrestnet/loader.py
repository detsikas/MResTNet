import torch
import numpy as np


def load_conv2d(state_dict, weights, state_name, weights_name):
    state_dict[f'{state_name}.weight'].copy_(
        torch.from_numpy(weights[f'{weights_name}/kernel']).permute(3, 2, 0, 1))
    del weights[f'{weights_name}/kernel']
    state_dict[f'{state_name}.bias'].copy_(
        torch.from_numpy(weights[f'{weights_name}/bias']))
    del weights[f'{weights_name}/bias']


def load_parameter(state_dict, weights, state_name, weights_name):
    state_dict[state_name].copy_(torch.from_numpy(weights[weights_name]))
    del weights[weights_name]


def load_parameter_with_func(state_dict, weights, state_name, weights_name, func):
    state_dict[state_name].copy_(torch.from_numpy(func(weights[weights_name])))
    del weights[weights_name]


def load_layer_norm(state_dict, weights, state_name, weights_name):
    state_dict[f'{state_name}.weight'].copy_(
        torch.from_numpy(weights[f'{weights_name}/scale']))
    del weights[f'{weights_name}/scale']
    state_dict[f'{state_name}.bias'].copy_(
        torch.from_numpy(weights[f'{weights_name}/bias']))
    del weights[f'{weights_name}/bias']


def load_multihead_self_attention(state_dict, weights, state_name, weights_name):
    query_kernel = weights[f'{weights_name}/query/kernel']
    query_kernel = query_kernel.reshape(
        query_kernel.shape[0], -1).transpose()
    key_kernel = weights[f'{weights_name}/key/kernel']
    key_kernel = key_kernel.reshape(key_kernel.shape[0], -1).transpose()
    value_kernel = weights[f'{weights_name}/value/kernel']
    value_kernel = value_kernel.reshape(
        value_kernel.shape[0], -1).transpose()
    in_kernel = np.concatenate((query_kernel, key_kernel, value_kernel))
    state_dict[f'{state_name}.in_proj_weight'].copy_(
        torch.from_numpy(in_kernel))
    del weights[f'{weights_name}/query/kernel']
    del weights[f'{weights_name}/key/kernel']
    del weights[f'{weights_name}/value/kernel']

    query_bias = weights[f'{weights_name}/query/bias'].flatten()
    key_bias = weights[f'{weights_name}/key/bias'].flatten()
    value_bias = weights[f'{weights_name}/value/bias'].flatten()
    in_kernel = np.concatenate((query_bias, key_bias, value_bias))
    state_dict[f'{state_name}.in_proj_bias'].copy_(
        torch.from_numpy(in_kernel))
    del weights[f'{weights_name}/query/bias']
    del weights[f'{weights_name}/key/bias']
    del weights[f'{weights_name}/value/bias']

    out_kernel = weights[f'{weights_name}/out/kernel']
    out_kernel = out_kernel.reshape(-1, out_kernel.shape[-1]).transpose()
    state_dict[f'{state_name}.out_proj.weight'].copy_(
        torch.from_numpy(out_kernel))
    del weights[f'{weights_name}/out/kernel']

    out_bias = weights[f'{weights_name}/out/bias']
    state_dict[f'{state_name}.out_proj.bias'].copy_(
        torch.from_numpy(out_bias))
    del weights[f'{weights_name}/out/bias']


def load_dense(state_dict, weights, state_name, weights_name):
    state_dict[f'{state_name}.weight'].copy_(
        torch.from_numpy(weights[f'{weights_name}/kernel']).permute(1, 0))
    del weights[f'{weights_name}/kernel']
    state_dict[f'{state_name}.bias'].copy_(
        torch.from_numpy(weights[f'{weights_name}/bias']))
    del weights[f'{weights_name}/bias']
