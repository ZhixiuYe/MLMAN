import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = Variable(torch.arange(0, len(sequence_lengths)).long()).cuda()

    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
    # if input_lstm.bidirectional:
    #     for ind in range(0, input_lstm.num_layers):
    #         weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
    #         bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    #         nn.init.uniform(weight, -bias, bias)
    #         weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
    #         bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    #         nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        # if input_lstm.bidirectional:
        #     for ind in range(0, input_lstm.num_layers):
        #         weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
        #         weight.data.zero_()
        #         weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        #         weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
        #         weight.data.zero_()
        #         weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

def init_cnn(input_cnn):
    n = input_cnn.in_channels
    for k in input_cnn.kernel_size:
        n *= k
    stdv = np.sqrt(6./n)
    input_cnn.weight.data.uniform_(-stdv, stdv)
    if input_cnn.bias is not None:
        input_cnn.bias.data.uniform_(-stdv, stdv)
