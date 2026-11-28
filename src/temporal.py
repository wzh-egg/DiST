"""Same causal kernel-3 convolution, evaluated by window packing + linear.

Weights and biases remain the original Conv1d parameters. No new parameters,
no altered receptive field, no disabled deterministic convolution setting.
"""
import torch.nn.functional as F


def causal_conv3(x, conv):
    if (conv.kernel_size != (3,) or conv.stride != (1,) or
            conv.padding != (0,) or conv.groups != 1):
        raise ValueError('This implementation requires unpadded kernel-3 Conv1d.')
    dilation = conv.dilation[0]
    padded = F.pad(x, (2 * dilation, 0))
    # (sequences, channels, time, taps), taps at 0,d,2d.
    windows = padded.unfold(2, 2 * dilation + 1, 1)[..., ::dilation]
    packed = windows.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
    weight = conv.weight.reshape(conv.out_channels, -1)
    return F.linear(packed, weight, conv.bias).transpose(1, 2)
