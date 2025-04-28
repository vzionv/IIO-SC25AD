import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.optim import Optimizer, SGD, Adam
import copy
from torch.nn import init
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
import types
import torch.fx as fx
from torch.nn import Parameter
from warnings import warn

a_bits = 8
w_bits = 8


def norm(x):
    min_val = x.min()
    max_val = x.max()
    return torch.floor((x - min_val) / (max_val - min_val) * 255)


def re_shape(x):
    ndim = x.ndim
    if ndim < 2:
        raise ValueError("Expected at least 2D input.")
    indices = [0] * (len(x.shape) - 2) + [slice(None), slice(None)]
    x = x[indices]
    h, w = x.shape
    ratio = w / h
    if ratio >= 4:
        m = int(np.sqrt(ratio))
        x = x[:, :w // m * m]
        x = torch.cat(torch.chunk(x, m, dim=1), dim=0)
        return x
    elif ratio <= 1 / 4:
        m = int(np.sqrt(1 / ratio))
        x = x[:h // m * m, :]
        x = torch.cat(torch.chunk(x, m, dim=0), dim=1)
        return x
    else:
        return x


def log_wanda(a, qa, w, qw, **kwargs):
    global step
    a = re_shape(a)
    qa = re_shape(qa)
    w = re_shape(w)
    qw = re_shape(qw)
    tag = kwargs.get("tag", "")
    for ax in axes.flatten():
        ax.cla()
    im = axes[0, 0].imshow(a.detach().numpy(), cmap="rainbow")
    fig.colorbar(im, cax=axes[1, 0], orientation='horizontal')
    im = axes[0, 1].imshow(qa.detach().numpy(), cmap="rainbow")
    fig.colorbar(im, cax=axes[1, 1], orientation='horizontal')
    im = axes[0, 2].imshow((a - qa).detach().numpy(), cmap="rainbow")
    fig.colorbar(im, cax=axes[1, 2], orientation='horizontal')
    writer.add_figure(tag=tag + "_a", figure=fig, global_step=step)
    for ax in axes.flatten():
        ax.cla()
    im = axes[0, 0].imshow(w.detach().numpy(), cmap="rainbow")
    fig.colorbar(im, cax=axes[1, 0], orientation='horizontal')
    im = axes[0, 1].imshow(qw.detach().numpy(), cmap="rainbow")
    fig.colorbar(im, cax=axes[1, 1], orientation='horizontal')
    im = axes[0, 2].imshow((w - qw).detach().numpy(), cmap="rainbow")
    fig.colorbar(im, cax=axes[1, 2], orientation='horizontal')
    writer.add_figure(tag=tag + "_w", figure=fig, global_step=step)
    for ax in axes.flatten():
        ax.cla()


def log_grad(g, **kwargs):
    global step
    g = re_shape(g)
    # g = norm(g)
    # if g.ndim == 2:
    #     g = g.unsqueeze(0)
    tag = kwargs.get("tag", "")
    for ax in axes.flatten():
        ax.cla()
    im = axes[0, 0].imshow(g.detach().numpy(), cmap="rainbow")
    fig.colorbar(im, cax=axes[1, 0], orientation='horizontal')
    # writer.add_image(tag=tag + "_g", img_tensor=g.detach(), global_step=step)
    writer.add_figure(tag=tag + "_g", figure=fig, global_step=step)
    for ax in axes.flatten():
        ax.cla()


class Round(Function):
    """
    This function is used to round a float or fixed tensor
    """

    @staticmethod
    def forward(self, input):
        """
        round forward method
        """
        sign = torch.sign(input)
        # except Exception as e:
        #     print(e)
        #     print(input.device)
        #     print(input.dtype)
        #     exit(0)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        """
        using STE as gradient backward method
        """
        grad_input = grad_output.clone()
        return grad_input


class DSRound(nn.Module):
    def __init__(self, init_alpha=0.002):
        super().__init__()
        self.dsq_alpha = nn.Parameter(torch.tensor(init_alpha, requires_grad=True))

    def forward(self, x: torch.Tensor):
        tanh_scale = 1 / (1 - self.dsq_alpha**2)
        tanh_k = torch.log((tanh_scale + 1) / (tanh_scale - 1))
        delta = (tanh_scale * torch.tanh(tanh_k * (x - x.floor() - 0.5))) * 0.5 + 0.5
        x = x.floor() + delta
        x = (x.round() - x).detach() + x
        return x, delta


class ControlledParameter(nn.Parameter):
    def __init__(self, *args, **kwargs):
        super(ControlledParameter, self).__init__()
        self._frozen = False
        self._stored_value = self.data.detach().clone()

    @property
    def freeze(self):
        return self._frozen

    @freeze.setter
    def freeze(self, freeze_or_not):
        self._frozen = freeze_or_not
        self._stored_value = self.data.detach().clone()
        # if freeze_or_not:
        #     self.beta.requires_grad = False
        # else:
        #     self.beta.requires_grad = True

    def restore(self):
        with torch.no_grad():
            super(ControlledParameter, self).data = self._stored_value

    # def __getattribute__(self, name):
    #     print(name)
    #     if "add" in name or "sub" in name or "mul" in name or "div" in name or "pow" in name:
    #         if self._frozen:
    #             return lambda *args, **kwargs: self
    #         else:
    #             return super(ControlledParameter, self).__getattribute__(name)
    #     else:
    #         return super(ControlledParameter, self).__getattribute__(name)

    def __getattribute__(self, name):
        if name == "grad":
            # print(self._frozen)
            if self._frozen:
                return None
            else:
                return super(ControlledParameter, self).__getattribute__(name)
        else:
            return super(ControlledParameter, self).__getattribute__(name)


#  Activation Quantification
class ActivationQuantizer(nn.Module):
    """
     Activation here should be considered as the input feature map of each layer or the output feature map of the previous layer
    """

    def __init__(self, a_bits, pre_w_bits=w_bits, version=2, init_beta=None):
        super(ActivationQuantizer, self).__init__()
        self._freeze_beta = False
        self.a_bits = a_bits
        self.pre_w_bits = pre_w_bits
        if init_beta is None:
            init_beta = pre_w_bits / 2
        self.beta = ControlledParameter(torch.tensor(init_beta))
        # self.alpha = 1
        # self.register_parameter("beta", beta)
        # self.beta_error = torch.tensor(0., requires_grad=False)
        self._version = version

    #  rounding a float or fixed tensor to integer one(STE)
    def round(self, input):
        output = Round.apply(input)
        return output

    @property
    def scale(self):
        return float(2 ** self.a_bits - 1)

    @property
    def freeze_beta(self):
        return self._freeze_beta

    @freeze_beta.setter
    def freeze_beta(self, freeze_or_not):
        self._freeze_beta = freeze_or_not
        self.beta.freeze = freeze_or_not
        # if freeze_or_not:
        #     self.beta.requires_grad = False
        # else:
        #     self.beta.requires_grad = True

    # quantization / dequantization with alpha
    def forward(self, input):
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:
            raise AssertionError("! Binary quantization is not supported !")
        else:
            if self._version == 1:
                beta_int = Round.apply(torch.clamp(self.beta, 0, self.pre_w_bits))
            elif self._version == 2:
                if self.freeze_beta:
                    # beta_int = Round.apply(torch.clamp(self.beta, -self.pre_w_bits, self.pre_w_bits))
                    beta_int = Round.apply(torch.clamp(self.beta, 0, self.pre_w_bits))
                else:
                    # beta_int = torch.clamp(self.beta, -self.pre_w_bits, self.pre_w_bits)
                    beta_int = torch.clamp(self.beta, 0, self.pre_w_bits)
            else:
                raise ValueError("version should be 1 or 2")
            # self.beta_error = ((self.beta - beta_int) ** 3)
            # self.beta_error = (self.beta - beta_int) ** 3
            alpha = (2 ** self.pre_w_bits - 1) / (2 ** (self.pre_w_bits + beta_int))
            output = torch.clamp(input * alpha, 0, 1)
            scale = self.scale
            output = self.round(output * scale) / scale
        self.beta_error = torch.mean((output - input) ** 2)
        return output

    def forward_copy(self, input):
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:
            raise AssertionError("! Binary quantization is not supported !")
        else:
            alpha = (2 ** self.pre_w_bits - 1) / (2 ** (self.pre_w_bits + Round.apply(F.relu(self.beta))))
            # alpha = self.alpha
            output = torch.clamp(input * alpha, 0, 1)
            
            scale = self.scale  # scale
            output = self.round(output * scale) / scale
        return output

    def inference(self, input):
        # global w_bits
        input = Round.apply(
            input / (2 ** (self.pre_w_bits + Round.apply(F.relu(self.beta))))) 
        input = torch.clamp(input, 0, self.scale)
        return input


class SignedQuantizer(ActivationQuantizer):
    def __init__(self, a_bits, pre_w_bits=w_bits, version=2):
        super().__init__(a_bits, pre_w_bits, version)

    def forward(self, input):
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:
            raise AssertionError("! Binary quantization is not supported !")
        else:
            if self._version == 1:
                beta_int = Round.apply(torch.clamp(self.beta, 0, self.pre_w_bits))
            elif self._version == 2:
                if self.freeze_beta:
                    # beta_int = Round.apply(torch.clamp(self.beta, -self.pre_w_bits, self.pre_w_bits))
                    beta_int = Round.apply(torch.clamp(self.beta, 0, self.pre_w_bits))
                else:
                    # beta_int = torch.clamp(self.beta, -self.pre_w_bits, self.pre_w_bits)
                    beta_int = torch.clamp(self.beta, 0, self.pre_w_bits)
            else:
                raise ValueError("version should be 1 or 2")
            input_sign = torch.sign(input)
            abs_input = torch.abs(input)
            alpha = (2 ** self.pre_w_bits - 1) / (2 ** (self.pre_w_bits + beta_int))
            output = torch.clamp(abs_input * alpha, 0,
                                 1)
            scale = self.scale  # scale
            output = self.round(output * scale) / scale
            output = input_sign * (2 ** beta_int) * output
        self.beta_error = torch.mean((output - input) ** 2)
        return output


# Weight Quantification
class WeightQuantizer(nn.Module):
    def __init__(self, w_bits, version=2, theta_init=2.0, norm=True):
        super(WeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self._version = version
        the = ControlledParameter(torch.tensor(theta_init))
        self.register_parameter("the", the)
        self.norm = norm

    def sharpness(self, x):
        ex = torch.exp(-1 * F.relu(self.the + 1E-8) * x)
        result = torch.div(1 - ex, 1 + ex)
        return result

    # Rounding a float or fixed tensor to integer one(STE)
    def round(self, input):
        output = Round.apply(input)
        return output

    @property
    def scale(self):
        return float(2 ** self.w_bits - 1)

    # quantization / dequantization without alpha
    def forward(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            raise AssertionError("! Binary quantization is not supported !")
        else:

            output = self.sharpness(input)

            if self._version == 1:
                output = output / 2 + 0.5
            elif self._version == 2:
                if not self.norm:
                    output = output / 2 + 0.5
                else:
                    output = output / 2 / torch.max(torch.abs(output)) + 0.5
            else:
                raise ValueError("vserion should be 1 or 2")
            scale = self.scale  # scale
            output = self.round(output * scale) / scale
            output = 2 * output - 1
        return output

    def forward_copy(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            raise AssertionError("! Binary quantization is not supported !")
        else:
            output = torch.tanh(input)
            output = output / 2 / (torch.max(torch.abs(output)) + 1E-8) + 0.5
            scale = self.scale  # scale
            output = self.round(output * scale) / scale
            output = 2 * output - 1
        return output

    def inference(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            raise AssertionError("! Binary quantization is not supported !")
        else:
            output = torch.tanh(input)
            if self._version == 1:
                output = output / 2 / torch.max(torch.abs(output)) + 0.5
            elif self._version == 2:
                output = output / 2 + 0.5
            else:
                raise ValueError("vserion should be 1 or 2")

            scale = self.scale
            output = self.round(output * scale)
        return output


class ErrorStatic:
    def __init__(self):
        self._active_error = []
        self._active_error_per_q = []
        self._active_error_per_i = []
        self._weight_error = []
        self._weight_error_per_q = []
        self._weight_error_per_i = []

    def update(self, errors, aorw="aw"):
        if aorw == "a":
            self._active_error.append(errors[0])
            self._active_error_per_q.append(errors[1])
            self._active_error_per_i.append(errors[2])
        elif aorw == "w":
            self._weight_error.append(errors[0])
            self._weight_error_per_q.append(errors[1])
            self._weight_error_per_i.append(errors[2])
        else:
            self._active_error.append(errors[0][0])
            self._active_error_per_q.append(errors[0][1])
            self._active_error_per_i.append(errors[0][2])
            self._weight_error.append(errors[1][0])
            self._weight_error_per_q.append(errors[1][1])
            self._weight_error_per_i.append(errors[1][2])

    def clear(self):
        self._active_error.clear()
        self._active_error_per_q.clear()
        self._active_error_per_i.clear()
        self._weight_error.clear()
        self._weight_error_per_q.clear()
        self._weight_error_per_i.clear()

    def cal_avg_error(self, layer_name=""):
        ae = torch.mean(torch.hstack(self._active_error)).item()
        aeq = torch.mean(torch.hstack(self._active_error_per_q)).item()
        aei = torch.mean(torch.hstack(self._active_error_per_i)).item()
        we = torch.mean(torch.hstack(self._weight_error)).item()
        weq = torch.mean(torch.hstack(self._weight_error_per_q)).item()
        wei = torch.mean(torch.hstack(self._weight_error_per_i)).item()
        print(f"Layer {layer_name}: active error: {ae:.5f}-{aeq:.2f}%q-{aei:.2f}%i")
        print(f"Layer {layer_name}: weight error: {we:.5f}-{weq:.2f}%q-{wei:.2f}%i")
        self.clear()
        return ae, we


class ConvBn(nn.Module):
    def __init__(self, conv, bn):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class QuantIdentity(nn.Identity):
    def __init__(self, quant_inference=False):
        super(QuantIdentity, self).__init__()
        self.quant_inference = quant_inference

    def inference(self, input):
        return input


class QuantConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
            a_bits=8,
            w_bits=8,
            quant_inference=False,
            writer=None,
            name="",
            first_layer=False
    ):
        super(QuantConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_inference = quant_inference
        if first_layer:
            self.activation_quantizer = SignedQuantizer(a_bits=a_bits)
        else:
            self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)
        # self.e = ErrorStatic()
        self.writer = writer
        self.name = name
        self.recording = False
        self.record_epoch = 0
        self.rank = 0

    @property
    def w_bits(self):
        return self.weight_quantizer.w_bits

    @property
    def a_bits(self):
        return self.activation_quantizer.a_bits

    def forward(self, input):
        quant_input = self.activation_quantizer(input)  # from float to fixed, quant_input≈alpha*input

        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)  # from float to fixed, quant_weight≈weight
        else:
            quant_weight = self.weight

        # quant_weight = self.weight
        if self.recording and self.rank == 0:
            self.writer.add_histogram(f'{self.name}_weight', self.weight, self.record_epoch)
            self.writer.add_histogram(f'{self.name}_quant_weight', quant_weight, self.record_epoch)

        # self.e.update((cal_error(quant_input, input), cal_error(quant_weight, self.weight)))
        output = F.conv2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def _inference(self, input):
        warn("QuantConv2d._inference is deprecated and will be removed in the future.", DeprecationWarning)
        input = self.activation_quantizer.inference(input)

        w_bits = self.weight_quantizer.w_bits
        int_weight = self.weight_quantizer.inference(
            self.weight)  # if weight is already int, please comment this line
        x_next = 2 * F.conv2d(input, int_weight, self.bias, self.stride, self.padding, self.dilation, self.groups) - F.conv2d(
            input, torch.ones_like(int_weight, requires_grad=False), self.bias, self.stride, self.padding, self.dilation,
            self.groups) * (2 ** w_bits - 1)

        return x_next

    def inference(self, input):
        # ! self.activation_quantizer.forward should be reflected onto one's inference
        # If this is not realized, please using self._inference instead
        # the same as self.weight_quantizer
        input = self.activation_quantizer(input)

        w_bits = self.weight_quantizer.w_bits
        int_weight = self.weight_quantizer(
            self.weight)  # if weight is already int, please comment this line
        x_next = 2 * F.conv2d(input, int_weight, self.bias, self.stride, self.padding, self.dilation, self.groups) - F.conv2d(
            input, torch.ones_like(int_weight, requires_grad=False), self.bias, self.stride, self.padding, self.dilation,
            self.groups) * (2 ** w_bits - 1)

        return x_next

    def get_quanted_params(self):
        int_weight = self.weight_quantizer.inference(self.weight)
        int_weight = int_weight.to(determin_storage_type(int_weight))
        return {"int_weight": int_weight}


class QuantConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
            a_bits=8,
            w_bits=8,
            quant_inference=False,
    ):
        super(QuantConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_inference = quant_inference
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)

    @property
    def w_bits(self):
        return self.weight_quantizer.w_bits

    @property
    def a_bits(self):
        return self.activation_quantizer.a_bits

    def forward(self, input, **kwargs):
        quant_input = self.activation_quantizer(input)  # from float to fixed, quant_input≈alpha*input

        # quant_input = input
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)  # from float to fixed, quant_weight≈weight
        else:
            quant_weight = self.weight
        # quant_weight = self.weight

        # with torch.no_grad():
        #     log_wanda(input[0, 0], quant_input[0, 0], self.weight[0, 0], quant_weight[0, 0], **kwargs)

        output = F.conv1d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def _inference(self, input):
        warn("QuantConv1d._inference is deprecated and will be removed in the future.", DeprecationWarning)
        input = self.activation_quantizer.inference(input)

        w_bits = self.weight_quantizer.w_bits
        int_weight = self.weight_quantizer.inference(
            self.weight)  # if weight is already int, please comment this line
        x_next = 2 * F.conv1d(input, int_weight, self.bias, self.stride, self.padding, self.dilation, self.groups) - F.conv1d(
            input, torch.ones_like(int_weight, requires_grad=False), self.bias, self.stride, self.padding, self.dilation,
            self.groups) * (2 ** w_bits - 1)

        return x_next

    def inference(self, input):
        # ! self.activation_quantizer.forward should be reflected onto one's inference
        # If this is not realized, please using self._inference instead
        # the same as self.weight_quantizer
        input = self.activation_quantizer(input)

        w_bits = self.weight_quantizer.w_bits
        int_weight = self.weight_quantizer(
            self.weight)  # if weight is already int, please comment this line
        x_next = 2 * F.conv1d(input, int_weight, self.bias, self.stride, self.padding, self.dilation, self.groups) - F.conv1d(
            input, torch.ones_like(int_weight, requires_grad=False), self.bias, self.stride, self.padding, self.dilation,
            self.groups) * (2 ** w_bits - 1)

        return x_next

    def get_quanted_params(self):
        int_weight = self.weight_quantizer.inference(self.weight)
        int_weight = int_weight.to(determin_storage_type(int_weight))
        return {"int_weight": int_weight}


class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
            a_bits=8,
            w_bits=8,
            quant_inference=False,
    ):
        super(QuantConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_inference = quant_inference
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)

    @property
    def w_bits(self):
        return self.weight_quantizer.w_bits

    @property
    def a_bits(self):
        return self.activation_quantizer.a_bits

    def forward(self, input, output_size=None):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        output = F.conv_transpose2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )
        return output

    def _inference(self, input):
        warn("QuantConvTranspose2d._inference is deprecated and will be removed in the future.", DeprecationWarning)
        input = self.activation_quantizer.inference(input)

        w_bits = self.weight_quantizer.w_bits
        int_weight = self.weight_quantizer.inference(
            self.weight)  # if weight is already int, please comment this line
        x_next = 2 * F.conv_transpose2d(input, int_weight, self.bias, self.stride, self.padding, self.dilation,
                                        self.groups) - F.conv_transpose2d(
            input, torch.ones_like(int_weight, requires_grad=False), self.bias, self.stride, self.padding, self.dilation,
            self.groups) * (2 ** w_bits - 1)

        return x_next

    def inference(self, input):
        # ! self.activation_quantizer.forward should be reflected onto one's inference
        # If this is not realized, please using self._inference instead
        # the same as self.weight_quantizer
        input = self.activation_quantizer(input)

        w_bits = self.weight_quantizer.w_bits
        int_weight = self.weight_quantizer(
            self.weight)  # if weight is already int, please comment this line
        x_next = 2 * F.conv_transpose2d(input, int_weight, self.bias, self.stride, self.padding, self.dilation,
                                        self.groups) - F.conv_transpose2d(input, torch.ones_like(int_weight, requires_grad=False),
                                                                          self.bias, self.stride, self.padding, self.dilation,
                                                                          self.groups) * (2 ** w_bits - 1)

        return x_next

    def get_quanted_params(self):
        int_weight = self.weight_quantizer.inference(self.weight)
        int_weight = int_weight.to(determin_storage_type(int_weight))
        return {"int_weight": int_weight}


class QuantReLU6(nn.Module):
    def __init__(self, a_bits, w_bits, inplace=True, quant_inference=False):
        super().__init__()
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.inplace = inplace
        self.quant_inference = quant_inference

    @property
    def scale(self):
        return (2 ** self.a_bits) * (2 ** self.w_bits)

    def forward(self, input):
        return F.relu6(input, self.inplace)

    def inference(self, input):
        return torch.clamp(input, 0, self.scale * 6)


class QuantMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,
                 quant_inference=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.quant_inference = quant_inference

    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)

    def inference(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)


class QuantMaxPool1d(nn.MaxPool1d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,
                 quant_inference=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.quant_inference = quant_inference

    def forward(self, input):
        return F.max_pool1d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)

    def inference(self, input):
        return F.max_pool1d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)


class QuantAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, quant_inference=False):
        super().__init__(kernel_size, stride, padding, ceil_mode)
        self.quant_inference = quant_inference

    def forward(self, input):
        return F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, ceil_mode=self.ceil_mode
                            )

    def inference(self, input):
        return Round.apply(F.avg_pool2d(input, self.kernel_size, self.stride,
                                        self.padding, ceil_mode=self.ceil_mode
                                        ))


class QuantAvgPool1d(nn.AvgPool1d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, quant_inference=False):
        super().__init__(kernel_size, stride, padding, ceil_mode)
        self.quant_inference = quant_inference

    def forward(self, input):
        return F.avg_pool1d(input, self.kernel_size, self.stride,
                            self.padding, ceil_mode=self.ceil_mode
                            )

    def inference(self, input):
        return Round.apply(F.avg_pool1d(input, self.kernel_size, self.stride,
                                        self.padding, ceil_mode=self.ceil_mode
                                        ))


class QuantLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            a_bits=8,
            w_bits=8,
            quant_inference=False,
    ):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = SignedQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits, norm=False)

    @property
    def w_bits(self):
        return self.weight_quantizer.w_bits

    @property
    def a_bits(self):
        return self.activation_quantizer.a_bits

    def forward(self, input):
        quant_input = self.activation_quantizer(input)

        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        # with torch.no_grad():
        #     log_wanda(input, quant_input, self.weight, quant_weight, **kwargs)

        output = F.linear(quant_input, quant_weight, self.bias)
        return output

    def _inference(self, input):
        warn("QuantLinear._inference is deprecated and will be removed in the future.", DeprecationWarning)
        input = self.activation_quantizer.inference(input)

        w_bits = self.weight_quantizer.w_bits
        int_weight = self.weight_quantizer.inference(
            self.weight)  # if weight is already int, please comment this line

        x_next = 2 * F.linear(input, int_weight, self.bias) - F.linear(
            input, torch.ones_like(int_weight, requires_grad=False), self.bias) * (2 ** w_bits - 1)

        return x_next

    def inference(self, input):
        # ! self.activation_quantizer.forward should be reflected onto one's inference
        # If this is not realized, please using self._inference instead
        # the same as self.weight_quantizer
        input = self.activation_quantizer(input)

        w_bits = self.weight_quantizer.w_bits
        int_weight = self.weight_quantizer(self.weight)  # if weight is already int, please comment this line

        x_next = 2 * F.linear(input, int_weight, self.bias) - F.linear(
            input, torch.ones_like(int_weight, requires_grad=False), self.bias) * (2 ** w_bits - 1)

        return x_next

    def get_quanted_params(self):
        int_weight = self.weight_quantizer.inference(self.weight)
        int_weight = int_weight.to(determin_storage_type(int_weight))
        return {"int_weight": int_weight}


class QuantBatchNorm2D(nn.BatchNorm2d):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.5,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None,
                 w_bits=8,
                 pre_w_bits=8,
                 pre_a_bits=8,
                 quant_inference=False,
                 theta_init=2.0,
                 writer=None,
                 name=""):
        super(QuantBatchNorm2D, self).__init__(num_features, eps, momentum, affine, track_running_stats)  # , device, dtype)
        # self.weight_quantizer = WeightQuantizer(w_bits=w_bits, theta_init=theta_init, norm=False)
        self.weight_quantizer = SignedQuantizer(a_bits=w_bits)
        self.pre_a_bits = pre_a_bits
        self.pre_w_bits = pre_w_bits
        self.quant_inference = quant_inference
        self.writer = writer
        self.name = name
        self.recording = False
        self.record_epoch = 0
        self.rank = 0

    @property
    def w_bits(self):
        return self.weight_quantizer.w_bits

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # weight_mean = Round.apply(self.weight.mean())
        # quant_weight = (self.weight_quantizer(self.weight - weight_mean) + weight_mean) * (2 ** self.pre_w_bits - 1) / (
        #         2 ** self.pre_w_bits)
        quant_weight = self.weight_quantizer(self.weight)
        if self.recording and self.rank == 0:
            self.writer.add_histogram(f'{self.name}_weight', self.weight, self.record_epoch)
            self.writer.add_histogram(f'{self.name}_quant_weight', quant_weight, self.record_epoch)
        # quant_weight = self.weight

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            quant_weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def _inference(self, input):
        warn("QuantConv2d._inference is deprecated and will be removed in the future.", DeprecationWarning)

        int_weight = self.weight_quantizer.inference(
            self.weight - 1)  # if weight is already int, please comment this line

        int_running_mean = self.running_mean * (2 ** self.pre_a_bits - 1) * (
                2 ** self.pre_w_bits - 1) if not self.training or self.track_running_stats else None
        float_running_var = self.running_var if not self.training or self.track_running_stats else None
        int_bias = self.bias * (2 ** self.pre_a_bits - 1) * (2 ** self.pre_w_bits - 1) * (2 ** (self.w_bits - 1))

        x_next = F.batch_norm(
            input,
            int_running_mean,
            float_running_var,
            int_weight,
            int_bias,
            False,
            self.momentum,
            self.eps,
        )

        return x_next / (2 ** (self.w_bits - 1))

    def inference(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean = input.mean([0, 2, 3])
            var = input.var([0, 2, 3], unbiased=False) / ((2 ** self.pre_a_bits - 1) * (
                    2 ** self.pre_w_bits - 1)) ** 2
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        int_weight = self.weight_quantizer(
            self.weight - 1)

        int_running_mean = mean if not self.training or self.track_running_stats else None
        float_running_var = var if not self.training or self.track_running_stats else None
        int_bias = Round.apply(self.bias * (2 ** self.pre_a_bits - 1) * (2 ** self.pre_w_bits - 1) * (2 ** (self.w_bits - 1)))

        output = Round.apply(
            (input - int_running_mean[None, :, None, None]) / (torch.sqrt(float_running_var[None, :, None, None] + self.eps)))

        if self.affine:
            output = output * int_weight[None, :, None, None] + int_bias[None, :, None, None]

        return output / (2 ** (self.w_bits - 1))

    def get_quanted_params(self):
        int_weight = self.weight_quantizer.inference(
            self.weight - 1)  # if weight is already int, please comment this line

        # int_running_mean = self.running_mean * (2 ** self.pre_a_bits - 1) * (
        #         2 ** self.pre_w_bits - 1) if not self.training or self.track_running_stats else None
        int_running_mean = self.running_mean if not self.training or self.track_running_stats else None
        float_running_var = self.running_var if not self.training or self.track_running_stats else None
        int_bias = self.bias * (2 ** self.pre_a_bits - 1) * (2 ** self.pre_w_bits - 1)
        int_weight = int_weight.to(determin_storage_type(int_weight))
        int_running_mean = int_running_mean.to(determin_storage_type(int_running_mean))
        int_bias = int_bias.to(determin_storage_type(int_bias))
        return {"int_weight": int_weight, "int_running_mean": int_running_mean, "float_running_var": float_running_var,
                "int_bias": int_bias}


class QuantBatchNorm1D(nn.BatchNorm1d):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.5,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None,
                 w_bits=8,
                 pre_w_bits=8,
                 pre_a_bits=8,
                 quant_inference=False):
        super(QuantBatchNorm1D, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)
        self.pre_a_bits = pre_a_bits
        self.pre_w_bits = pre_w_bits
        self.quant_inference = quant_inference

    @property
    def w_bits(self):
        return self.weight_quantizer.w_bits

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None: 
                self.num_batches_tracked.add_(1) 
                if self.momentum is None: 
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        quant_weight = (self.weight_quantizer(self.weight - 1) + 1) * (2 ** self.pre_w_bits - 1) / (2 ** self.pre_w_bits)
        # quant_weight = self.weight

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            quant_weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def _inference(self, input):
        warn("QuantBatchNorm1D._inference is deprecated and will be removed in the future.", DeprecationWarning)
        int_weight = self.weight_quantizer.inference(
            self.weight - 1)  # if weight is already int, please comment this line

        int_running_mean = self.running_mean * (2 ** self.pre_a_bits - 1) * (
                2 ** self.pre_w_bits - 1) if not self.training or self.track_running_stats else None
        float_running_var = self.running_var if not self.training or self.track_running_stats else None
        int_bias = self.bias * (2 ** self.pre_a_bits - 1) * (2 ** self.pre_w_bits - 1) * (2 ** (self.w_bits - 1))

        x_next = F.batch_norm(
            input,
            int_running_mean,
            float_running_var,
            int_weight,
            int_bias,
            False,
            self.momentum,
            self.eps,
        )
        return x_next / (2 ** (self.w_bits - 1))

    def inference(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean = input.mean([0, 2])  # 注意此处mean是int, var是float
            var = input.var([0, 2], unbiased=False) / ((2 ** self.pre_a_bits - 1) * (
                    2 ** self.pre_w_bits - 1)) ** 2
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        int_weight = self.weight_quantizer(
            self.weight - 1)  # if weight is already int, please comment this line, the same as followings

        int_running_mean = mean if not self.training or self.track_running_stats else None
        float_running_var = var if not self.training or self.track_running_stats else None
        int_bias = self.bias * (2 ** self.pre_a_bits - 1) * (2 ** self.pre_w_bits - 1) * (2 ** (self.w_bits - 1))

        output = (input - int_running_mean[None, :, None]) / (torch.sqrt(float_running_var[None, :, None] + self.eps))

        if self.affine:
            output = output * int_weight[None, :, None] + int_bias[None, :, None]

        return output / (2 ** (self.w_bits - 1))

    def get_quanted_params(self):
        int_weight = self.weight_quantizer.inference(
            self.weight - 1)  # if weight is already int, please comment this line

        # int_running_mean = self.running_mean * (2 ** self.pre_a_bits - 1) * (
        #         2 ** self.pre_w_bits - 1) if not self.training or self.track_running_stats else None
        int_running_mean = self.running_mean if not self.training or self.track_running_stats else None
        float_running_var = self.running_var if not self.training or self.track_running_stats else None
        int_bias = self.bias * (2 ** self.pre_a_bits - 1) * (2 ** self.pre_w_bits - 1)
        int_weight = int_weight.to(determin_storage_type(int_weight))
        int_running_mean = int_running_mean.to(determin_storage_type(int_running_mean))
        int_bias = int_bias.to(determin_storage_type(int_bias))
        return {"int_weight": int_weight, "int_running_mean": int_running_mean, "float_running_var": float_running_var,
                "int_bias": int_bias}


class QuantConvBN2d(QuantConv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
            eps: float = 1e-5,
            momentum: float = 0.5,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
            a_bits=8,
            w_bits=8,
            quant_inference=False,
    ):
        super(QuantConvBN2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            a_bits=a_bits,
            w_bits=w_bits,
        )
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.register_buffer("running_scale1", torch.tensor(0.0))
        self.register_buffer("running_scale2", torch.tensor(0.0))
        if self.affine:
            self.weight_bn = Parameter(torch.empty(out_channels, **factory_kwargs))
            self.bias_bn = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("weight_bn", None)
            self.register_parameter("bias_bn", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(out_channels, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(out_channels, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        if self.track_running_stats:
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]
        if self.affine:
            init.ones_(self.weight_bn)
            init.zeros_(self.bias_bn)
        # self.e = ErrorStatic()
        self.quant_inference = quant_inference

    def forward(self, input, **kwargs):
        quant_input = self.activation_quantizer(input)  # from float to fixed, quant_input≈alpha*input

        # quant_input = input
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)  # from float to fixed, quant_weight≈weight
        else:
            quant_weight = self.weight

        output = F.conv2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        # self.e.update((cal_error(quant_input, input), cal_error(quant_weight, self.weight)))
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1 
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean = output.mean([0, 2, 3])
            var = output.var([0, 2, 3], unbiased=False)
            n = output.numel() / output.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        output = (output - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        quant_weight_bn = (self.weight_quantizer(self.weight_bn - 1) + 1) * (2 ** self.w_bits - 1) / (2 ** self.w_bits)

        if self.affine:
            output = output * quant_weight_bn[None, :, None, None] + self.bias_bn[None, :, None, None]

        return output

    def _inference(self, input):
        input = self.activation_quantizer.inference(input)

        w_bits = self.w_bits
        quant_weight_bn = (self.weight_quantizer(self.weight_bn - 1) + 1) * (2 ** w_bits - 1) / (2 ** w_bits)  # gamma∈[0,2]

        gamma = (quant_weight_bn / torch.sqrt(self.running_var + self.eps))
        int_weight_fused1 = 2 * self.weight_quantizer.inference(
            self.weight) * gamma[:, None, None, None]

        int_weight_fused2 = torch.ones_like(self.weight, requires_grad=False) * gamma[:, None, None, None] * (2 ** w_bits - 1) * (
                2 ** w_bits)
        int_bias_fused = Round.apply((2 ** self.a_bits - 1) * (2 ** w_bits - 1) * (
                self.bias_bn - gamma * self.running_mean))

        x_next = F.conv2d(input, int_weight_fused1, self.bias, self.stride, self.padding, self.dilation,
                          self.groups) - (
                     F.conv2d(input, int_weight_fused2, self.bias, self.stride, self.padding, self.dilation,
                              self.groups)) / (2 ** w_bits)
        return Round.apply(x_next) + int_bias_fused[None, :, None, None]

    def inference(self, input):
        # with torch.no_grad():
        quant_input = self.activation_quantizer.forward_copy(
            input / ((2 ** self.a_bits - 1) * (2 ** self.w_bits - 1)))

        quant_weight = self.weight_quantizer.forward_copy(self.weight)

        output = F.conv2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        int_input = self.activation_quantizer(input)

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean = output.mean([0, 2, 3])
            var = output.var([0, 2, 3], unbiased=False)
            n = output.numel() / output.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        w_bits = self.w_bits

        quant_weight_bn = self.weight_quantizer(self.weight_bn - 1) / (2 ** (w_bits - 1))

        gamma = (quant_weight_bn / torch.sqrt(var + self.eps))

        int_weight_fused1 = Round.apply(self.weight_quantizer(self.weight) * gamma[:, None, None, None])

        scale1 = torch.ceil(torch.log2(int_weight_fused1.max() / (2 ** w_bits - 1)))
        self.running_scale1 = exponential_average_factor * scale1 \
                              + (1 - exponential_average_factor) * self.running_scale1.data

        int_weight_fused1 = Round.apply(torch.clamp(int_weight_fused1 / (2 ** self.running_scale1), 0, 2 ** w_bits - 1))

        int_weight_fused2 = Round.apply(
            torch.ones_like(self.weight, requires_grad=False) * gamma[:, None, None, None] * (2 ** w_bits - 1))
        scale2 = torch.ceil(torch.log2(int_weight_fused2.max() / (2 ** w_bits - 1)))
        self.running_scale2 = exponential_average_factor * scale2 \
                              + (1 - exponential_average_factor) * self.running_scale2.data

        int_weight_fused2 = Round.apply(torch.clamp(int_weight_fused2 / (2 ** self.running_scale2), 0, 2 ** w_bits - 1))

        int_bias_fused = Round.apply((2 ** self.a_bits - 1) * (2 ** w_bits - 1) * (
                self.bias_bn - gamma * mean))

        x_next = 2 * (2 ** self.running_scale1) * F.conv2d(int_input, int_weight_fused1, self.bias, self.stride, self.padding,
                                                           self.dilation,
                                                           self.groups) - (
                     F.conv2d(int_input, int_weight_fused2, self.bias, self.stride, self.padding, self.dilation,
                              self.groups)) * (2 ** self.running_scale2)

        return Round.apply(x_next) + int_bias_fused[None, :, None, None]

    def get_quanted_params(self):
        quant_weight_bn = (self.weight_quantizer.forward(self.weight_bn - 1) + 1) * (2 ** self.w_bits - 1) / (2 ** self.w_bits)
        gamma = (quant_weight_bn / torch.sqrt(self.running_var + self.eps))
        int_weight_fused1 = self.weight_quantizer.inference(
            self.weight) * gamma[:, None, None, None]
        int_weight_fused2 = torch.ones_like(self.weight, requires_grad=False) * gamma[:, None, None, None] * (2 ** w_bits - 1)
        int_bias_fused = Round.apply((2 ** self.a_bits - 1) * (2 ** self.w_bits - 1) * (
                self.bias_bn - gamma * self.running_mean))

        int_weight_fused1 = int_weight_fused1.to(determin_storage_type(int_weight_fused1))
        int_weight_fused2 = int_weight_fused2.to(determin_storage_type(int_weight_fused2))
        int_bias_fused = int_bias_fused.to(determin_storage_type(int_bias_fused))
        return {"int_weight_fused1": int_weight_fused1, "int_weight_fused2": int_weight_fused2, "int_bias_fused": int_bias_fused}


class QuantAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    def __init__(self, output_size, quant_inference=False):
        super().__init__(output_size)
        self.quant_inference = quant_inference

    def forward(self, input):
        return F.adaptive_avg_pool1d(input, self.output_size)

    def inference(self, input):
        return Round.apply(F.adaptive_avg_pool1d(input, self.output_size))


class QuantAdaptiveMaxPool1d(nn.AdaptiveMaxPool1d):
    def __init__(self, output_size, quant_inference=False):
        super().__init__(output_size)
        self.quant_inference = quant_inference

    def forward(self, input):
        return F.adaptive_max_pool1d(input, self.output_size)

    def inference(self, input):
        return F.adaptive_max_pool1d(input, self.output_size)


def minmaxnorm(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1E-8)


def cal_error(x, y):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    x = minmaxnorm(x.detach())
    y = minmaxnorm(y.detach())
    assert x.ndim == y.ndim
    error = F.mse_loss(x, y, reduction="mean")
    # print(f"error = {error}")
    return error, error / x.mean() * 100, error / y.mean() * 100


def determin_storage_type(x: torch.Tensor):
    x_max = x.abs().max()
    if x_max < 2 ** 8:
        return torch.int8
    elif x_max < 2 ** 16:
        return torch.int16
    elif x_max < 2 ** 32:
        return torch.int32
    else:
        raise ValueError("Some value in tensor is too large, please check this tensor")


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8, quant_inference=False, writer=None, prefix_name="net"):
    module: nn.Module
    for name, child in module.named_children():
        full_name = prefix_name + "." + name
        if isinstance(child, nn.Conv2d):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 0:
                if child.bias is not None:
                    quant_conv = QuantConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=True,
                        padding_mode=child.padding_mode,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                        writer=writer,
                        name=full_name,
                        first_layer=layer_counter[0] == 1
                    )
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=False,
                        padding_mode=child.padding_mode,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                        writer=writer,
                        name=full_name,
                        first_layer=layer_counter[0] == 1
                    )
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
            # layer_counter[0] += 1
        elif isinstance(child, nn.Conv1d):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    quant_conv = QuantConv1d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=True,
                        padding_mode=child.padding_mode,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                    )
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv1d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=False,
                        padding_mode=child.padding_mode,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                    )
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
            # layer_counter[0] += 1
        elif isinstance(child, nn.ConvTranspose2d):
            continue
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    quant_conv_transpose = QuantConvTranspose2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        output_padding=child.output_padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=True,
                        padding_mode=child.padding_mode,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                    )
                    quant_conv_transpose.bias.data = child.bias
                else:
                    quant_conv_transpose = QuantConvTranspose2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        output_padding=child.output_padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=False,
                        padding_mode=child.padding_mode,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                    )
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    quant_linear = QuantLinear(
                        child.in_features,
                        child.out_features,
                        bias=True,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                    )
                    quant_linear.bias.data = child.bias
                else:
                    quant_linear = QuantLinear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        quant_inference=quant_inference,
                    )
                quant_linear.weight.data = child.weight
                module._modules[name] = quant_linear
        elif isinstance(child, (nn.ReLU6, nn.ReLU)):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 0:
                quant_relu6 = QuantReLU6(a_bits=a_bits, w_bits=w_bits, inplace=child.inplace, quant_inference=quant_inference)
                module._modules[name] = quant_relu6
        elif isinstance(child, nn.MaxPool2d):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                quant_maxpool2d = QuantMaxPool2d(child.kernel_size, child.stride, child.padding, child.dilation,
                                                 child.return_indices, child.ceil_mode, quant_inference=quant_inference)
                module._modules[name] = quant_maxpool2d
        elif isinstance(child, nn.MaxPool1d):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                quant_maxpool1d = QuantMaxPool1d(child.kernel_size, child.stride, child.padding, child.dilation,
                                                 child.return_indices, child.ceil_mode, quant_inference=quant_inference)
                module._modules[name] = quant_maxpool1d
        elif isinstance(child, nn.AvgPool2d):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                quant_avgpool2d = QuantAvgPool2d(child.kernel_size, child.stride, child.padding, child.ceil_mode,
                                                 quant_inference=quant_inference)
                module._modules[name] = quant_avgpool2d
        elif isinstance(child, nn.AvgPool1d):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                quant_avgpool1d = QuantAvgPool1d(child.kernel_size, child.stride, child.padding, child.ceil_mode,
                                                 quant_inference=quant_inference)
                module._modules[name] = quant_avgpool1d
        elif isinstance(child, nn.AdaptiveAvgPool1d):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                quant_adaptiveavgpool1d = QuantAdaptiveAvgPool1d(child.output_size, quant_inference=quant_inference)
                module._modules[name] = quant_adaptiveavgpool1d
        elif isinstance(child, nn.AdaptiveMaxPool1d):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                quant_adaptivemaxpool1d = QuantAdaptiveMaxPool1d(child.output_size, quant_inference=quant_inference)
                module._modules[name] = quant_adaptivemaxpool1d
        elif isinstance(child, nn.BatchNorm2d):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 0:
                quant_bn2d = QuantBatchNorm2D(child.num_features, eps=child.eps, momentum=0.5, affine=child.affine,
                                              track_running_stats=child.track_running_stats, w_bits=w_bits, pre_a_bits=a_bits,
                                              pre_w_bits=w_bits, quant_inference=quant_inference, writer=writer, theta_init=2.0,
                                              name=full_name)
                if child.affine:
                    quant_bn2d.weight.data = child.weight.detach().clone()
                    quant_bn2d.bias.data = child.bias.detach().clone()
                module._modules[name] = quant_bn2d
        elif isinstance(child, nn.BatchNorm1d):
            # continue
            layer_counter[0] += 1
            if layer_counter[0] > 3:
                quant_bn1d = QuantBatchNorm1D(child.num_features, eps=child.eps, momentum=child.momentum, affine=child.affine,
                                              track_running_stats=child.track_running_stats, w_bits=w_bits, pre_a_bits=a_bits,
                                              pre_w_bits=w_bits, quant_inference=quant_inference)
                if child.affine:
                    quant_bn1d.weight.data = child.weight.detach().clone()
                    quant_bn1d.bias.data = child.bias.detach().clone()
                module._modules[name] = quant_bn1d
        elif isinstance(child, ConvBn):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                conv = child.conv
                bn = child.bn
                bias = True if conv.bias is not None else False
                fused_convbn = QuantConvBN2d(in_channels=conv.in_channels,
                                             out_channels=conv.out_channels,
                                             kernel_size=conv.kernel_size, stride=conv.stride,
                                             padding=conv.padding, dilation=conv.dilation,
                                             groups=conv.groups, bias=bias,
                                             padding_mode=conv.padding_mode, eps=bn.eps,
                                             momentum=bn.momentum, affine=bn.affine,
                                             track_running_stats=bn.track_running_stats,
                                             a_bits=a_bits,
                                             w_bits=w_bits,
                                             quant_inference=quant_inference)
                fused_convbn.weight.data = conv.weight.detach().clone()
                if bias:
                    fused_convbn.bias.data = conv.bias.detach().clone()
                if bn.affine:
                    fused_convbn.weight_bn.data = bn.weight.detach().clone()
                    fused_convbn.bias_bn.data = bn.bias.detach().clone()
                if bn.track_running_stats:
                    fused_convbn.running_var.data = bn.running_var.detach().clone()
                    fused_convbn.running_mean.data = bn.running_mean.detach().clone()
                module._modules[name] = fused_convbn
        # elif isinstance(child, (nn.Sequential, nn.ModuleList)):
        #     add_quant_op(
        #         child,
        #         layer_counter,
        #         a_bits=a_bits,
        #         w_bits=w_bits,
        #         quant_inference=quant_inference,
        #     )
        else:
            add_quant_op(
                child,
                layer_counter,
                a_bits=a_bits,
                w_bits=w_bits,
                quant_inference=quant_inference,
                writer=writer,
                prefix_name=full_name
            )


def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def fuse_conv_bn_eval(module: torch.nn.Module, conv_bn_pairs):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            for conv, bn in conv_bn_pairs:
                if conv is m:
                    module._modules[n] = ConvBn(conv, bn)
                    break
        elif isinstance(m, nn.BatchNorm2d):
            for conv, bn in conv_bn_pairs:
                if bn is m:
                    module._modules[n] = QuantIdentity()
                    break
        else:
            fuse_conv_bn_eval(m, conv_bn_pairs)


def fuse(model, inplace=False):
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(nn.Conv1d, nn.BatchNorm1d),
                (nn.Conv2d, nn.BatchNorm2d),
                (nn.Conv3d, nn.BatchNorm3d)]
    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = fx_model.graph

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                fuse_conv_bn_eval(model, [[conv, bn]])
    return model


def pre_hook(module, input):
    if hasattr(module, "activation_quantizer"):
        if isinstance(input, tuple):
            inp = input[0]
        else:
            inp = input
        if module.quant_inference:
            inp_max = inp.max() / ((2 ** a_bits - 1) * (2 ** w_bits - 1))
        else:
            inp_max = inp.max()

        b = torch.clamp(torch.ceil(torch.log2(inp_max)), 0, 8)
        # b = torch.clamp(torch.log2(inp_max), 0, 8)
        module.activation_quantizer.beta.data = b.detach().clone().data
    if hasattr(module, "weight_quantizer"):
        if torch.max(torch.abs(module.weight)) < 1E-6:
            print("Warning: there is some weight close to zero, please check your state_dict")
        if isinstance(module, QuantBatchNorm2D):
            b = torch.clamp(torch.ceil(torch.log2(torch.abs(module.weight).max())), 0, 8)
            module.weight_quantizer.beta.data = b.detach().clone().data
    return input


def replace_forward_with_inference(model):
    for name, module in model.named_modules():
        if hasattr(module, 'inference'):
            module.forward = module.inference

def find_error(net):
    errors = []
    for m in net.children():
        if hasattr(m, "activation_quantizer"):
            errors.append(m.activation_quantizer.beta_error)
        elif hasattr(m, "weight_quantizer") and isinstance(m, QuantBatchNorm2D):
            errors.append(m.weight_quantizer.beta_error)
        else:
            errors.extend(find_error(m))
    return errors

def find_beta(net):
    beta_paras = []
    for m in net.children():
        # if isinstance(m, QuantConvBN2d):
        if hasattr(m, "activation_quantizer"):
            beta_paras.append(m.activation_quantizer.beta)
        elif hasattr(m, "weight_quantizer") and isinstance(m, QuantBatchNorm2D):
            beta_paras.append(m.weight_quantizer.beta)
        else:
            beta_paras.extend(find_beta(m))
    return beta_paras


def find_beta_with_name(net, prefix=""):
    beta_paras = []
    names = []
    for n, m in net.named_children():
        n = (prefix + "_" + n).replace(".", "_")
        # if isinstance(m, QuantConvBN2d):
        if hasattr(m, "activation_quantizer"):
            beta_paras.append(m.activation_quantizer.beta)
            names.append(n)
        elif hasattr(m, "weight_quantizer") and isinstance(m, QuantBatchNorm2D):
            beta_paras.append(m.weight_quantizer.beta)
            names.append(n)
        else:
            b_, n_ = find_beta_with_name(m, n)
            beta_paras.extend(b_)
            names.extend(n_)
    return beta_paras, names


def find_the_with_name(net, prefix=""):
    theta_paras = []
    names = []
    for n, m in net.named_children():
        n = (prefix + "_" + n).replace(".", "_")
        # if isinstance(m, QuantConvBN2d):
        if hasattr(m, "weight_quantizer") and isinstance(m, (QuantLinear, QuantConv2d, QuantConvBN2d)):
            theta_paras.append(m.weight_quantizer.the)
            names.append(n)
        else:
            b_, n_ = find_the_with_name(m, n)
            theta_paras.extend(b_)
            names.extend(n_)
    return theta_paras, names


def find_the(net):
    theta_paras = []
    for m in net.children():
        # if isinstance(m, QuantConvBN2d):
        if hasattr(m, "weight_quantizer") and isinstance(m, (QuantConv2d, QuantLinear, QuantConvBN2d)):
            theta_paras.append(m.weight_quantizer.the)
        else:
            theta_paras.extend(find_the(m))
    return theta_paras


def initi_beta_with_input(net, input):
    net: nn.Module
    net.train()
    handles = []
    for m in net.modules():
        handles.append(m.register_forward_pre_hook(pre_hook))
    with torch.no_grad():
        net(input)
        print("beta has been inited with: ", list(i.item() for i in find_beta(net)))
    net.eval()
    for h in handles:
        h.remove()


def prepare(model, inplace=False, a_bits=8, w_bits=8, fuse_model=False, quant_inference=False, calibration_input=None,
            writer=None):
    if fuse_model:
       model = fuse(model, inplace)
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(
        model,
        layer_counter,
        a_bits=a_bits,
        w_bits=w_bits,
        quant_inference=quant_inference,
        writer=writer
    )
    if quant_inference:
        replace_forward_with_inference(model)
    if calibration_input is not None:
        initi_beta_with_input(model, calibration_input)
    return model


def save_state(model, best_acc):
    print("==> Saving model ...")
    state = {
        "best_acc": best_acc,
        "state_dict": model.state_dict(),
    }
    state_copy = state["state_dict"].copy()
    for key in state_copy.keys():
        if "module" in key:
            state["state_dict"][key.replace("module.", "")] = state["state_dict"].pop(
                key
            )
    torch.save(state, "./dorefatestmodel.pth")


def get_quanted_params(model, name=""):
    quanted_params_dict = {}
    for n, m in model.named_children():
        new_name = name + "." + n
        if hasattr(m, "get_quanted_params"):
            params = m.get_quanted_params()
            for item in params:
                quanted_params_dict[new_name + "." + item] = params[item]
        else:
            quanted_params_dict.update(get_quanted_params(m, new_name))
    return quanted_params_dict


def save_quanted_state(model):
    print("==> Saving quanted model ...")
    quanted_params = get_quanted_params(model)
    torch.save(quanted_params, "./dorefatestmodel_quanted_fused.pth")


def adjust_learning_rate(optimizer, epoch):
    update_list = [40, 70]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.5
    return


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

