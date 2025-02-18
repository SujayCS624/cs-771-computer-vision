import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math

from utils import resize_image
import custom_transforms as transforms
from custom_blocks import (PatchEmbed, window_partition, window_unpartition,
                           DropPath, MLP, trunc_normal_)


#################################################################################
# You will need to fill in the missing code in this file
#################################################################################


#################################################################################
# Part I.1: Understanding Convolutions
#################################################################################
class CustomConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
        """
        Forward propagation of convolution operation.
        We only consider square filters with equal stride/padding in width and height!

        Args:
          input_feats: input feature map of size N * C_i * H * W
          weight: filter weight of size C_o * C_i * K * K
          bias: (optional) filter bias of size C_o
          stride: (int, optional) stride for the convolution. Default: 1
          padding: (int, optional) Zero-padding added to both sides of the input. Default: 0

        Outputs:
          output: responses of the convolution  w*x+b

        """
        # sanity check
        assert weight.size(2) == weight.size(3)
        assert input_feats.size(1) == weight.size(1)
        assert isinstance(stride, int) and (stride > 0)
        assert isinstance(padding, int) and (padding >= 0)

        # save the conv params
        kernel_size = weight.size(2)
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_height = input_feats.size(2)
        ctx.input_width = input_feats.size(3)

        # make sure this is a valid convolution
        assert kernel_size <= (input_feats.size(2) + 2 * padding)
        assert kernel_size <= (input_feats.size(3) + 2 * padding)

        #################################################################################
        # Fill in the code here
        input_unfolded = unfold(input_feats, kernel_size=kernel_size, stride=stride, padding=padding) # N, Ci*K*K, L*L
        weight_unfolded = weight.reshape(weight.size(0), -1) # Co, Ci*K*K
        output_flatten = torch.matmul(weight_unfolded, input_unfolded) # N, Co, L*L
        if bias is not None:
            output_flatten = output_flatten + bias.unsqueeze(1) # (1,) C_o, 1
        L = int(math.sqrt(output_flatten.size(2))) # Patch count *per dimension*
        output = output_flatten.reshape(output_flatten.size(0), output_flatten.size(1), L, L) # N, Co, L, L
        #################################################################################

        # save for backward (you need to save the unfolded tensor into ctx)
        ctx.save_for_backward(input_unfolded, weight, bias)

        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation of convolution operation

        Args:
          grad_output: gradients of the outputs

        Outputs:
          grad_input: gradients of the input features
          grad_weight: gradients of the convolution weight
          grad_bias: gradients of the bias term

        """
        # unpack tensors and initialize the grads
        input_unfolded, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # recover the conv params
        kernel_size = weight.size(2)
        stride = ctx.stride
        padding = ctx.padding
        input_height = ctx.input_height
        input_width = ctx.input_width

        #################################################################################
        # Fill in the code here
        # grad_output: N, Co, L, L
        # input_unfolded: N, Ci*K*K, L*L
        grad_output_flatten = grad_output.reshape(grad_output.size(0), grad_output.size(1), -1) # N, Co, L*L

        grad_weight_flatten = torch.matmul(grad_output_flatten, input_unfolded.transpose(1, 2)) # N, Co, Ci*K*K
        grad_weight = grad_weight_flatten.sum(0).reshape(*weight.shape) # Co, Ci, K, K

        # grad_output_flatten: N, Co, L * L
        weight_flatten = weight.reshape(weight.size(0), -1) # Co, Ci*K*K
        grad_input_unfolded = torch.matmul(weight_flatten.transpose(0, 1), grad_output_flatten) # N, Ci*K*K, L*L
        grad_input = fold(grad_input_unfolded, (input_height, input_width),
                          kernel_size=kernel_size, stride=stride, padding=padding)


        #################################################################################
        # compute the gradients w.r.t. input and params

        if bias is not None and ctx.needs_input_grad[2]:
            # compute the gradients w.r.t. bias (if any)
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None


custom_conv2d = CustomConv2DFunction.apply


class CustomConv2d(Module):
    """
    The same interface as torch.nn.Conv2D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CustomConv2d, self).__init__()
        assert isinstance(kernel_size, int), "We only support squared filters"
        assert isinstance(stride, int), "We only support equal stride"
        assert isinstance(padding, int), "We only support equal padding"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # not used (for compatibility)
        self.dilation = dilation
        self.groups = groups

        # register weight and bias as parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialization using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # call our custom conv2d op
        return custom_conv2d(input, self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


#################################################################################
# Part I.2: Design and train a convolutional network
#################################################################################
class SimpleNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.attack = default_attack(nn.CrossEntropyLoss(), num_steps=5)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.consintat_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        if self.training:
        #   # generate adversarial sample based on x
            # Randomly decide to perturb the input 20% of the time
            if np.random.rand() < 0.2:
                self.eval()
                x = self.attack.perturb(self, x)
                self.train()
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class BetterNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(BetterNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.attack = default_attack(nn.CrossEntropyLoss(), num_steps=5)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.consintat_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        # if self.training:
        # #   # generate adversarial sample based on x
        #     # Randomly decide to perturb the input 20% of the time
        #     if np.random.rand() < 0.2:
        #         self.eval()
        #         x = self.attack.perturb(self, x)
        #         self.train()
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# change this to your model!
# default_cnn_model = SimpleNet
default_cnn_model = BetterNet

#################################################################################
# Part II.1: Understanding self-attention
#################################################################################
class Attention(nn.Module):
    """Multi-head Self-Attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
    ):
        """
        Args:
            dim (int): Number of input channels. We assume Q, K, V will be of
                same dimension as the input.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # linear projection for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # linear projection at the end
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # input size (B, H, W, C)
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(
                B, H * W, 3, self.num_heads, -1
            ).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        #################################################################################
        # Fill in the code here

        # Calculate attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale 

        # Apply softmax to get attention weights
        attn_weights = attn_scores.softmax(dim=-1)  

        # Compute the output
        x = (attn_weights @ v) 

        # Reshape back to the original dimensions while maintaining the number of patches
        x = x.transpose(1, 2).reshape(B, H * W, _) 

        # Final projection
        x = self.proj(x) 

        # Reshape back to (B, H, W, C) to maintain original input structure
        x = x.view(B, H, W, _)  

        #################################################################################
        return x

class TransformerBlock(nn.Module):
    """Transformer blocks with support of local window self-attention"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        window_size=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            window_size (int): Window size for window attention blocks.
                If it equals 0, then global attention is used.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer
        )

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#################################################################################
# Part II.2: Design and train a vision Transformer
#################################################################################
class SimpleViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=128,
        num_classes=100,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_size=4,
        window_block_indexes=(0, 2),
    ):
        """
        Args:
            img_size (int): Input image size.
            num_classes (int): Number of object categories
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            window_size (int): Window size for local attention blocks.
            window_block_indexes (list): Indexes for blocks using local attention.
                Local window attention allows more efficient computation, and can be
                coupled with standard global attention. E.g., [0, 2] indicates the
                first and the third blocks will use local window attention, while
                other block use standard attention.

        Feel free to modify the default parameters here.
        """
        super(SimpleViT, self).__init__()

        if use_abs_pos:
            # Initialize absolute positional embedding with image size
            # The embedding is learned from data
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # patch embedding layer
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        ########################################################################
        # Fill in the code here

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=window_size if i in window_block_indexes else 0  # Use local window attention for specific blocks
            ) for i in range(depth)
        ])

        # Final classification layer
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        ########################################################################
        # The implementation shall define some Transformer blocks

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        # add any necessary weight initialization here

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        ########################################################################
        # Fill in the code here
        
        # Patch embedding
        x = self.patch_embed(x) 

        # Add positional embedding if available
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Apply layer norm
        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=[1,2])  

        # Classification head - Now we have each input image, with all the logits for all classes.
        x = self.head(x)  

        ########################################################################
        return x

# change this to your model!
default_vit_model = SimpleViT

# define data augmentation used for training, you can tweak things if you want
def get_train_transforms():
    train_transforms = []
    train_transforms.append(transforms.Scale(144))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomColor(0.15))
    train_transforms.append(transforms.RandomRotate(15))
    train_transforms.append(transforms.RandomSizedCrop(128))
    train_transforms.append(transforms.ToTensor())
    # mean / std from imagenet
    train_transforms.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ))
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms

# define data augmentation used for validation, you can tweak things if you want
def get_val_transforms():
    val_transforms = []
    val_transforms.append(transforms.Scale(144))
    val_transforms.append(transforms.CenterCrop(128))
    val_transforms.append(transforms.ToTensor())
    # mean / std from imagenet
    val_transforms.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ))
    val_transforms = transforms.Compose(val_transforms)
    return val_transforms


#################################################################################
# Part III: Adversarial samples
#################################################################################
class PGDAttack(object):
    def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.1):
        """
        Attack a network by Project Gradient Descent. The attacker performs
        k steps of gradient descent of step size a, while always staying
        within the range of epsilon (under l infinity norm) from the input image.

        Args:
          loss_fn: loss function used for the attack
          num_steps: (int) number of steps for PGD
          step_size: (float) step size of PGD (i.e., alpha in our lecture)
          epsilon: (float) the range of acceptable samples
                   for our normalization, 0.1 ~ 6 pixel levels
        """
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

    def perturb(self, model, input):
        """
        Given input image X (torch tensor), return an adversarial sample
        (torch tensor) using PGD of the least confident label.

        See https://openreview.net/pdf?id=rJzIBfZAb

        Args:
          model: (nn.module) network to attack
          input: (torch tensor) input image of size N * C * H * W

        Outputs:
          output: (torch tensor) an adversarial sample of the given network
        """
        # clone the input tensor and disable the gradients
        output = input.clone()
        input.requires_grad = False
        output.requires_grad = True

        # loop over the number of steps
        for _ in range(self.num_steps):
        #################################################################################
        # Fill in the code here

            # Get the logits for the input image
            logits = model(output)
            # Find the index of the least confident label
            _, least_conf_pred_idx = torch.min(logits, 1)
            # Use least confident label as proxy for incorrect label
            loss = self.loss_fn(logits, least_conf_pred_idx)

            # Backprop to compute the gradients
            model.zero_grad()
            loss.backward()

            # Perturb the input as per fast gradient sign method
            output_grad = output.grad.data
            output_grad_sign = torch.sign(output_grad)
            output.data = output.data + self.step_size * output_grad_sign
            # Clip the result to be within the epsilon boundary
            # output.data = torch.max(torch.min(output.data, input + self.epsilon), input - self.epsilon)
            output.data = torch.clamp(output.data, input - self.epsilon, input + self.epsilon)
            output.grad.zero_()

        #################################################################################

        return output

default_attack = PGDAttack


def vis_grid(input, n_rows=10):
    """
    Given a batch of image X (torch tensor), compose a mosaic for visualziation.

    Args:
      input: (torch tensor) input image of size N * C * H * W
      n_rows: (int) number of images per row

    Outputs:
      output: (torch tensor) visualizations of size 3 * HH * WW
    """
    # concat all images into a big picture
    output_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
    return output_imgs

default_visfunction = vis_grid
