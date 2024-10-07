import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from irpe import get_rpe_config, build_rpe

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        
    Additional Args:
        oup (int): Number of output channels
        contextual_rpe(bool): If True, use relative RPE, else use bias RPE
        contextual_input(str): Add contextual RPE on keys, queries and values. 
                            For example, to add contextual RPE on queries and values, 
                            the input string should be 'q,v'
    
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 oup = None,
                 # the following argumennt is for contextual RPE
                 contextual_rpe=False, contextual_input = None,
                 ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.oup = oup
        self.contextual_rpe = contextual_rpe
        self.contextual_input = contextual_input
        
        if self.contextual_rpe:
            # contextual RPE
            rpe_config = get_rpe_config(
                ratio=1.9,
                method="product",
                mode='contextual',
                shared_head=True,
                skip=0,
                rpe_on= self.contextual_input,
            )
            self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config,
                          head_dim=head_dim,
                          num_heads=num_heads)
        else:
            # bias RPE
            
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
    
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, oup)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if not self.contextual_rpe:
            trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # batch*window_number, head, token, channel => batch*window_number, head, token, token
        
        if self.contextual_rpe:
            # image relative position on keys
            if self.rpe_k is not None:
                attn += self.rpe_k(q, height=self.window_size[0], width=self.window_size[1])
            
            # image relative position on queries
            if self.rpe_q is not None:
                attn += self.rpe_q(k * self.scale).transpose(2, 3)
        
        else:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn) # output => batch*window_number, head, token, token
        
        out = attn @ v
        
        # image relative position on values
        if self.contextual_rpe:
            if self.rpe_v is not None:
                out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4, 
                 
                 
                 shift_size=None,
                 num_head=0,
                 ):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)
    

class Swin_TNT_block(nn.Module):
    def __init__(self, inp, oup, image_size, num_head=0, dim_head=32, dropout=0., shift_size=None,
                 mlp_ratio=4., window_size = 7, drop_path = None,
                 drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 #fused_window_process=False,
                 qkv_bias=True, qk_scale=None,
                 contextual_rpe=False,
                 contextual_input = None,
                 T_in_T = False,
                 ):
        super().__init__()
        #hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.inp = inp
        self.oup = oup
        self.input_resolution = image_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        ### build layers for self-attention within windows
        dim = inp
        self.norm1 = norm_layer(dim)
        self.dim = dim
        self.num_heads = num_head
        num_heads = num_head
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            oup = oup,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            # the following argumennt is for contextual RPE
            contextual_rpe=contextual_rpe,
            contextual_input = contextual_input,
            )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(oup)
        mlp_hidden_dim = int(oup * mlp_ratio)
        self.mlp = Mlp(in_features=oup, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        
        fused_window_process = False
        self.fused_window_process = fused_window_process

        # build layers for self-attention between windows
        dim = oup
        self.T_in_T = T_in_T
        if self.T_in_T:
            self.T_in_T_pooling = nn.AdaptiveAvgPool1d(1)
            self.GA_upsample = nn.Upsample(scale_factor=self.window_size, mode='nearest')
            self.norm1_GA = norm_layer(dim)
            self.drop_path_GA = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.norm2_GA = norm_layer(dim, elementwise_affine=True)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp_GA = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            
            H, W =self.input_resolution
            self.group_size_GA = (H // self.window_size, W // self.window_size)
            self.attn_GA = WindowAttention(
                dim, window_size=self.group_size_GA, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                oup = oup,
                contextual_rpe=contextual_rpe,
                contextual_input = contextual_input,
                )
            
            self.attn_mask_GA = None

    def forward(self, x):

        H, W = self.ih, self.iw
        B, C, Hx, Wx = x.shape
        assert Hx == H, "input feature has wrong size"
        assert Wx == W, "input feature has wrong size"
        
        x = x.permute(0,2,3,1)
        x = x.view(B, H* W, C)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        
        _,_,C = attn_windows.shape
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        _,_,C = x.shape
        if self.T_in_T:
            shortcut = x
            x = self.norm1_GA(x)
            x = x.view(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                if not self.fused_window_process:
                    shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                    # partition windows
                    x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
                else:
                    x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
            else:
                shifted_x = x
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

 
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
            
            # Pool each window into a token
            x_windows = x_windows.permute(0, 2, 1) # nW*B, C, window_size*window_size
            x_windows = self.T_in_T_pooling(x_windows)
            x_windows = x_windows.flatten(1)
            x_windows = x_windows.view(B, (H // self.window_size) * (W // self.window_size), C) # # B, nW, C
                
            # W-MSA/SW-MSA between windows
            attn_windows = self.attn_GA(x_windows, mask=self.attn_mask_GA)
            
            # Upsample each token into the window size
            attn_windows = attn_windows.view(B, H // self.window_size, W // self.window_size, C).permute(0, 3, 1, 2) # B, C, nW (horizontal), nW (vertical)
            attn_windows = self.GA_upsample(attn_windows) # B, C, H, W
            attn_windows = attn_windows.permute(0, 2, 3, 1) # B, H, W, C
            attn_windows = attn_windows.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C).permute(0, 1, 3, 2, 4, 5)
            attn_windows = attn_windows.reshape(B * H * W // self.window_size**2, self.window_size**2, C) # nW*B, window_size*window_size, C
            
            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

            # reverse cyclic shift
            if self.shift_size > 0:
                if not self.fused_window_process:
                    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                    x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                else:
                    x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
            else:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = shifted_x
            x = x.view(B, H * W, C)
            x = shortcut + self.drop_path_GA(x)

            # FFN
            x = x + self.drop_path_GA(self.mlp_GA(self.norm2_GA(x)))
            
        x = x.view(B, H, W, C).permute(0,3,1,2) #B, C, H, W
        
        return x


class LG_Swin_TNT_block(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0., shift_size=None,
                 mlp_ratio=4., window_size = 7, drop_path = None,
                 drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_head=0,
                 qkv_bias=True, qk_scale=None,
                 contextual_rpe=False,
                 contextual_input = None,
                 T_in_T = False,
                 ):
        super().__init__()
        
        # MBConv layer
        self.mbconv = MBConv(inp, oup, image_size, downsample=downsample)
        
        # followed by 2 Swin-TNT block
        inp_transformer = oup
        oup_transformer = oup
        shift_size_swinblock_1 = 0
        shift_size_swinblock_2 = window_size // 2     
        self.swinblock_1 = Swin_TNT_block(
                            inp_transformer, oup_transformer, image_size, 
                            shift_size=shift_size_swinblock_1,
                            drop_path=drop_path,
                            drop=drop, attn_drop=attn_drop,
                            num_head=num_head,
                            qkv_bias=True, qk_scale=None,
                            contextual_rpe=contextual_rpe,
                            contextual_input = contextual_input,
                            T_in_T = T_in_T,
                            )
        self.swinblock_2 = Swin_TNT_block(
                            inp_transformer, oup_transformer, image_size, 
                            shift_size=shift_size_swinblock_2,
                            drop_path=drop_path,
                            drop=drop, attn_drop=attn_drop,
                            num_head=num_head,
                            qkv_bias=True, qk_scale=None,
                            contextual_rpe=contextual_rpe,
                            contextual_input = contextual_input,
                            T_in_T = T_in_T,
                            )
        
    def forward(self, x):
        x = self.mbconv(x)
        x = self.swinblock_1(x)
        x = self.swinblock_2(x)
        return x


class histo_densevit(nn.Module):
    """
    Explanation of key args in Histo-DenseViT:
        T_in_T (bool): If True, implement the self-attention between windows in the Swin-TNT block
        deep_dense (bool): If True, implement dense connection between stages
        contextual_rpe(bool): If True, use relative RPE, else use bias RPE
        contextual_input(str): Add contextual RPE on keys, queries and values. 
                            For example, to add contextual RPE on queries and values, 
                            the input string should be 'q,v'
    """
    def __init__(self, image_size, num_classes=1000,
                 in_channels = 3,
                 channels = [64, 96, 192, 384, 768],
                 num_blocks = [2, 1, 1, 1, 1],
                 num_heads=[0, 3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 T_in_T = False,
                 deep_dense = False,
                 contextual_rpe=False,
                 contextual_input = None,
                 ):
        super().__init__()
        ih, iw = image_size
        self.drop_path_rate =  drop_path_rate
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.depths = num_blocks
        self.dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale

        self.contextual_rpe=contextual_rpe
        self.contextual_input = contextual_input
        self.T_in_T = T_in_T
        self.deep_dense = deep_dense
        
        if self.deep_dense:
            num_heads=[0, 1, 3, 6, 12]
            self.s0 = self._make_layer(
                conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
            self.s1 = self._make_layer(
                LG_Swin_TNT_block, channels[0], channels[1]-channels[0], num_blocks[1], (ih // 4, iw // 4),
                i_layer = 1, num_head=num_heads[1])
            self.s2 = self._make_layer(
                LG_Swin_TNT_block, channels[1], channels[2]-channels[1], num_blocks[2], (ih // 8, iw // 8),
                i_layer = 2, num_head=num_heads[2])
            self.s3 = self._make_layer(
                LG_Swin_TNT_block, channels[2], channels[3]-channels[2], num_blocks[3], (ih // 16, iw // 16),
                i_layer = 3, num_head=num_heads[3])
            self.s4 = self._make_layer(
                LG_Swin_TNT_block, channels[3], channels[4]-channels[3], num_blocks[4], (ih // 32, iw // 32),
                i_layer = 4, num_head=num_heads[4])
            
            self.pool_s1 = nn.MaxPool2d(3, 2, 1)
            self.pool_s2 = nn.MaxPool2d(3, 2, 1)
            self.pool_s3 = nn.MaxPool2d(3, 2, 1)
            self.pool_s4 = nn.MaxPool2d(3, 2, 1)

        else:
            self.s0 = self._make_layer(
                conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
            self.s1 = self._make_layer(
                LG_Swin_TNT_block, channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4),
                i_layer = 1, num_head=num_heads[1])
            self.s2 = self._make_layer(
                LG_Swin_TNT_block, channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8),
                i_layer = 2, num_head=num_heads[2])
            self.s3 = self._make_layer(
                LG_Swin_TNT_block, channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16),
                i_layer = 3, num_head=num_heads[3])
            self.s4 = self._make_layer(
                LG_Swin_TNT_block, channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32),
                i_layer = 4, num_head=num_heads[4])

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)
        
    def forward(self, x):
        if self.deep_dense:
            # x = B,C,H,W
            x = self.s0(x)
            x = torch.cat((self.s1(x), self.pool_s1(x)), 1)
            x = torch.cat((self.s2(x), self.pool_s2(x)), 1)
            x = torch.cat((self.s3(x), self.pool_s3(x)), 1)
            x = torch.cat((self.s4(x), self.pool_s4(x)), 1)
        else:
            x = self.s0(x)
            x = self.s1(x)
            x = self.s2(x)
            x = self.s3(x)
            x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size,
                    i_layer = 0, num_head=0,
                    ):
        layers = nn.ModuleList([])
        
        drop_path=self.dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])]
        for i in range(depth):
            window_size = 7
            if i == 0:
                downsample = True
            else:
                downsample = False
            
            if block == LG_Swin_TNT_block:
                layers.append(block(inp, oup, image_size, downsample=downsample, 
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                                    num_head=num_head, window_size = window_size,
                                    qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                    contextual_rpe=self.contextual_rpe,
                                    contextual_input = self.contextual_input,
                                    T_in_T = self.T_in_T,
                                    ))
            elif block == conv_3x3_bn:
                layers.append(block(inp, oup, image_size, downsample=downsample))
            
            '''
                if block == LG_Swin_TNT_block:
                    layers.append(block(inp, oup, image_size, downsample=True, shift_size=shift_size,
                                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                        num_head=num_head, window_size = window_size,
                                        
                                        qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                        
                                        i=i,
                                        
                                        # the following argumennt is for contextual RPE
                                        contextual_rpe=self.contextual_rpe,
                                        contextual_input = self.contextual_input,
                                        
                                        # the following argumennt is for Transformer in Transformer
                                        T_in_T = self.T_in_T,
                                        ))
                elif block == conv_3x3_bn:
                    layers.append(block(inp, oup, image_size, downsample=True))

            else:
                if block == LG_Swin_TNT_block:
                    layers.append(block(oup, oup, image_size, shift_size=shift_size,
                                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                        num_head=num_head, window_size = window_size,
                                        
                                        i=i,
                                        
                                        # the following argumennt is for contextual RPE
                                        contextual_rpe=self.contextual_rpe,
                                        contextual_input = self.contextual_input,
                                        
                                        # the following argumennt is for Transformer in Transformer
                                        T_in_T = self.T_in_T,
                                        ))
                elif block == conv_3x3_bn:
                    layers.append(block(oup, oup, image_size, downsample=False))
            '''

        return nn.Sequential(*layers)
