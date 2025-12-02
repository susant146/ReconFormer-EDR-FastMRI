"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from data import transforms
from data import transformsPB as Tpb
from models.RS_attention import RPTL, PatchEmbed, PatchUnEmbed
from torch.nn import functional as F
import numpy as np

class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self):
        super(DataConsistencyInKspace, self).__init__()

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def data_consistency(self,k, k0, mask):
        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        out = (1 - mask) * k + mask * k0
        return out

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        x = x.permute(0, 2, 3, 1)
        k0 = k0.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)

        k = Tpb.fft2c_new(x)     #k = transforms.fft2(x)
        # print('k shape: ', k.shape)
        out = self.data_consistency(k, k0, mask)
        x_res = Tpb.ifft2c_new(out)  #x_res = transforms.ifft2(out)
        # print('x_res shape: ', x_res.shape)
        x_res = x_res.permute(0, 3, 1, 2)

        return x_res

class RFB(nn.Module):
    """
    ReconFormer Block
    """

    def __init__(self,img_size,nf,depth,num_head,window_size,mlp_ratio,use_checkpoint,
                 resi_connection, down=True, up_scale=None, down_scale=None):
        super(RFB, self).__init__()

        if down:
            img_size = img_size // down_scale
        else:
            img_size = int(img_size * up_scale)
        embed_dim = nf
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=1, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=1, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm)

        self.RPTL1 = RPTL(dim=embed_dim,
                          input_resolution=(patches_resolution[0],
                                            patches_resolution[1]),
                          depth=depth,
                          num_heads=num_head,
                          window_size=window_size,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=True, qk_scale=None,
                          drop=0., attn_drop=0.,
                          drop_path=0.,  # no impact on SR results
                          norm_layer=nn.LayerNorm,
                          downsample=None,
                          use_checkpoint=use_checkpoint[0],
                          img_size=img_size,
                          patch_size=1,
                          resi_connection=resi_connection,
                          rec_att=True,
                          )
        self.RPTL2 = RPTL(dim=embed_dim,
                           input_resolution=(patches_resolution[0],
                                             patches_resolution[1]),
                           depth=depth,
                           num_heads=num_head,
                           window_size=window_size,
                           mlp_ratio=mlp_ratio,
                           qkv_bias=True, qk_scale=None,
                           drop=0., attn_drop=0.,
                           drop_path=0.,  # no impact on SR results
                           norm_layer=nn.LayerNorm,
                           downsample=None,
                           use_checkpoint=use_checkpoint[1],
                           img_size=img_size,
                           patch_size=1,
                           resi_connection=resi_connection,
                           rec_att=True,
                           shift=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden, h1_att, h2_att):

        x_size = (hidden.shape[2], hidden.shape[3])
        hidden = self.patch_embed(hidden)

        hidden = (hidden, h1_att)  # {'x': hi, 'p_att': c1_att}
        h1 = self.RPTL1(hidden, x_size)
        h1_att = h1[1]
        h1 = h1[0]

        h1 = (h1, h2_att)  # {'x': ic1, 'p_att': c2_att}
        h2 = self.RPTL2(h1, x_size)
        h2_att = h2[1]
        h2 = h2[0]

        h2 = self.norm(h2)  # B L C
        h2 = self.patch_unembed(h2, x_size)

        return h2, h1_att, h2_att
    
# ----------- Modified HERE ----------------------------
class ResidualBlock(nn.Module):
    """Improved residual block with instance normalization"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)

class ChannelAttention(nn.Module):
    """Lightweight channel attention mechanism"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels//reduction),
            nn.SiLU(),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)
    

class TransBlock_UC(nn.Module):
    """Enhanced undercomplete block with residual learning"""
    def __init__(self, in_channels=2, out_channels=2, nf=64, down_scale=2, img_size=256,
                num_head=6, depth=6, window_size=7, mlp_ratio=2.,
                use_checkpoint=(False,False), resi_connection='1conv'):
        super().__init__()

        # Configurable downsampling
        self.down_scale = down_scale
        kernel1, stride1 = 3, 1
        kernel2, stride2 = (4,2) if down_scale==2 else (3,1)

        # Enhanced encoder with residual blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel1, stride1, 1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.SiLU(inplace=True),
            ResidualBlock(nf),
            ChannelAttention(nf),
            nn.Conv2d(nf, nf, kernel2, stride2, 1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.SiLU(inplace=True)
        )

        self.RFB = RFB(img_size, nf, depth, num_head, window_size, mlp_ratio,
                      use_checkpoint, resi_connection, down=True, down_scale=down_scale)

        # Improved decoder with pixel shuffle
        self.decoder = nn.Sequential(
            nn.Conv2d(nf, nf*(down_scale**2), 3, padding=1, bias=False),
            nn.PixelShuffle(down_scale),
            nn.InstanceNorm2d(nf),
            nn.SiLU(inplace=True),
            ResidualBlock(nf),
            ChannelAttention(nf),
            nn.Conv2d(nf, out_channels, 3, padding=1, bias=True)
        )

        self.DC_layer = DataConsistencyInKspace()

    def forward(self, x, hidden=None, h1_att=None, h2_att=None, k0=None, mask=None):
        if hidden is None:
            hidden = self.encoder(x)
        else:
            h2, h1_att, h2_att = self.RFB(hidden, h1_att, h2_att)
            hidden = self.encoder(x) + h2  # Residual connection

        out = self.decoder(hidden)
        out = self.DC_layer(out, k0, mask)
        return out, hidden, h1_att, h2_att

class TransBlock_OC(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, nf=64, up_scale=2, img_size=256,
                 num_head=6, depth=6, window_size=7, mlp_ratio=2.,
                 use_checkpoint=(False,False), resi_connection='1conv'):
        super().__init__()
        
        # Ensure integer upscaling factor
        self.up_scale = int(up_scale)

        # Fixed encoder with explicit integer casting
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, padding=1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.SiLU(inplace=True),
            nn.Conv2d(nf, int(nf * (self.up_scale ** 2)), 3, padding=1, bias=False),
            nn.PixelShuffle(self.up_scale),
            nn.InstanceNorm2d(nf),
            nn.SiLU(inplace=True),
            ResidualBlock(nf),
            ChannelAttention(nf)
        )

        self.RFB = RFB(img_size, nf, depth, num_head, window_size, mlp_ratio,
                      use_checkpoint, resi_connection, down=False, up_scale=self.up_scale)

        # Fixed decoder with scale factor validation
        self.decoder = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=1/self.up_scale) if self.up_scale > 1 else nn.Identity(),
            nn.InstanceNorm2d(nf),
            nn.SiLU(inplace=True),
            ResidualBlock(nf),
            ChannelAttention(nf),
            nn.Conv2d(nf, out_channels, 3, padding=1, bias=True)
        )

        self.DC_layer = DataConsistencyInKspace()


    def forward(self, x, hidden=None, h1_att=None, h2_att=None, k0=None, mask=None):
        if hidden is None:
            hidden = self.encoder(x)
        else:
            h2, h1_att, h2_att = self.RFB(hidden, h1_att, h2_att)
            hidden = self.encoder(x) + h2  # Residual connection

        out = self.decoder(hidden)
        out = self.DC_layer(out, k0, mask)
        return out, hidden, h1_att, h2_att

# ----------- CHANGE HERE --------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels, bias=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class RefineModule(nn.Module):
    """
    Lightweight and efficient Refine Module with:
    - Depthwise Separable Convolutions
    - Residual connections
    - Group Normalization
    - SiLU activation
    """

    def __init__(self, in_channels, nf, out_channels):
        super(RefineModule, self).__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels, nf)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=nf)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = DepthwiseSeparableConv(nf, nf)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=nf)
        self.act2 = nn.SiLU(inplace=True)

        self.conv3 = DepthwiseSeparableConv(nf, nf)
        self.conv4 = nn.Conv2d(nf, out_channels, kernel_size=3, padding=1, bias=True)

        self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True) if in_channels != out_channels else nn.Identity()
        self.DC_layer = DataConsistencyInKspace()

    def forward(self, x, k0=None, mask=None):
        identity = self.res_proj(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = x + identity  # Lightweight residual connection
        return self.DC_layer(x, k0, mask)
# ----------------------------------------------------------------


class ReconFormer_EDR(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_ch=(64, 64, 64), down_scales=(2,1,1.5),
                 num_iter=5, img_size=256, num_heads=(6,6,6), depths=(6,6,6), window_sizes=(8,8,8),
                 resi_connection ='1conv', mlp_ratio=2.,
                 use_checkpoint = (False,False,False,False,False,False)):
        super(ReconFormer_EDR, self).__init__()

        self.num_iter = num_iter

        self.block1 = TransBlock_UC(in_channels=in_channels, out_channels=out_channels, nf=num_ch[0],
                                    down_scale=down_scales[0], num_head=num_heads[0], depth=depths[0],
                                    img_size=img_size, window_size=window_sizes[0], mlp_ratio=mlp_ratio,
                                    use_checkpoint=(use_checkpoint[0],use_checkpoint[1]),
                                    resi_connection =resi_connection)

        self.block2 = TransBlock_UC(in_channels=in_channels, out_channels=out_channels, nf=num_ch[1],
                                    down_scale=down_scales[1], num_head=num_heads[1], depth=depths[1],
                                    img_size=img_size,window_size=window_sizes[1], mlp_ratio=mlp_ratio,
                                    use_checkpoint=(use_checkpoint[2],use_checkpoint[3]),
                                    resi_connection = resi_connection)

        self.block3 = TransBlock_OC(in_channels=in_channels, out_channels=out_channels, nf=num_ch[2],
                                    up_scale=down_scales[2], num_head=num_heads[2], depth=depths[2],
                                    img_size=img_size,window_size=window_sizes[2], mlp_ratio=mlp_ratio,
                                    use_checkpoint=(use_checkpoint[4],use_checkpoint[5]),
                                    resi_connection =resi_connection)

        self.RM = RefineModule(in_channels=int(out_channels * 3),nf=num_ch[2],out_channels=out_channels)

    def forward(self, x, k0=None, mask=None):
        outputs = []
        for i in range(self.num_iter):
            if i ==0:
                x1, h1, _, _ = self.block1(x,  k0=k0, mask=mask)
                x2, h2, _, _ = self.block2(x1, k0=k0, mask=mask)
                x3, h3, _, _ = self.block3(x2, k0=k0, mask=mask)
            elif i ==1:
                x = outputs[-1]
                x1, h1, b1_c1_att, b1_c2_att = self.block1(x,  hidden=h1, k0=k0, mask=mask)
                x2, h2, b2_c1_att, b2_c2_att = self.block2(x1, hidden=h2, k0=k0, mask=mask)
                x3, h3, b3_c1_att, b3_c2_att = self.block3(x2, hidden=h3, k0=k0, mask=mask)
            else:
                x = outputs[-1]
                x1, h1, b1_c1_att, b1_c2_att = self.block1(x,  hidden=h1, h1_att=b1_c1_att, h2_att=b1_c2_att, k0=k0, mask=mask)
                x2, h2, b2_c1_att, b2_c2_att = self.block2(x1, hidden=h2, h1_att=b2_c1_att, h2_att=b2_c2_att, k0=k0, mask=mask)
                x3, h3, b3_c1_att, b3_c2_att = self.block3(x2, hidden=h3, h1_att=b3_c1_att, h2_att=b3_c2_att, k0=k0, mask=mask)
            out = torch.cat((x1, x2, x3), dim=1)
            out = self.RM(out, k0, mask)
            outputs.append(out)

        return outputs[-1]

if __name__ == '__main__':
    x_ = torch.randn((1, 128, 100, 100))
    model = ReconFormer_EDR()
