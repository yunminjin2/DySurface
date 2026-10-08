
from .voxelBlock_fcgf import DownVoxelStage, UpVoxelStage, ResidualBlock
from .utils import initial_voxelize, voxel_to_point
import torch
import torch.nn as nn

from torchsparse import PointTensor
import torchsparse.nn as spnn

class MinkUNet(nn.Module):
    def __init__(self,num_feats,vsize=0.05,**kwargs):
        super(MinkUNet, self).__init__()

        self.input_channel = num_feats
        self.vsize = vsize
        self.cr = kwargs.get('cr')
        self.cs = torch.Tensor(kwargs.get('cs'))
        self.cs = (self.cs*self.cr).int()
        self.output_channel = 64
        
        # self.output_channel = self.cs[5].item()

        ''' voxel branch '''
        # self.voxel_stem = nn.Sequential(
        #     spnn.Conv3d(1, self.cs[0], kernel_size=5, stride=1),
        #     spnn.BatchNorm(self.cs[0]), spnn.ReLU(True),
        #     spnn.Conv3d(self.cs[0], self.cs[0], kernel_size=3, stride=1),
        #     spnn.BatchNorm(self.cs[0]), spnn.ReLU(True))
        self.voxel_init = DownVoxelStage(self.input_channel,self.cs[0],
                                      b_kernel_size=5,b_stride=1,b_dilation=1,
                                      kernel_size=3,stride=1,dilation=1)
        self.voxel_down1 = DownVoxelStage(self.cs[0], self.cs[1],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_down2 = DownVoxelStage(self.cs[1], self.cs[2],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        # self.voxel_down4 = DownVoxelStage(self.cs[2], self.cs[3],
        #                               b_kernel_size=3, b_stride=2, b_dilation=1,
        #                               kernel_size=3, stride=1, dilation=1)
        self.voxel_bottle = nn.Sequential(
            # BasicDeconvolutionBlock(self.cs[3], self.cs[4],
            #                         kernel_size=3, stride=2),
            ResidualBlock(self.cs[2], self.cs[3],
                          kernel_size=3, stride=1)
        )
        self.voxel_up1 = UpVoxelStage(self.cs[3],self.cs[4],self.cs[2],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_up2 = UpVoxelStage(self.cs[4],self.cs[5],self.cs[1],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        # self.voxel_up3 = UpVoxelStage_withoutres(self.cs[6],self.cs[7],self.cs[0],
        #                                          kernel_size=3,stride=1)
        self.voxel_final = spnn.Conv3d(self.cs[5],self.output_channel,
                                       kernel_size=1,stride=1,bias=True)

        

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    
    def forward(self,lidar):
        points = PointTensor(lidar.F,lidar.C.float())
        # print(points.F.shape)
        v0 = initial_voxelize(points,self.vsize)
        
        voxel_s1 = self.voxel_init(v0)
        voxel_s2 = self.voxel_down1(voxel_s1)
        voxel_s4 = self.voxel_down2(voxel_s2)
        
        voxel_s4_tr = self.voxel_bottle(voxel_s4)
        
        voxel_s2_tr = self.voxel_up1(voxel_s4_tr, voxel_s4)
        voxel_s1_tr = self.voxel_up2(voxel_s2_tr, voxel_s2)
        voxel_out_final = self.voxel_final(voxel_s1_tr)

        out = voxel_to_point(voxel_out_final, points)
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True))

        return out




class MinkUNetCustom(nn.Module):
    def __init__(
        self,
        num_feats,
        out_channel=64,
        vsize=0.05,
        num_down=None,
        normalize_output=True,
        **kwargs,
    ):
        super(MinkUNetCustom, self).__init__()

        self.input_channel = num_feats
        self.vsize = vsize
        self.cr = kwargs.get('cr', 1.0)
        cs = kwargs.get('cs', None)
        if cs is None:
            raise ValueError("MinkUNetCustom requires kwargs['cs'] (channel schedule list).")

        cs_t = torch.as_tensor(cs, dtype=torch.float32)
        cs_t = (cs_t * float(self.cr)).round().to(torch.int64)
        self.cs = cs_t

        # Infer number of down/up levels if not provided.
        # Expected channel layout (same convention as MinkUNet):
        #   enc:  cs[0] (init), cs[1..num_down] (downs)
        #   bottle_out: cs[num_down+1]
        #   dec: cs[num_down+2 .. num_down+1+num_down] (ups)
        # Total length = 2*num_down + 2
        if num_down is None:
            if (len(self.cs) - 2) % 2 != 0:
                raise ValueError(
                    f"Cannot infer num_down from cs of length {len(self.cs)}. "
                    "Expected len(cs) = 2*num_down + 2 (even, >= 4)."
                )
            num_down = (len(self.cs) - 2) // 2
        self.num_down = int(num_down)
        if self.num_down < 1:
            raise ValueError(f"num_down must be >= 1, got {self.num_down}.")

        expected_len = 2 * self.num_down + 2
        if len(self.cs) != expected_len:
            raise ValueError(
                f"cs length mismatch: got len(cs)={len(self.cs)}, expected {expected_len} for num_down={self.num_down}. "
                "(Expected layout: [init] + [downs]*num_down + [bottle_out] + [ups]*num_down)"
            )

        self.output_channel = int(out_channel)
        self.normalize_output = bool(normalize_output)

        enc_channels = [int(self.cs[i].item()) for i in range(self.num_down + 1)]
        bottle_out_ch = int(self.cs[self.num_down + 1].item())
        dec_channels = [
            int(self.cs[self.num_down + 2 + i].item()) for i in range(self.num_down)
        ]

        # Encoder
        self.voxel_init = DownVoxelStage(
            self.input_channel,
            enc_channels[0],
            b_kernel_size=5,
            b_stride=1,
            b_dilation=1,
            kernel_size=3,
            stride=1,
            dilation=1,
        )
        self.voxel_downs = nn.ModuleList()
        for i in range(1, self.num_down + 1):
            self.voxel_downs.append(
                DownVoxelStage(
                    enc_channels[i - 1],
                    enc_channels[i],
                    b_kernel_size=3,
                    b_stride=2,
                    b_dilation=1,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                )
            )

        # Bottleneck (same spatial resolution as deepest level)
        self.voxel_bottle = nn.Sequential(
            ResidualBlock(enc_channels[-1], bottle_out_ch, kernel_size=3, stride=1)
        )

        # Decoder
        self.voxel_ups = nn.ModuleList()
        for i in range(self.num_down):
            in_ch = bottle_out_ch if i == 0 else dec_channels[i - 1]
            skip_ch = enc_channels[-1 - i]
            out_ch = dec_channels[i]
            self.voxel_ups.append(
                UpVoxelStage(
                    in_ch,
                    out_ch,
                    skip_ch,
                    b_kernel_size=3,
                    b_stride=2,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                )
            )

        self.voxel_final = spnn.Conv3d(
            dec_channels[-1], self.output_channel, kernel_size=1, stride=1, bias=True
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, lidar):
        points = PointTensor(lidar.F, lidar.C.float())
        v0 = initial_voxelize(points, self.vsize)

        x = self.voxel_init(v0)
        enc_feats = [x]
        for down in self.voxel_downs:
            x = down(x)
            enc_feats.append(x)

        x = self.voxel_bottle(x)

        # Each UpVoxelStage concatenates (x, skip) BEFORE upsampling.
        # Therefore skip must be at the same resolution as x.
        for i, up in enumerate(self.voxel_ups):
            skip = enc_feats[-1 - i]
            x = up(x, skip)

        voxel_out_final = self.voxel_final(x)
        out = voxel_to_point(voxel_out_final, points)
        if self.normalize_output:
            out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-12)
        return out


# -------------------------
# Usage examples
# -------------------------
# 1) Same topology as MinkUNet (fixed 2 downs / 2 ups), but choose output dim.
#    Here cs has length 6 => num_down inferred as (6-2)/2 = 2
#
# model = MinkUNetCustom(
#     num_feats=F_in,
#     out_channel=64,
#     vsize=0.05,
#     cs=[16, 32, 64, 128, 64, 32],
#     cr=1.0,
# )
#
# 2) Explicit depth (num_down=3). Then cs must have length 2*num_down+2 = 8.
#    Layout: [init] + [downs]*3 + [bottle_out] + [ups]*3
#
# model = MinkUNetCustom(
#     num_feats=F_in,
#     out_channel=32,
#     num_down=3,
#     vsize=0.02,
#     cs=[16, 32, 64, 96, 128, 96, 64, 32],
#     cr=1.0,
#     normalize_output=False,
# )
#
# 3) Channel multiplier (cr) to scale a base cs schedule.
#
# model = MinkUNetCustom(
#     num_feats=F_in,
#     out_channel=64,
#     vsize=0.05,
#     cs=[16, 32, 64, 128, 64, 32],
#     cr=0.5,  # channels become ~[8,16,32,64,32,16]
# )


