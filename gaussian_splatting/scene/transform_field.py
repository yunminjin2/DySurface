import torch
from torch import nn
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torch import Tensor
# from minkunet import MinkUNet
from gaussian_splatting.mink.minkunet import MinkUNet, MinkUNetCustom
from gaussian_splatting.scene.voxel_util import VoxelQuery


def move2center(pcd):
    min_coords, _  = torch.min(pcd, 0)
    max_coords, _  = torch.max(pcd, 0)
    range_coords = max_coords - min_coords
    normalized_pcd = ((pcd - min_coords) / range_coords) * 2 - 1
    center = torch.mean(normalized_pcd, 0)

    return normalized_pcd - center

def scale_tensor(
    dat, inp_scale=None, tgt_scale=None
):
    if inp_scale is None:
        inp_scale = (-0.5, 0.5)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    assert tgt_scale[1] > tgt_scale[0] and inp_scale[1] > inp_scale[0]
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat.clamp(tgt_scale[0] + 1e-6, tgt_scale[1] - 1e-6)



class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        
        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
    
    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        
        return torch.cat(out, -1)


class TransformField(nn.Module):
    def __init__(self, W1=64, W2=256, out_dim=10, voxelsize=0.005, pos_emb=6, time_emb=6, args=None):
        super(TransformField, self).__init__()
        self.W1 = W1
        self.W2 = W2
        self.out_dim = out_dim
        pos_emb = args.position_emb_level if args else pos_emb
        time_emb = args.time_emb_level if args else time_emb
        self.in_channels_xyz = pos_emb * 2 * 3 + 3 
        self.in_channels_t = time_emb * 2 + 1

        self.front_net_in_dim = self.in_channels_xyz
        
        self.voxel_size = voxelsize

        self.embedding_xyz = Embedding(3, pos_emb)
        self.embedding_t = Embedding(1, time_emb)

        self.front_net = nn.Sequential(
            nn.Linear(self.front_net_in_dim, self.W1), nn.ReLU(True),
            nn.Linear(self.W1, self.W1)
        )
        self.back_net = nn.Sequential(
            nn.Linear(self.W1+64, self.W1), nn.ReLU(True),
            nn.Linear(self.W1, self.W1), nn.ReLU(True),
            nn.Linear(self.W1, self.W1)
        )

        self.unet = MinkUNet(self.front_net_in_dim, vsize=self.voxel_size, cr=1, cs=[32,64,128,128,64,32])

        self.network = nn.Sequential(
            nn.Linear(self.in_channels_t+self.front_net_in_dim+self.W1, self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.W2), nn.ReLU(True),
            nn.Linear(self.W2 + self.in_channels_t+self.front_net_in_dim+self.W1, self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.out_dim)
        )
   

    def forward_frontnet(self, x):
        return self.front_net(x)

    def voxelize(self, _xyz, x_emb):
        pc = torch.round(_xyz / self.voxel_size).type(torch.int32).cpu().numpy()
        _, inds, inverse_map = sparse_quantize(pc, return_index=True, return_inverse=True)
        
        coord_,feat_ = (torch.cat([_xyz[inds], torch.zeros(_xyz[inds].shape[0],1).to('cuda')], 1), x_emb[inds])
        inputpcd = SparseTensor(feats=feat_,coords=coord_)

        return inputpcd, inds, inverse_map
    
    def devoxelize(self, out, inverse_map):
        torch.zeros([inverse_map.shape[0], 3]).to('cuda')
        return out[inverse_map]


    def forward_backnet(self, x_emb, t_emb, front_feat, _pcd):
        back_feat = self.back_net(torch.cat([front_feat, _pcd],-1))
        _x = torch.cat([x_emb, t_emb, back_feat], -1)

        _xx = _x
        for idx, n in enumerate(self.network):
            if idx != 4:
                h = n(_x)
            else:
                h = n(torch.cat([_x, _xx], -1))
            _x = h
        out = _x

        return out, back_feat

    def forward(self, x, t, return_feat=False):
        x_emb = self.embedding_xyz(x)
        if len(t.shape) == 0:
            t = t[None,]
        t_emb = self.embedding_t(t)


        if len(t_emb.shape) == 1:
            t_emb = t_emb[None,]
        if x_emb.shape[0] != t_emb.shape[0]:
            t_emb = t_emb.repeat(x_emb.shape[0], 1)
        
        _xyz = x_emb[:,:3]
        # center_xyz = move2center(_xyz)
        # x_emb = torch.cat([c, x_emb,], -1)
        
        front_feat = self.forward_frontnet(x_emb)

        inputpcd, inds, inverse_map = self.voxelize(_xyz, x_emb)
        vox_out = self.unet(inputpcd)             # N*32
        sparse_gs_vox = SparseTensor(feats=vox_out, coords=inputpcd.C)
        _pcd = self.devoxelize(vox_out, inverse_map)
        
        out, back_feat = self.forward_backnet(x_emb, t_emb, front_feat, _pcd)

        if return_feat:
            return out, inds, sparse_gs_vox
        return out
        

class TransformFieldRay(nn.Module):
    def __init__(self, W1=64, W2=256, out_dim=10, voxelsize=0.005, pos_emb=6, time_emb=6, color_emb=4, scene_radius=1.0, config=None, args=None, ablation_include_voxgs=True):
        super(TransformFieldRay, self).__init__()
        self.W1 = W1
        self.W2 = W2
        self.out_dim = out_dim
        self.config = config
        pos_emb = args.position_emb_level if args else pos_emb
        time_emb = args.time_emb_level if args else time_emb
        self.scene_radius = scene_radius
        self.in_channels_xyz = pos_emb * 2 * 3 + 3 
        self.in_channels_t = time_emb * 2 + 1
        self.in_channels_color = color_emb * 2 * 3 + 3

        self.front_net_in_dim = self.in_channels_xyz 
        self.ablation_include_voxgs = ablation_include_voxgs
        self.voxel_size = voxelsize

        self.embedding_xyz = Embedding(3, pos_emb)
        self.embedding_t = Embedding(1, time_emb)
        self.embedding_color = Embedding(3, color_emb)

        self.sparsify_in_dim = self.front_net_in_dim + self.in_channels_t
        

        self.front_net = nn.Sequential(
            nn.Linear(self.front_net_in_dim + self.in_channels_t, self.W1), nn.ReLU(True),
            nn.Linear(self.W1, self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.W2)
        )
        self.sparsify_net = nn.Sequential(
            nn.Linear(self.W2 + (self.in_channels_color if self.ablation_include_voxgs else 0), self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.W2)
        )

        self.unet = MinkUNetCustom(
            num_feats=self.front_net_in_dim, 
            vsize=self.voxel_size, 
            cr=1, 
            cs=[16, 32, 64, 32],
            out_channel=self.W1 // 2
        )

        self.back_net = nn.Sequential(
            # nn.Linear(3 + self.in_channels_xyz + self.W2, self.W1), nn.ReLU(True),
            nn.Linear(self.W2 + self.W2, self.W1), nn.ReLU(True),
            nn.Linear(self.W1, self.W1), nn.ReLU(True),
            nn.Linear(self.W1, self.W1)
        )

        self.network = nn.Sequential(
            nn.Linear(self.in_channels_t+self.front_net_in_dim+self.W1, self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.W2), nn.ReLU(True),
            nn.Linear(self.W2 + self.in_channels_t + self.front_net_in_dim + self.W1, self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.out_dim)
        )

        self.voxel_query = VoxelQuery(voxel_size=self.voxel_size)

    def forward_frontnet(self, x):
        return self.front_net(x)

    def voxelize(self, _xyz, x_emb):
        pc = torch.round(_xyz / self.voxel_size).type(torch.int32).cpu().numpy()
        voxel_coords, inds, inverse_map = sparse_quantize(pc, return_index=True, return_inverse=True)
        coord_, feat_ = (
            torch.cat([_xyz[inds], torch.zeros(_xyz[inds].shape[0],1).to('cuda')], 1), 
            x_emb[inds]
        )
        
        voxel_coords = torch.from_numpy(voxel_coords * self.voxel_size).float().cuda()
        gs_pcd = SparseTensor(feats=feat_, coords=coord_)
    
        return voxel_coords, gs_pcd, inds, inverse_map
    
    def devoxelize(self, out, inverse_map):
        torch.zeros([inverse_map.shape[0], 3]).to('cuda')
        return out[inverse_map]


    def forward_backnet(self, x_emb, t_emb, front_feat, _pcd):
        back_feat = self.back_net(torch.cat([front_feat, _pcd],-1))
        _x = torch.cat([x_emb, t_emb, back_feat], -1)

        _xx = _x
        for idx, n in enumerate(self.network):
            if idx != 4:
                h = n(_x)
            else:
                h = n(torch.cat([_x, _xx], -1))
            _x = h
        out = _x

        return out, back_feat

    def forward(self, ray_xyz, gs_asset, t, return_feat=False, return_grad=False):
        # points_d_ = (ray_xyz).clamp(-1.4, 1.4)
        # points_d = scale_anything(points_d_, (-1.4, 1.4), (0, 1))
                        
        x_emb = self.embedding_xyz(ray_xyz)
        if len(t.shape) == 0:
            t = t[None,]
        t_emb = self.embedding_t(t)

        if len(t_emb.shape) == 1:
            t_emb = t_emb[None,]
        if x_emb.shape[0] != t_emb.shape[0]:
            t_emb = t_emb.repeat(x_emb.shape[0], 1)
        

        d_g_xyz = gs_asset['xyz'] + gs_asset['delta']  # deformed gaussian positions
        d_g_feat = gs_asset['feature']
        
        if gs_asset['vox'] is not None:
            sparse_gs_pcd = gs_asset['vox']  
        else:
            voxel_gs, sparse_gs_pcd, inds, gs_inverse_inds = self.voxelize(d_g_xyz, torch.cat([d_g_feat], dim=-1))
            
        q_asset = gs_asset['GS'].query_point_attributes(ray_xyz, v_pcd=sparse_gs_pcd, cal_normal=False, distance_threshhold=self.config.gs_distance_threshold)
        
        q_feat = q_asset['feat']
        q_feat = self.embedding_color(q_feat) # color embedding
        
        front_feat = self.forward_frontnet(torch.cat([x_emb, t_emb], dim=-1)) # , sdf_field['sdf'][:, None], sdf_field['sdf_grad']], -1))
        eta_feat = front_feat
        if self.ablation_include_voxgs:
            eta_feat = torch.cat([front_feat, q_feat], dim=-1)

        ray_feat = self.sparsify_net(eta_feat) # W1 + in_channels_color

        out, back_feat = self.forward_backnet(x_emb, t_emb, front_feat, ray_feat)
        ray_delta = out[:, :3]
        
        return {
            'delta': ray_delta,
            'feature': back_feat
        }
