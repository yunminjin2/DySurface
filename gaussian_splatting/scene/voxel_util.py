import torch

class VoxelQuery:
    def __init__(self, voxel_size, dim_size=100000):
        """
        초기화 시에는 고정 파라미터만 설정합니다.
        """
        self.voxel_size = voxel_size
        self.dim_size = dim_size
        
        # 상태 저장을 위한 변수 (Placeholder)
        self.sorted_db_hash = None
        self.sort_idx = None
        self.voxel_features = None
        self.device = None

    def update(self, voxel_world_coords, voxel_features):
        """
        매 프레임 변형된(Deformed) GS 좌표와 피처로 DB를 갱신합니다.
        
        Args:
            voxel_world_coords: (M, 4) [batch, x, y, z] - Deformed XYZ
            voxel_features: (M, C) - Deformed Features (or original features)
        """
        self.device = voxel_world_coords.device
        self.voxel_features = voxel_features # 참조만 복사 (메모리 절약)
        
        # 1. Quantize & Hash
        grid_coords = self._quantize(voxel_world_coords)
        db_hash = self._hash(grid_coords)
        
        # 2. Sort (이 과정이 매번 필수)
        self.sorted_db_hash, self.sort_idx = torch.sort(db_hash)

    def query(self, ray_world_coords):
        """
        Ray 좌표를 받아 매칭 수행
        """
        if self.sorted_db_hash is None:
            raise RuntimeError("update() must be called before query()")

        # 1. Query Hashing
        query_grid = self._quantize(ray_world_coords)
        query_hash = self._hash(query_grid)
        
        # 2. Search
        idx_in_sorted = torch.searchsorted(self.sorted_db_hash, query_hash)
        idx_in_sorted = torch.clamp(idx_in_sorted, max=self.sorted_db_hash.shape[0] - 1)
        
        # 3. Validation
        matched_hash = self.sorted_db_hash[idx_in_sorted]
        is_matched = (matched_hash == query_hash)
        
        # 4. Gather
        output_features = torch.zeros((ray_world_coords.shape[0], self.voxel_features.shape[1]), 
                                      device=self.device, dtype=self.voxel_features.dtype)
        
        original_indices = self.sort_idx[idx_in_sorted]
        output_features[is_matched] = self.voxel_features[original_indices[is_matched]]
        
        return output_features, is_matched

    def _quantize(self, coords):
        if coords.shape[1] != 4:
            coords = torch.cat([coords, torch.zeros((coords.shape[0], 1), device=coords.device)], dim=1)
        xyz = coords[:, :3]
        b = coords[:, 3].long() 

        xyz_int = torch.round(xyz / self.voxel_size).long()
    
        return torch.cat([b.unsqueeze(1), xyz_int], dim=1)

    def _hash(self, coords):
        return (coords[:, 0] * self.dim_size**3 + 
                coords[:, 1] * self.dim_size**2 + 
                coords[:, 2] * self.dim_size + 
                coords[:, 3])