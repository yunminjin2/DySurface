import torch
import torch.nn.functional as F


def get_normals(vertices, faces):
    '''
    vertices b n 3
    faces f 3
    '''
    
    if len(vertices.shape) == 2:
        vertices = vertices.unsqueeze(0)
    verts_normals = torch.zeros_like(vertices)

    vertices_faces = vertices[:, faces]  # b f 3 3

    verts_normals.index_add_(
        1,
        faces[:, 1],
        torch.cross(
            vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
            vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 2],
        torch.cross(
            vertices_faces[:, :, 0] - vertices_faces[:, :, 2],
            vertices_faces[:, :, 1] - vertices_faces[:, :, 2],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 0],
        torch.cross(
            vertices_faces[:, :, 1] - vertices_faces[:, :, 0],
            vertices_faces[:, :, 2] - vertices_faces[:, :, 0],
            dim=2,
        ),
    )

    verts_normals = F.normalize(verts_normals, p=2, dim=2, eps=1e-6)
    # verts_normals = mynormalize(verts_normals, p=2, dim=2, eps=1e-6)

    return verts_normals


def get_edges(verts, faces):
    V = verts.shape[0]
    f = faces.shape[0]
    device = verts.device
    v0, v1, v2 = faces.chunk(3, dim=1)
    e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

    edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
    edges, _ = edges.sort(dim=1)
    edges_hash = V * edges[:, 0] + edges[:, 1]
    u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
    sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
    unique_mask = torch.ones(edges_hash.shape[0], dtype=torch.bool, device=device)
    unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
    edges = torch.stack([u // V, u % V], dim=1)

    faces_to_edges = inverse_idxs.reshape(3, f).t()

    return edges, faces_to_edges


def laplacian_cot(verts, faces):
    '''
    verts n 3
    faces f 3
    '''
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]  # f 3 3
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot = cot / 4.0

    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]

    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    L = L + L.t()

    idx = faces.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas

def get_adjacency(verts, faces):
    V = verts.shape[0]
    device = verts.device
    
    edges, faces_to_edges = get_edges(verts, faces)

    e0, e1 = edges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    return A

def compute_laplacian(verts, faces):
    # first compute edges

    V = verts.shape[0]
    device = verts.device

    edges, faces_to_edges = get_edges(verts, faces)

    e0, e1 = edges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L


def laplacian_smoothing(verts, faces, method="uniform", return_sum=True):
    weights = 1.0 / verts.shape[0]

    with torch.no_grad():
        if method == "uniform":
            L = compute_laplacian(verts, faces)
        else:
            L, inv_areas = laplacian_cot(verts, faces)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas

    if method == "uniform":
        loss = L.mm(verts)
    elif method == "cot":
        loss = L.mm(verts) * norm_w - verts
    else:
        loss = (L.mm(verts) - L_sum * verts) * norm_w

    loss = loss.norm(dim=1)

    loss = loss * weights
    if return_sum:
        return loss.sum()
    else:
        return loss