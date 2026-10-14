import torch
from torch_geometric.transforms import BaseTransform


class AddVirtualNode(BaseTransform):
    def __init__(self, vnode_node_type=28, vnode_edge_type=4):
        self.vnode_node_type = vnode_node_type
        self.vnode_edge_type = vnode_edge_type

    def forward(self, data):
        num_nodes = data.num_nodes
        vnode_idx = num_nodes
        device = data.x.device if data.x is not None else 'cpu'

        for pe_attr in ['random_walk_pe', 'laplacian_pe']:
            if hasattr(data, pe_attr) and data[pe_attr] is not None:
                pe = data[pe_attr]
                if pe.dim() == 2 and pe.size(0) == num_nodes:
                    pad = torch.zeros(1, pe.size(1), device=pe.device, dtype=pe.dtype)
                    data[pe_attr] = torch.cat([pe, pad], dim=0)

        if data.x is not None:
            if data.x.dtype in (torch.long, torch.int, torch.int32):
                if data.x.dim() == 1:
                    vnode_x = torch.tensor([self.vnode_node_type], device=device)
                else:
                    vnode_x = torch.full((1, data.x.size(1)), self.vnode_node_type,
                                         dtype=data.x.dtype, device=device)
            else:
                if data.x.dim() == 1:
                    vnode_x = torch.zeros(1, device=device, dtype=data.x.dtype)
                else:
                    vnode_x = torch.zeros(1, data.x.size(1), device=device, dtype=data.x.dtype)
            data.x = torch.cat([data.x, vnode_x], dim=0)

        real_nodes = torch.arange(num_nodes, device=device)
        vnode_repeated = torch.full((num_nodes,), vnode_idx, dtype=torch.long, device=device)
        new_src = torch.cat([real_nodes, vnode_repeated])
        new_dst = torch.cat([vnode_repeated, real_nodes])
        new_edges = torch.stack([new_src, new_dst], dim=0)
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

        if data.edge_attr is not None:
            num_new_edges = 2 * num_nodes
            if data.edge_attr.dim() == 1:
                new_edge_attr = torch.full((num_new_edges,), self.vnode_edge_type,
                                           dtype=data.edge_attr.dtype, device=device)
            else:
                new_edge_attr = torch.full((num_new_edges, data.edge_attr.size(1)),
                                           self.vnode_edge_type,
                                           dtype=data.edge_attr.dtype, device=device)
            data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

        data.vnode_idx = vnode_idx

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'vnode_node_type={self.vnode_node_type}, '
                f'vnode_edge_type={self.vnode_edge_type})')


class SafeLaplacianPE(BaseTransform):
    def __init__(self, k=16, attr_name='laplacian_pe', is_undirected=True):
        from torch_geometric.transforms import AddLaplacianEigenvectorPE
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self._base_class = AddLaplacianEigenvectorPE

    def forward(self, data):
        n = data.num_nodes
        actual_k = min(self.k, n - 1)

        if actual_k <= 0:
            data[self.attr_name] = torch.zeros(n, self.k)
            return data

        transform = self._base_class(
            k=actual_k,
            attr_name=self.attr_name,
            is_undirected=self.is_undirected,
        )
        data = transform(data)

        if actual_k < self.k:
            pe = data[self.attr_name]
            padding = torch.zeros(n, self.k - actual_k, device=pe.device, dtype=pe.dtype)
            data[self.attr_name] = torch.cat([pe, padding], dim=-1)

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(k={self.k})'
