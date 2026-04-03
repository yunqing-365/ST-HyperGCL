import torch
import numpy as np
import os

processed_dir = "../data/processed/"
edge_indices = torch.load(os.path.join(processed_dir, "dynamic_edge_indices.pt"))
masks = torch.load(os.path.join(processed_dir, "sector_masks.pt"))

def get_sector_stats(mask, sector_name):
    sector_node_indices = torch.where(mask)[0]
    all_years_avg_degree = []
    
    for t, edge_index in enumerate(edge_indices):
        # edge_index[0] 是发明人, edge_index[1] 是公司
        # 统计属于该行业的公司的连接边
        is_in_sector = torch.isin(edge_index[1], sector_node_indices)
        sector_edges = edge_index[:, is_in_sector]
        
        if sector_edges.shape[1] > 0:
            # 计算在该时间点，平均每个公司连接了多少个研发人员
            unique_comp, counts = torch.unique(sector_edges[1], return_counts=True)
            all_years_avg_degree.append(counts.float().mean().item())
            
    print(f"📊 {sector_name}: 8年平均研发关联密度 = {np.mean(all_years_avg_degree):.2f}")

get_sector_stats(masks['c38'], "C38 (电气机械)")
get_sector_stats(masks['c39'], "C39 (电子设备)")