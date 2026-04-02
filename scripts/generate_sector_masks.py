import pandas as pd
import torch
import os

def format_stkcd(code):
    return str(code).split('.')[0].zfill(6)

# 路径设置
processed_dir = "../data/processed/"
industry_file = "../data/raw_data/STK_INDUSTRYCLASS.csv"

# 1. 加载张量对应的 Stkcd 顺序
if not os.path.exists(os.path.join(processed_dir, "stkcd_order.pt")):
    raise FileNotFoundError("请先在 Notebook 中运行保存 stkcd_order.pt 的代码")
stkcd_order = torch.load(os.path.join(processed_dir, "stkcd_order.pt"))

# 2. 加载行业原始数据
df_ind = pd.read_csv(industry_file, encoding='utf-8')
df_ind['Symbol'] = df_ind['Symbol'].apply(format_stkcd)

# 3. 筛选证监会2012标准(P0207)并获取每个公司最新的行业分类
df_ind = df_ind[df_ind['IndustryClassificationID'] == 'P0207']
df_ind = df_ind.sort_values('ImplementDate').drop_duplicates('Symbol', keep='last')
industry_map = dict(zip(df_ind['Symbol'], df_ind['IndustryCode']))

# 4. 构造布尔掩码
c38_mask = torch.tensor([industry_map.get(s) == 'C38' for s in stkcd_order])
c39_mask = torch.tensor([industry_map.get(s) == 'C39' for s in stkcd_order])

# 5. 保存结果
torch.save({'c38': c38_mask, 'c39': c39_mask}, os.path.join(processed_dir, "sector_masks.pt"))
print(f"✅ 行业掩码生成完毕：C38({c38_mask.sum()})，C39({c39_mask.sum()})")