# SpatioCube

面向空间转录组（ST）多切片对齐与三维空间聚类的 Python 工具。

> 这份 README 也作为“交接文档”：下一次优化时，新的 agent 只要先读这里，就能快速理解项目目标、数据约定、核心 API、关键参数与常见坑。

## Demo 目前实现了什么（你当前能用到的能力）

- **切片拆分**：按 `adata.obs['sampleid']` 拆分多切片（不依赖 index/字符串排序推断顺序）。
- **切片顺序推断**：基于表达 embedding 的跨切片距离（可选 OT 距离）+ 全局最短路径，推断一个“整体流畅”的线性顺序。
- **2D 刚性对齐**：对每张切片做 **旋转 + 平移**（刚性变换），把各自坐标系叠到同一个 XY 参考系中（相邻片对齐并累积）。
- **构建 3D 坐标**：将对齐后的 XY 与每片的 z 合并，写入 `adata.obsm['spatial_3d']`。
- **3D 聚类**：在 3D 邻接图上做 Leiden（或可选对比学习 embedding 后聚类），写入 `adata.obs['SpatioCube_cluster']`。
- **3D 可视化**：Plotly / PyVista 进行叠片式 3D 展示与检查。

## 关键数据约定（非常重要）

- **切片标识**：默认使用 `adata.obs['sampleid']` 作为切片键（可通过 `slice_key=` 修改）。
- **2D 坐标来源**：
  - 若 `adata.obsm['spatial']` 已存在，直接使用
  - 否则自动从 `adata.obs['coor_x_ad2']`、`adata.obs['coor_y_ad2']` 写入 `obsm['spatial']`
- **对齐前坐标快照**：`SpatioCube.from_merged_h5ad(..., spatial_raw_key='spatial_xy_raw')` 会把对齐前的 XY 备份到 `obsm['spatial_xy_raw']`（便于对比对齐前后）。

## 安装（开发模式）

```bash
cd /cluster2/huanglab/jiamao/Project/SpatioCube
pip install -e ".[dev,leiden,align_ot,viz]"
```

如果你只想用核心 IO / 容器类：

```bash
pip install -e .
```

## 最小使用流程（端到端示例）

以 notebook `notebooks/01_align_cluster_viz.ipynb` 为准，核心调用链大致如下（伪代码/示意）：

```python
import spatiocube as scb

adata = scb.read_merged_h5ad()  # 或传入路径

cube = scb.SpatioCube.from_merged_h5ad(
    adata,
    slice_key="sampleid",
    order_mode="infer",  # 不信任 sampleid 字符串顺序：推断切片前后关系
    order_config=scb.OrderConfig(
        subsample_n=2000,
        svd_dim=50,
        knn=30,
        use_ot=True,   # 用 OT 距离评估切片间相似度（更鲁棒）
        ot_reg=0.05,
    ),
    # 3D 视觉层间距（用于看起来像“实体器官”叠片）
    z_spacing=50.0,
)

# 关键：把每片坐标系旋转/平移到一起（写回 cube.adatas[i].obsm['spatial']）
scb.align_adjacent_slices_ot(
    cube,
    transport="emd",   # 小 demo 推荐（子采样不大时更稳定/可解释）；过大时自动回退 sinkhorn
    subsample_n=2000,
    svd_dim=50,
    expr_knn=None,     # None 表示 dense cost（子采样规模下更稳）
    n_iter=3,
)

# 写入 3D 坐标（xy 来自对齐后的 obsm['spatial']，z 来自 cube.z_positions）
cube.write_back()

# 构图 + Leiden 聚类（写入 obs['SpatioCube_cluster']）
A = scb.build_3d_adjacency(cube, k_intra=15, k_inter=5, prefer_mapping=True)
labels = scb.leiden_cluster(A, resolution=1.0)
cube.set_clusters(labels)

# 3D 可视化（Plotly/PyVista）
fig = scb.plot_3d_plotly(cube, color_key="SpatioCube_cluster")
```

## MouseBrain 数据（集群路径）

- **数据根目录**：`/cluster3/labData/jiamao/MouseBrain/`
- **合并 h5ad**：你已从 `.rds` 转换为 `.h5ad`（参考 `data/format.ipynb`）
- **切片键**：默认使用 `adata.obs['sampleid']` 拆片（可通过 `slice_key=` 修改）
- **空间坐标**：
  - 若 `adata.obsm['spatial']` 已存在，直接使用
  - 否则自动从 `adata.obs['coor_x_ad2']`、`adata.obs['coor_y_ad2']` 写入 `obsm['spatial']`
- **环境变量**：可设置 `SPATIOCUBE_MOUSEBRAIN_H5AD=/path/to/file.h5ad`，让 `read_merged_h5ad()` 直接读取

## z 轴：`z_spacing`（可视化层间距） vs `lambda_z`（建图/距离权重）

这两个参数经常被混淆：

- **`z_spacing`（推荐你关注）**：控制 `cube.z_positions` 的实际数值间距，直接影响 `obsm['spatial_3d']` 的第三维，从而影响 3D 叠片的“厚度/实体感”。  
  - XY 范围很大时（你给的 bbox 就是这种情况），如果 z 仍是 0..9，视觉上会非常扁；应把 `z_spacing` 调到与 XY 同量级（例如几十到几百）。
- **`lambda_z`（用于距离度量/聚类）**：控制 3D 距离里 z 方向的权重（影响邻接图/聚类），不等同于可视化层间距。

## `lambda_z`（切片间距权重，用于距离/建图）

SpatioCube 使用度量：

\[
d^2 = \|x_i-x_j\|^2 + \lambda_z (z_i-z_j)^2
\]

切片间距通常远大于 spot 间距，推荐把 `lambda_z` 作为需要调参的关键超参（例如 0.001–0.1 的量级起步）。

## 对齐函数要点：`align_adjacent_slices_ot`（旋转/平移叠片）

对齐的目标是把每片从“各自坐标系”变成“共享坐标系”，最终反映在：

- `cube.adatas[i].obsm['spatial']`：对齐后的 2D 坐标
- `cube.adatas[i].obsm['spatial_3d']`：对齐后的 3D 坐标（`cube.write_back()` 写入）

你最常需要调的参数：

- **`transport`**：
  - `transport="emd"`：对子采样点做一对一线性分配（更稳定、可解释），适合小 demo 或 subsample 不大时使用
  - `transport="sinkhorn"`：熵正则 OT（需要 POT），适合更大规模或不等长情况（但更需要调 `ot_reg`）
  - 注：当 `emd` 不适用（子采样过大/不等长）会自动回退到 `sinkhorn` 并给出 warning
- **`subsample_n`**：对齐子采样规模。越大越准但越慢。
- **`expr_knn`**：
  - `None`：对子采样点用 dense expression cost（更稳，但 \(O(n^2)\)）
  - 整数：用表达 KNN 限制候选（更快，但过强裁剪可能错过真匹配）
- **`paired_subsample=True`**：尽量让源/目标子采样“成对”，避免 OT 代价矩阵退化。

## 输出字段（写回到 AnnData 的结果）

- **`adata.obsm['spatial']`**：每张切片的 2D 坐标（对齐后会被更新）
- **`adata.obsm['spatial_xy_raw']`**：对齐前的 2D 坐标快照（如果启用 `spatial_raw_key`）
- **`adata.obsm['spatial_3d']`**：3D 坐标（对齐后的 XY + 切片 z）
- **`adata.obs['SpatioCube_cluster']`**：3D 聚类标签
- **`adata.uns['SpatioCube']`**：元信息（slice_key、slice_order、z_positions、对齐参数等）

## 常见问题 / 排错清单

- **“看起来还是没叠在一起（像原始分片）”**：
  - 检查是否真的运行了对齐，并且对齐后 `obsm['spatial']` 发生变化（对比 `obsm['spatial_xy_raw']`）。
  - 若 3D 里 z 太小，视觉上会像“薄饼”：增大 `z_spacing`。
- **Plotly `fig.show()` 报 `nbformat`**：
  - 安装 `nbformat`/`ipython` 或改用 `fig.write_html(...)`。
- **Notebook import `spatiocube` 失败**：
  - 推荐 `pip install -e .`（或在 notebook 里把 `src/` 加到 `sys.path`）。

## 目录结构（基础骨架）

- `src/spatiocube/`: 主包代码
- `data/`: 你的 demo 数据（建议放大文件时自行忽略/不提交）
- `notebooks/`: 交互式分析与流程演示
- `reference/`: 参考资料（你提供的 articles/notes）
- `tests/`: 单元测试
- `utils/`: 一次性脚本/杂项工具（可选）

