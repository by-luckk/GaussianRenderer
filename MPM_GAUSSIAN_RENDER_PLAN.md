# MPM Gaussian 渲染实施计划

## 目标

基于现有 [tests/mpm_fall_flower.py](/home/boyuan/projects/GaussianRenderer/tests/mpm_fall_flower.py) 的 MPM 方法，使用 [src/gaussian_renderer](/home/boyuan/projects/GaussianRenderer/src/gaussian_renderer) 中现有渲染能力，对 [tests/assets/flower1.ply](/home/boyuan/projects/GaussianRenderer/tests/assets/flower1.ply) 做逐帧 Gaussian 渲染。

目标链路：

1. 从 `flower1.ply` 读取 Gaussian 资产
2. 从 Gaussian 点中采样出用于 MPM 的粒子初始状态
3. 运行 Genesis MPM 仿真
4. 将 MPM 粒子运动转换为 Gaussian 的逐帧位移
5. 用 `gaussian_renderer` 渲染每一帧，输出图像序列和 GIF

约束：

- `gaussian_renderer` 只做最小改动
- 仿真侧和 Gaussian 变形桥接逻辑集中写在 `src/genesis/gaussian/`
- `tests/` 中只保留测试入口和结果导出逻辑
- 可调参数放在 `tests/config/`
- 实现遵守 [AGENTS.md](/home/boyuan/projects/GaussianRenderer/AGENTS.md)：最小、清晰、高性能

## 现状判断

### 现有 MPM 链路

[tests/mpm_fall_flower.py](/home/boyuan/projects/GaussianRenderer/tests/mpm_fall_flower.py) 已经具备：

- 从 `.ply` 读取 Gaussian 点位置和 opacity
- 对点云做 `voxel_downsample_indices(...)`
- 用 `prepare_flower_particles(...)` 生成 MPM 粒子初始位置
- 构建 Genesis MPM scene
- 导出粒子轨迹
- 用 matplotlib 把粒子轨迹渲染成点云 GIF

### 现有 Gaussian 渲染链路

[src/gaussian_renderer/core/gs_renderer.py](/home/boyuan/projects/GaussianRenderer/src/gaussian_renderer/core/gs_renderer.py) 已经具备：

- 加载 `.ply` 为 `GaussianData`
- 管理多个 Gaussian 模型
- 用 `render_batch(...)` 做批量渲染
- 用 `update_gaussian_properties(...)` 做刚体位姿更新

### 当前缺口

当前缺的是“非刚体逐帧更新 Gaussian”的最小接口：

- 不能直接把一帧新的 `xyz` 写给 renderer
- 不能在不走刚体 body pose 的情况下，按粒子/点变形驱动 Gaussian
- MPM 侧的 Gaussian 采样、绑定、驱动逻辑还散落在测试文件里，无法复用

## 设计原则

### 原则 1：先做 `xyz` 驱动

第一版只更新 Gaussian 的 `xyz`。

不做：

- 每帧 `rot` 更新
- 每帧 `scale` 更新
- SH 旋转

理由：

- 这是最小可用链路
- 对 `gaussian_renderer` 改动最小
- 先验证 MPM 粒子到 Gaussian 形变映射是否稳定

### 原则 2：MPM 粒子是驱动代理，不是渲染对象

MPM 只负责输出形变场。

渲染仍然使用完整的 Gaussian 资产：

- 渲染点数量不等于 MPM 粒子数
- MPM 粒子数由仿真预算控制
- Gaussian 数量保持来自原始 `.ply`

### 原则 3：绑定只做一次

初始化阶段预计算：

- 每个 Gaussian 绑定到哪个 MPM 粒子
- 或绑定到若干个近邻粒子的加权关系

每帧只做：

- 从当前粒子位置查表
- 计算新的 Gaussian `xyz`

避免每帧做全量最近邻搜索。

## 推荐目录

### 新增目录

```text
src/genesis/gaussian/
├── __init__.py
├── config.py
├── gaussian_asset.py
├── particle_sampling.py
├── binding.py
├── deformation.py
├── mpm_scene.py
└── renderer_bridge.py
```

### 各文件职责

- `config.py`
  - 定义 MPM 和 Gaussian 渲染相关 dataclass
- `gaussian_asset.py`
  - 读取 `.ply`
  - 规范化 Gaussian 数据
  - 提供 rest pose 数据
- `particle_sampling.py`
  - 放 `voxel_downsample_indices(...)`
  - 放 `prepare_flower_particles(...)`
  - 从测试文件里抽出来做复用
- `binding.py`
  - 建立 `Gaussian -> MPM 粒子` 的静态绑定
  - 第一版先做最近邻或 KNN 加权
- `deformation.py`
  - 用当前粒子位置更新 Gaussian `xyz`
- `mpm_scene.py`
  - 封装 Genesis MPM scene 构建、step、状态读取
- `renderer_bridge.py`
  - 把每帧 Gaussian `xyz` 写回 `GSRenderer`
  - 统一暴露仿真 + 渲染入口

## 对 gaussian_renderer 的最小改动

### 必要改动

只做一个最小接口增强。

建议改动：

1. 在 [src/gaussian_renderer/core/gs_renderer.py](/home/boyuan/projects/GaussianRenderer/src/gaussian_renderer/core/gs_renderer.py) 增加一个直接更新 Gaussian 属性的方法

推荐接口：

```python
def apply_gaussian_xyz(self, xyz: Tensor | np.ndarray) -> None:
    ...
```

或：

```python
def apply_gaussian_deformation(
    self,
    xyz: Tensor | np.ndarray,
    rot: Tensor | np.ndarray | None = None,
    scale: Tensor | np.ndarray | None = None,
) -> None:
    ...
```

建议选第二种，但第一版只传 `xyz`。

2. 保持现有 `update_gaussian_properties(...)` 不变

理由：

- MuJoCo / Motrix 的刚体路径不能被破坏
- Genesis 柔性体和原有刚体更新应并存

### 不建议做的事

- 不改 `batch_rasterization.py` 的渲染逻辑
- 不改 `.ply` IO
- 不把 MPM 仿真逻辑塞进 `gaussian_renderer`
- 不为单次测试设计统一后端抽象

## 核心数据流

### 初始化

1. 从 `flower1.ply` 读取完整 `GaussianData`
2. 用 `particle_sampling.py` 从 Gaussian 点中生成 MPM 粒子
3. 建立 MPM scene
4. 建立 `Gaussian -> 粒子` 绑定
5. 初始化 renderer，并保存 Gaussian rest `xyz`

### 每帧

1. `scene.step()`
2. 读取当前粒子位置
3. 根据绑定关系得到当前帧 Gaussian `xyz`
4. 调用 `GSRenderer.apply_gaussian_deformation(...)`
5. 调用 `render_batch(...)`
6. 保存 RGB / depth / GIF 帧

## 绑定策略

### 第一版：KNN 粒子加权

建议每个 Gaussian 绑定到 `K=4` 或 `K=8` 个最近粒子。

初始化阶段保存：

- `particle_indices`: `(N_gaussian, K)`
- `weights`: `(N_gaussian, K)`
- `rest_offset`: `(N_gaussian, 3)`

每帧更新：

```text
gaussian_xyz_t = weighted_sum(particle_pos_t[particle_indices]) + rest_offset
```

优点：

- 比最近单粒子更平滑
- 实现简单
- 不依赖 mesh 拓扑
- 很适合当前 MPM 粒子驱动链路

### 第二版再考虑

后续如果第一版效果不够，可以再引入：

- 局部刚性拟合
- APIC/MLS 风格局部仿射场
- 用邻域变形估计 `rot` / `scale`

第一版不做这些。

## 配置设计

### 配置位置

放在：

```text
tests/config/
```

建议新增：

```text
tests/config/
└── mpm_gaussian_flower.py
```

### 配置内容

建议把以下参数放入 config：

- 输入资产路径
- MPM 最大粒子数
- opacity 过滤分位数
- 目标物体尺度
- 掉落高度 / clearance
- 平面高度
- 重力
- MPM 材料参数
- 绑定近邻数 `K`
- 渲染图像尺寸
- 相机位姿
- 输出目录
- GIF fps
- 是否渲染 depth

推荐结构：

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FlowerMPMGaussianConfig:
    input_ply: Path
    max_particles: int
    target_extent: float
    opacity_quantile: float
    drop_height: float | None
    drop_clearance: float | None
    plane_height: float
    gravity_z: float
    youngs_modulus: float
    poisson_ratio: float
    density: float
    bind_k: int
    steps: int
    width: int
    height: int
    fovy_deg: float
    camera_pos: tuple[float, float, float]
    camera_xmat: tuple[float, ...]
    output_dir: Path
    gif_fps: int
```

## 测试与脚本规划

### 测试入口

建议新增一个新的测试脚本：

[tests/mpm_gaussian_flower_render.py](/home/boyuan/projects/GaussianRenderer/tests/mpm_gaussian_flower_render.py)

职责：

- 读取 config
- 初始化 MPM + Gaussian bridge
- 运行仿真
- 导出渲染结果

### 输出内容

建议输出：

- `rgb_XXXX.png`
- `depth_XXXX.npy` 或 `depth_XXXX.png`
- `gaussian_xyz.npy`
- `particle_pos.npy`
- `render.gif`

位置统一放到：

```text
tests/results/flower1_mpm_gaussian/
```

### 旧测试脚本的调整

[tests/mpm_fall_flower.py](/home/boyuan/projects/GaussianRenderer/tests/mpm_fall_flower.py)

建议改成：

- 只保留 MPM 粒子仿真与粒子可视化测试
- 删除已经迁移到 `src/genesis/gaussian/particle_sampling.py` 的复用函数
- 通过 import 复用新模块

这样可以避免测试脚本继续堆业务逻辑。

## 分阶段实施

### Phase 1：抽公共模块

目标：把测试里和 Gaussian 相关的可复用逻辑移出。

任务：

1. 新建 `src/genesis/gaussian/`
2. 新建 `config.py`
3. 新建 `particle_sampling.py`
4. 把 `voxel_downsample_indices(...)` 和 `prepare_flower_particles(...)` 从测试脚本迁移出去
5. 让 `tests/mpm_fall_flower.py` 改为复用新模块

完成标准：

- MPM 原测试仍能跑
- 采样逻辑不再重复定义在测试里

### Phase 2：补最小 renderer 变形入口

目标：让 renderer 支持非刚体逐帧 `xyz` 更新。

任务：

1. 在 `GaussianData` / `GSRenderer` 路径中增加最小更新接口
2. 不改现有刚体 API
3. 用一个最小单元测试验证新接口会真正改动渲染输入

完成标准：

- 不依赖 body pose 也能更新 Gaussian 位置
- 原 MuJoCo / Motrix 接口不受影响

### Phase 3：建立 MPM-Gaussian 绑定

目标：让粒子能驱动完整 Gaussian 资产。

任务：

1. 新建 `gaussian_asset.py`
2. 新建 `binding.py`
3. 新建 `deformation.py`
4. 实现：
   - 读取 `.ply`
   - 建立 `Gaussian -> KNN 粒子` 绑定
   - 根据粒子位置计算逐帧 Gaussian `xyz`

完成标准：

- 给定粒子轨迹，能稳定输出每帧 Gaussian `xyz`
- 输出 shape 与原始 `.ply` 高斯数一致

### Phase 4：封装 scene + renderer bridge

目标：把测试入口压缩成少量流程代码。

任务：

1. 新建 `mpm_scene.py`
2. 新建 `renderer_bridge.py`
3. 统一封装：
   - scene 初始化
   - scene.step()
   - 读取粒子状态
   - 更新 Gaussian
   - 渲染一帧

完成标准：

- 主测试脚本只负责配置、循环和导出
- MPM / Gaussian / renderer 三层边界清晰

### Phase 5：写最终测试脚本和配置

目标：跑通 flower1 的 MPM Gaussian 渲染。

任务：

1. 新建 `tests/config/mpm_gaussian_flower.py`
2. 新建 `tests/mpm_gaussian_flower_render.py`
3. 输出帧图和 GIF
4. 如有必要，补一个单独的“从帧图合成 GIF”的脚本

完成标准：

- 单命令可跑通 `flower1.ply -> MPM -> Gaussian render`
- 输出结果可复查

## 风险与对应策略

### 风险 1：MPM 粒子数太少，Gaussian 跟随不稳定

处理：

- 配置里暴露 `max_particles`
- 第一版用 KNN 平滑降低噪声

### 风险 2：Gaussian 数量远大于 MPM 粒子数，局部塌缩明显

处理：

- 第一版接受一定视觉误差
- 后续再考虑局部仿射拟合

### 风险 3：逐帧 CPU-Numpy 映射太慢

处理：

- 第一版先跑通
- 绑定索引和权重尽量缓存成 torch tensor
- 每帧更新尽量在 GPU 上完成

### 风险 4：renderer 接口改动影响旧调用

处理：

- 只新增接口，不修改旧接口语义
- 保持调用路径向后兼容

## 本次实施的完成定义

当以下结果全部满足时，本次任务算完成：

1. `src/genesis/gaussian/` 下形成可复用的 MPM-Gaussian 桥接模块
2. `tests/mpm_fall_flower.py` 中公共采样逻辑已迁移到模块内
3. `gaussian_renderer` 只做了最小必要改动，支持逐帧 Gaussian `xyz` 更新
4. `tests/config/` 下有可复用 config
5. `tests/` 下有可运行的 `flower1.ply` MPM Gaussian 渲染测试
6. 能输出逐帧渲染结果和 GIF

## 建议实现顺序

建议严格按下面顺序做：

1. 先抽 `particle_sampling.py`
2. 再加 `GSRenderer` 的最小更新接口
3. 再做 `binding.py` 和 `deformation.py`
4. 再封装 `mpm_scene.py` 和 `renderer_bridge.py`
5. 最后写测试入口和结果导出

这样做的好处是每一步都可单独验证，不会一次性把仿真、变形、渲染混在一起。
