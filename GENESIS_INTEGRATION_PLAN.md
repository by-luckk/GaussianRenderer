# Genesis 柔性体集成计划

## 目标

在 `GaussainRenderer` 中接入 Genesis 柔性体仿真。

核心链路：

1. Genesis 输出物体 Mesh 的逐帧形变
2. 将 Mesh 形变转换为 Gaussian kernel 的逐帧形变
3. 将更新后的 Gaussian 数据交给 `gaussian_renderer` 渲染

约束：

- Genesis 相关代码集中放在 `src/genesis/`
- `gaussian_renderer` 继续专注渲染，不直接承载仿真逻辑
- 实现保持最小、清晰、高性能，符合 `README.md`、`CONTRIBUTING.md`、`AGENTS.md`

## 现状判断

当前渲染主链路在 `src/gaussian_renderer/core/`：

- `gs_renderer.py` 负责组织 Gaussian 数据与批量渲染
- `batch_rasterization.py` 负责实际 rasterization
- `gaussiandata.py` 定义 `GaussianData`

当前动态更新能力只覆盖刚体：

- `GSRenderer.update_gaussian_properties()` 仅支持按物体位姿批量更新 `xyz` 和 `rot`
- `gs_renderer_mujoco.py` 的接入方式也是“刚体 body pose -> Gaussian pose”

因此 Genesis 柔性体接入的关键不是新增一个 renderer，而是补一条新的数据路径：

`deformed mesh -> gaussian deformation field -> GaussianData -> GSRenderer`

## 推荐架构

建议把功能拆成四层。

### 1. 仿真层

位置：`src/genesis/scene.py`

职责：

- 初始化 Genesis scene
- 加载柔性体对象
- 执行 step
- 读出当前帧 mesh 顶点位置

输出：

- `vertices_rest`
- `vertices_deformed`
- 可选的 `faces`

这一层只管 Genesis，不接触 renderer。

### 2. 绑定层

位置：`src/genesis/gaussian_binding.py`

职责：

- 在初始化阶段建立 `mesh vertex / triangle -> gaussian` 的静态绑定
- 保存每个 Gaussian 的局部坐标、参考三角形或参考点
- 保存 rest pose 下的参考旋转/尺度信息

推荐先做最小版本：

- 每个 Gaussian 绑定到最近三角形，或最近顶点加局部偏移
- 先保证 `xyz` 正确更新
- `rot` 和 `scale` 先做最小可用近似，不一开始追求复杂连续形变

建议绑定结果缓存成独立数据结构，避免每帧重复搜索最近点。

### 3. 形变转换层

位置：`src/genesis/deformation.py`

职责：

- 输入 rest mesh 和当前帧 deformed mesh
- 根据绑定关系计算每个 Gaussian 的：
  - 新位置 `xyz`
  - 新旋转 `rot`
  - 可选新尺度 `scale`

建议分两阶段实现：

第一阶段：

- 只更新 `xyz`
- `rot` 保持 rest pose 或由局部刚性拟合给出
- `scale` 保持不变

第二阶段：

- 加入局部旋转估计
- 视效果决定是否加入各向异性尺度更新

这样能先打通链路，再逐步提高视觉质量。

### 4. 渲染桥接层

位置：`src/genesis/renderer_bridge.py`

职责：

- 接收 `GaussianData` 与每帧 deformation 输出
- 将变化写回 `GSRenderer.gaussians`
- 提供面向外部的一致接口，例如：
  - `initialize_binding(...)`
  - `step_simulation(...)`
  - `update_gaussians(...)`
  - `render(...)`

这一层负责衔接 `src/genesis/` 与 `src/gaussian_renderer/`，避免两边互相侵入。

## 推荐目录

建议新增以下结构：

```text
src/genesis/
├── __init__.py
├── scene.py
├── mesh_state.py
├── gaussian_binding.py
├── deformation.py
├── renderer_bridge.py
└── types.py
```

建议职责如下：

- `scene.py`: Genesis scene 生命周期与 mesh 导出
- `mesh_state.py`: rest/deformed mesh 数据容器
- `gaussian_binding.py`: Gaussian 与 mesh 的静态绑定
- `deformation.py`: 每帧从 mesh 形变推导 Gaussian 形变
- `renderer_bridge.py`: 调用 `GSRenderer` 完成更新和渲染
- `types.py`: 小型 dataclass，避免跨文件传裸 dict

## 对现有代码的最小改动

为了保持边界清晰，建议只对 `gaussian_renderer` 做最小增强。

### 必要改动

1. 在 `GaussianData` 更新路径里支持直接写入新的 `xyz`、`rot`、`scale`
2. 在 `GSRenderer` 中补一个更通用的接口，例如：
   - `update_gaussians_xyz(...)`
   - `update_gaussians_xyz_rot(...)`
   - 或单个 `apply_deformation(...)`
3. 保持现有刚体接口不删，避免破坏 MuJoCo 用法

### 不建议现在做的事

- 不要一开始把 Genesis 逻辑混进 `gaussian_renderer/core/`
- 不要先做过度抽象的“统一仿真后端框架”
- 不要一开始就引入复杂的 per-Gaussian SH 旋转

## 实施阶段

### Phase 1: 打通最小链路

目标：看到 Genesis 柔性体驱动下的 Gaussian 渲染结果。

任务：

1. 创建 `src/genesis/`
2. 建立 Genesis scene，并能导出 rest mesh 与当前帧 mesh
3. 为每个 Gaussian 预计算 mesh 绑定关系
4. 每帧只更新 Gaussian 的 `xyz`
5. 复用现有 `GSRenderer.render_batch()` 完成渲染

交付标准：

- 单个柔性体对象能随仿真产生形变
- Gaussian 点云随 mesh 一起运动
- 渲染结果稳定，无明显索引错乱

### Phase 2: 补足姿态与局部形变

目标：让 Gaussian 不只是“跟点走”，还具备更合理的局部方向。

任务：

1. 基于三角形局部坐标或邻域拟合估计 `rot`
2. 评估是否需要更新 `scale`
3. 为局部旋转加入批量化 torch 实现

交付标准：

- 柔性弯折时，Gaussian 方向与表面局部朝向基本一致
- 相比只更新 `xyz`，视觉伪影明显减少

### Phase 3: 工程化

目标：让功能可维护、可复用。

任务：

1. 增加 Genesis optional dependency
2. 增加最小示例脚本
3. 增加单元测试与集成测试
4. 评估是否需要缓存、CUDA 化或 torch compile

交付标准：

- 新用户能按 README 跑通示例
- 核心绑定和形变计算有测试覆盖

## 数据结构建议

建议在 `src/genesis/types.py` 中定义少量 dataclass。

```python
@dataclass
class MeshState:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class GaussianBinding:
    gaussian_indices: np.ndarray
    face_indices: np.ndarray
    barycentric: np.ndarray
    local_offset: np.ndarray
```

如果第一版不做三角形重心绑定，也可以先简化为：

```python
@dataclass
class GaussianVertexBinding:
    vertex_indices: np.ndarray
    local_offset: np.ndarray
```

建议先从最简单且稳定的版本开始，不要提前设计过多层次。

## Mesh 到 Gaussian 的映射策略

建议按下面顺序推进。

### 方案 A: 最近顶点绑定

优点：

- 实现最快
- 容易验证

缺点：

- 形变不够平滑
- 局部旋转估计较弱

适合作为第一版。

### 方案 B: 最近三角形 + 重心坐标绑定

优点：

- 位置更新更平滑
- 更容易估计局部旋转

缺点：

- 初始化绑定更复杂

适合作为正式版本。

### 方案 C: 邻域加权蒙皮

优点：

- 最灵活

缺点：

- 计算和调参成本更高

不建议在第一轮实现。

## 性能建议

这个仓库是 GPU 密集型代码，柔性体更新部分要尽量避免 CPU 瓶颈。

建议：

- 绑定只做一次，结果缓存
- 每帧避免 Python for-loop 遍历 Gaussian
- 每帧形变计算尽量用 `numpy` 向量化或 `torch` 批量化
- 如果 Genesis 顶点输出在 CPU，先做清晰版本，再考虑减少 CPU-GPU copy
- 先测单物体，再扩展到多物体

## 依赖与打包改动

当前 `pyproject.toml` 只打包 `gaussian_renderer*`，如果 `src/genesis/` 要作为仓库内正式模块使用，需要补两处：

1. optional dependency，例如：
   - `genesis = ["genesis-world"]`
2. setuptools package include，例如：
   - `include = ["gaussian_renderer*", "genesis*"]`

如果暂时只是仓库内自用脚本，也可以先不发布到包管理流程，但最好尽早统一。

## 测试计划

建议至少覆盖三类测试。

### 1. 绑定测试

- 给定固定 rest mesh 和 Gaussian
- 验证绑定索引、重心坐标、局部偏移是否正确

### 2. 形变测试

- 构造一个可解析的小形变
- 验证 Gaussian 更新后的 `xyz` 与预期一致
- 如果实现了 `rot`，验证四元数方向基本正确

### 3. 渲染集成测试

- mock 一个最小 mesh deformation
- 确认 `GSRenderer` 能消费更新后的 Gaussian 数据
- 不要求真实 Genesis 环境也能跑测试

## 建议的首个里程碑

第一周目标建议非常克制：

1. 建 `src/genesis/`
2. 跑通一个单柔性体示例
3. 实现“最近顶点绑定”
4. 每帧只更新 Gaussian `xyz`
5. 能渲染出连续动画

只要这一步通了，后面再补 `rot`、`scale`、更平滑的 triangle binding，风险会低很多。

## 需要你后续确认的点

下面这些点会直接影响实现细节，建议在正式开工前定下来。

1. Genesis 中柔性体输出的是顶点级 mesh，还是已经有更高层的局部 frame 信息
2. 当前一个柔性体对象对应一个 Gaussian 模型，还是一个 mesh 需要驱动多个 Gaussian 子集
3. 第一版是否接受只更新 `xyz`
4. 是否需要把 Genesis 集成做成和 MuJoCo 类似的独立入口，例如 `gs_renderer_genesis.py`

## 推荐落地顺序

1. 先在 `src/genesis/` 做最小闭环，不动核心 rasterization
2. 只给 `GSRenderer` 增加一个通用 deformation 更新入口
3. 先做最近顶点绑定，验证视觉闭环
4. 再升级到三角形重心绑定
5. 最后再考虑旋转、尺度、SH 特征旋转

这个顺序最符合仓库当前风格：少改核心、先通链路、再做性能和精度优化。
