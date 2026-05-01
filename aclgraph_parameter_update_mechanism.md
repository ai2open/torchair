# AclGraph 参数更新识别机制详解

## 1. 核心问题

在 reduce-overhead 模式下，aclgraph 使用 NPU Graph 捕获和重放机制来降低开销。但存在一个矛盾：
- **固定 Shape**：为了避免重新捕获，需要固定输入的 shape
- **动态参数**：某些参数（如 actual_seq_lengths）需要在每次推理时动态更新

**解决方案**：识别哪些参数可以动态更新，哪些必须固定在 graph key 中。

---

## 2. 参数分类

### 2.1 两类参数

aclgraph 将所有输入参数分为两类：

| 参数类型 | 说明 | 是否影响 graph key | 是否需要动态更新 |
|---------|------|------------------|----------------|
| **Unupdated Params** | 形状相关参数，影响算子选择和编译 | ✅ 是 | ❌ 否 |
| **Updated Params** | 业务逻辑参数，不影响算子选择 | ❌ 否 | ✅ 是 |

### 2.2 典型示例

以 `npu_fused_infer_attention_score` 为例：

```python
attn_output = torch.ops.npu.npu_fused_infer_attention_score(
    query,                        # shape: [bs, seq_len, hidden_dim] - Unupdated
    key,                          # shape: [bs, kv_seq_len, hidden_dim] - Unupdated
    value,                        # shape: [bs, kv_seq_len, hidden_dim] - Unupdated
    attention_mask,               # shape: [bs, seq_len, kv_seq_len] - Unupdated
    actual_seq_lengths=[5,7,6],   # 实际序列长度列表 - Updated ✅
    actual_seq_lengths_kv=[8,9,7],# KV 实际长度 - Updated ✅
    scale=0.125,
)
```

- **Unupdated**: query/key/value/attention_mask 的 shape 必须固定（通过 padding）
- **Updated**: actual_seq_lengths 可以在每次推理时动态变化

---

## 3. 识别机制的实现

### 3.1 数据结构定义

#### StaticWorkspaceReplaceFunc

定义在 `acl_graph.py:26-42`：

```python
@dataclass
class StaticWorkspaceReplaceFunc:
    """
    定义算子的替换函数和可更新参数
    """
    get_workspace: Callable      # 获取最大 workspace 的函数
    out_operator: Callable        # 替换后的 out 算子
    workspace_keys: List[str]     # workspace 参数名列表
    output_keys: List[str]        # 输出参数名列表
    updated_param_keys: List[str] # 可动态更新的参数名列表 ✅✅✅
```

#### UpdatedNodeInfo

定义在 `acl_graph.py:45-65`：

```python
@dataclass
class UpdatedNodeInfo:
    """
    记录需要动态更新的节点信息
    """
    node_name: str                 # 节点名称
    updated_func: Callable         # 更新函数
    updated_param_name: List[str]  # 需要更新的参数名
    args: Any                      # 函数参数
    kwargs: Any                    # 函数关键字参数
    handle: Any                    # graph_task_group handle
    event: Any                     # ExternalEvent 用于同步
```

### 3.2 注册机制：_REPLACE_FUNC_MAP

定义在 `acl_graph.py:68-84`：

```python
_REPLACE_FUNC_MAP = {
    # Attention 算子
    torch.ops.npu.npu_fused_infer_attention_score.default: StaticWorkspaceReplaceFunc(
        get_workspace=torch.ops.npu._npu_fused_infer_attention_score_get_max_workspace.default,
        out_operator=torch.ops.npu.npu_fused_infer_attention_score.out,
        workspace_keys=["workspace"],
        output_keys=["attention_out", "softmax_lse"],
        updated_param_keys=["actual_seq_lengths", "actual_seq_lengths_kv", "actual_shared_prefix_len"],
        #                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                   这些参数可以动态更新！
    ),

    # 增量 FlashAttention 算子
    torch.ops.npu.npu_incre_flash_attention.default: StaticWorkspaceReplaceFunc(
        get_workspace=torch.ops.npu._npu_incre_flash_attention_get_workspace.default,
        out_operator=torch.ops.npu.npu_incre_flash_attention.out,
        workspace_keys=["workspace"],
        output_keys=["attention_out"],
        updated_param_keys=["actual_seq_lengths", "actual_seq_lengths_kv"],
        #                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                   这些参数可以动态更新！
    ),
}
```

**关键点**：
- 每个算子明确声明了哪些参数可以动态更新（`updated_param_keys`）
- 这是一个**白名单机制**：只有在此列表中的参数才能动态更新

---

### 3.3 识别流程

#### 阶段 1：识别 Unupdated 输入（生成 graph key）

`get_unupdated_sym_input_index()` - `acl_graph.py:167-211`

```python
def get_unupdated_sym_input_index(graph_module: torch.fx.GraphModule):
    """
    识别哪些输入索引不能被动态更新（需要包含在 graph key 中）

    算法逻辑：
    1. 从 _REPLACE_FUNC_MAP 提取所有可更新参数列表
    2. 遍历所有 placeholder 节点（输入参数）
    3. 对于每个包含 SymInt 的输入：
       a. 查找该输入的所有使用节点（users）
       b. 如果任一 user 不在 updated_op_params 中 → Unupdated
       c. 如果所有 users 都在 updated_op_params 中，但该输入不在 updated_param_keys 中 → Unupdated
    4. 返回所有 Unupdated 输入的索引列表
    """
    # 1. 构建 updated_op_params 字典
    updated_op_params = {}
    for func_iter in _REPLACE_FUNC_MAP.values():
        if len(func_iter.workspace_keys) > 0:
            updated_op_params[func_iter.get_workspace] = func_iter.updated_param_keys
        updated_op_params[func_iter.out_operator] = func_iter.updated_param_keys

    unupdated_sym_input_index = []
    data_idx = -1

    # 2. 遍历所有 placeholder 节点
    for node in graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
        data_idx = data_idx + 1

        # 跳过非符号输入
        if not hasattr(node, "meta") or not have_sym_in_meta(node.meta['val']):
            continue

        # 3. 分析该输入的所有使用节点
        have_unupdated_user = False
        for user_node in node.users:
            # 3a. 如果 user 不在 updated_op_params → Unupdated
            if user_node.target not in updated_op_params.keys():
                have_unupdated_user = True
                break

            # 3b. 如果 user 在 updated_op_params，但该输入不在 updated_param_keys → Unupdated
            unupdated_inputs = get_node_all_placeholder_inputs(
                user_node,
                excluded_kwargs=updated_op_params[user_node.target]  # 排除可更新参数
            )
            if node in unupdated_inputs:
                have_unupdated_user = True
                break

        # 4. 记录 Unupdated 输入索引
        if have_unupdated_user:
            unupdated_sym_input_index.append(data_idx)

    return unupdated_sym_input_index
```

**关键逻辑**：
- 如果一个输入被不支持动态更新的算子使用 → Unupdated
- 如果一个输入被支持动态更新的算子使用，但不在其 `updated_param_keys` 中 → Unupdated
- 只有明确在 `updated_param_keys` 中的参数才是 Updated

#### 阶段 2：生成 graph key

`gen_unupdated_input_key()` - `acl_graph.py:147-164`

```python
def gen_unupdated_input_key(*args: Any, **kwargs: Any):
    """
    根据 Unupdated 输入生成 graph key

    只有 Unupdated 参数的值影响 graph key
    """
    input_shape_list = []
    for idx in unupdated_input_index:
        if isinstance(args[idx], torch.Tensor):
            input_shape_list.append(str(list(args[idx].shape)))
        else:
            input_shape_list.append(str(args[idx]))
    return ",".join(input_shape_list)
```

**示例**：

假设模型输入为：
```python
forward(input_ids, attention_mask, position_ids, kv_len, past_key_values, actual_seq_lengths_kv)
```

- `unupdated_input_index = [0, 1, 2, 3, 4]`（前5个参数）
- `actual_seq_lengths_kv` 不在 unupdated_input_index 中

graph key 只依赖前5个参数的 shape：
```python
graph_key = "[1,256],[1,256],[1,256],5,[(1,32,256,128),(1,32,256,128),...]"
```

#### 阶段 3：识别需要更新的节点

`get_updated_ops_rulers_param()` - `acl_graph.py:267-299`

```python
def get_updated_ops_rulers_param(graph_module: torch.fx.GraphModule, meta_inputs: List):
    """
    分析图中哪些节点的哪些参数需要动态更新

    返回：
    - ops_update_rulers: {op_name: {param_name: update_ruler}}
    - need_updated_ops: {op_name: [param_name_list]}
    """
    placeholder_nodes = [node for node in graph_module.graph.nodes if node.op == "placeholder"]

    # 1. 构建 updated_dict: {operator: updated_param_keys}
    updated_dict = {}
    for func_iter in _REPLACE_FUNC_MAP.values():
        updated_dict[func_iter.out_operator] = func_iter.updated_param_keys

    # 2. 遍历所有节点，找到需要更新的节点
    ops_update_rulers = {}
    for node in graph_module.graph.nodes:
        if node.target not in updated_dict.keys():
            continue

        # 3. 为该节点生成 update ruler
        update_rulers = get_update_ruler(node, updated_dict[node.target], placeholder_nodes)
        if len(update_rulers) > 0:
            ops_update_rulers[node.name] = update_rulers

    # 4. 提取需要更新的参数名列表
    need_updated_ops: Dict[str, List] = {}
    for op_name, update_rulers in ops_update_rulers.items():
        updated_params = [param_name for param_name, _ in update_rulers.items()]
        need_updated_ops[op_name] = updated_params

    return ops_update_rulers, need_updated_ops
```

#### 阶段 4：生成 update ruler

`get_update_ruler()` - `acl_graph.py:218-244`

```python
def get_update_ruler(node, updated_param_keys, placeholder_nodes):
    """
    为节点生成参数更新规则

    update_ruler 格式：
    {
        "actual_seq_lengths": [("index", 5), ("fixed", 0)],
        "actual_seq_lengths_kv": [("index", 6)],
    }

    含义：
    - ("index", 5): 从第5个输入参数中获取
    - ("fixed", 0): 使用固定值0
    """
    update_rulers = {}

    # 遍历节点的所有 kwargs
    for kwarg_name, kwarg_value in node.kwargs.items():
        # 只处理在 updated_param_keys 中的参数
        if kwarg_name not in updated_param_keys:
            continue

        if not isinstance(kwarg_value, (list, tuple)):
            raise RuntimeError(f"For updated param type only list is supported")

        update_ruler = []
        have_sym_in_param = False

        # 分析参数值列表
        for kwarg_i in kwarg_value:
            if kwarg_i in placeholder_nodes:
                # 该值来自输入参数
                input_index = placeholder_nodes.index(kwarg_i)
                update_ruler.append(("index", input_index))
                have_sym_in_param = True
            else:
                # 该值是常量
                if not is_constant(kwarg_i):
                    raise RuntimeError(f"For updated param value only sym and constant is supported")
                update_ruler.append(("fixed", kwarg_i))

        # 只有包含符号输入的参数才需要更新
        if have_sym_in_param:
            update_rulers[kwarg_name] = update_ruler

    return update_rulers
```

---

### 3.4 捕获阶段：记录更新信息

#### UpdatedNodeCaptureInterp

`acl_graph.py:345-403`

```python
class UpdatedNodeCaptureInterp(fx.Interpreter):
    """
    自定义解释器，在捕获时记录需要动态更新的节点
    """
    def __init__(self, graph_module: fx.GraphModule, need_updated_ops: Dict):
        super().__init__(graph_module)
        self._need_updated_ops = need_updated_ops  # {op_name: [param_name_list]}
        self._captured_node_info: List = []

    def run_node(self, node):
        # 如果不是需要更新的节点，正常执行
        if node.name not in self._need_updated_ops.keys():
            return super().run_node(node)

        # 创建 ExternalEvent 用于同步
        external_event = torch.npu.ExternalEvent()
        capture_stream = torch.npu.current_stream()
        external_event.wait(capture_stream)
        external_event.reset(capture_stream)

        # 使用 graph_task_group 包裹节点执行
        torch.npu.graph_task_group_begin(capture_stream)
        result = super().run_node(node)
        handle = torch.npu.graph_task_group_end(capture_stream)

        # 记录节点信息
        node_args, node_kwargs = self.fetch_args_kwargs_from_env(node)
        node_args, node_kwargs = reconstruct_args_kwargs(node_args, node_kwargs)

        self._captured_node_info.append(UpdatedNodeInfo(
            node_name=node.name,
            updated_func=node.target,
            updated_param_name=self._need_updated_ops[node.name],
            args=node_args,
            kwargs=node_kwargs,
            handle=handle,      # ← graph_task_group handle
            event=external_event # ← 同步事件
        ))

        return result
```

**关键点**：
- `graph_task_group_begin/end`：标记一组操作，后续可以通过 handle 动态更新
- `UpdatedNodeInfo`：记录节点的所有信息，包括更新句柄

---

### 3.5 重放阶段：动态更新参数

#### CapturedGraphUpdateAndReplay

`acl_graph.py:406-437`

```python
class CapturedGraphUpdateAndReplay(nn.Module):
    """
    捕获图的重放和参数更新模块
    """
    def __init__(self, replay_graph: Any, updated_input_func: Callable, updated_node_infos: List):
        super().__init__()
        self._replay_graph = replay_graph           # NPU Graph
        self._updated_input_func = updated_input_func # 计算更新值的函数
        self._updated_node_infos = updated_node_infos # UpdatedNodeInfo 列表
        self._update_stream = torch.npu.Stream()    # 专用更新流

    def forward(self, *args: Any, **kwargs: Any):
        # 1. 重放捕获的图
        self._replay_graph.replay()

        # 2. 计算需要更新的参数值
        updated_kwargs = self._updated_input_func(*args, **kwargs)
        # 格式：{op_name: {param_name: new_value}}

        if len(updated_kwargs) == 0:
            return

        # 3. 在专用流上更新参数
        with torch.npu.stream(self._update_stream):
            for node_info in self._updated_node_infos:
                # 3a. 开始更新任务
                torch.npu.graph_task_update_begin(self._update_stream, node_info.handle)

                # 3b. 构造新的 kwargs
                node_kwargs = dict(node_info.kwargs)
                for key in node_info.updated_param_name:
                    node_kwargs[key] = updated_kwargs[node_info.node_name][key]

                # 3c. 执行更新操作（会记录在图中）
                node_info.updated_func(*node_info.args, **node_kwargs)

                # 3d. 结束更新任务
                torch.npu.graph_task_update_end(self._update_stream)

                # 3e. 记录事件用于同步
                self._update_stream.record_event(node_info.event)

        return
```

**关键 API**：
- `torch.npu.graph_task_update_begin(stream, handle)`：开始更新指定任务组
- `torch.npu.graph_task_update_end(stream)`：结束更新
- 在 begin/end 之间执行的操作会替换图中对应的操作

---

## 4. 完整示例：DeepSeek V3 Prefill vs Decode

### 4.1 模型结构

```python
class DeepseekV3Attention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        actual_seq_lengths_kv: Optional[List[int]] = None,  # ← 可更新参数
        **kwargs,
    ):
        # ...

        attn_output = torch.ops.npu.npu_fused_infer_attention_score(
            query_states,
            past_key_value[0],
            past_key_value[1],
            pse_shift=None,
            atten_mask=attention_mask,
            actual_seq_lengths=None,
            actual_seq_lengths_kv=actual_seq_lengths_kv,  # ← 动态更新
            sparse_mode=4,
            scale_value=self.scaling,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            inner_precise=0,
        )

        return attn_output
```

### 4.2 Prefill 阶段

```python
# Prefill: 处理完整的输入序列
input_ids = torch.tensor([[1, 2, 3, ..., 256]])  # [1, 256]
attention_mask = create_tril_mask(256)           # [1, 256, 256]
position_ids = torch.arange(256).unsqueeze(0)    # [1, 256]

model_inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "past_key_values": kv_cache,
    "actual_seq_lengths_kv": None,  # ← Prefill 不使用
}

output = model(**model_inputs)
```

**Graph Key**：
```
graph_key = "[1,256],[1,256,256],[1,256],kv_cache_shapes"
```

- 所有 tensor shape 都参与 graph key 生成
- `actual_seq_lengths_kv=None` 不参与 graph key

### 4.3 Decode 阶段

```python
# Decode: 每次生成一个 token
for step in range(max_new_tokens):
    input_ids = next_token.unsqueeze(0)          # [1, 1]
    attention_mask = None                        # Decode 不需要 mask
    position_ids = kv_len.unsqueeze(1)           # [1, 1]

    # 计算每个 batch 的实际 KV 长度
    actual_seq_lengths_kv = (kv_len + 1).cpu().detach().numpy().tolist()
    # 例如：[256, 260, 258]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": kv_cache,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,  # ← 每次都不同！
    }

    output = model(**model_inputs)
```

**Graph Key**（Decode 阶段固定）：
```
graph_key = "[1,1],None,[1,1],kv_cache_shapes"
```

**动态更新的参数**：
```python
# 每次 Decode 时：
actual_seq_lengths_kv = [256, 260, 258]  # step 0
actual_seq_lengths_kv = [257, 261, 259]  # step 1
actual_seq_lengths_kv = [258, 262, 260]  # step 2
# ...
```

### 4.4 参数识别结果

#### 输入分类

| 参数名 | Prefill | Decode | 类型 | 参与 graph key |
|--------|---------|--------|------|---------------|
| `input_ids` | [1, 256] | [1, 1] | Unupdated | ✅ |
| `attention_mask` | [1, 256, 256] | None | Unupdated | ✅ |
| `position_ids` | [1, 256] | [1, 1] | Unupdated | ✅ |
| `past_key_values` | KV cache | KV cache | Unupdated | ✅ |
| `actual_seq_lengths_kv` | None | [256, 260, 258] | **Updated** | ❌ |

#### 识别流程

```python
# 1. 从 _REPLACE_FUNC_MAP 获取 updated_param_keys
updated_param_keys = ["actual_seq_lengths", "actual_seq_lengths_kv", "actual_shared_prefix_len"]

# 2. 遍历图节点，找到 attention 节点
for node in graph.nodes:
    if node.target == torch.ops.npu.npu_fused_infer_attention_score.out:
        # 3. 分析 node.kwargs
        if "actual_seq_lengths_kv" in node.kwargs:
            kwarg_value = node.kwargs["actual_seq_lengths_kv"]
            # kwarg_value 是一个 placeholder 节点（输入参数）

            # 4. 生成 update ruler
            input_index = placeholder_nodes.index(kwarg_value)
            update_ruler = [("index", input_index)]

            # 5. 记录该节点需要更新
            ops_update_rulers[node.name] = {
                "actual_seq_lengths_kv": update_ruler
            }

# 6. 在捕获时记录 UpdatedNodeInfo
# 7. 在重放时动态更新 actual_seq_lengths_kv
```

---

## 5. 工作原理总结

### 5.1 识别机制

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 算子注册 (_REPLACE_FUNC_MAP)                             │
│    - 定义哪些算子支持参数更新                                  │
│    - 定义每个算子的 updated_param_keys                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 输入分析 (get_unupdated_sym_input_index)                 │
│    - 遍历所有输入参数                                          │
│    - 检查每个输入的使用节点                                     │
│    - 分类为 Unupdated 或 Updated                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Graph Key 生成 (gen_unupdated_input_key)                │
│    - 只使用 Unupdated 输入生成 key                            │
│    - Updated 参数不影响 graph key                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 节点分析 (get_updated_ops_rulers_param)                  │
│    - 找到所有使用可更新算子的节点                               │
│    - 为每个节点生成 update ruler                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 捕获 (UpdatedNodeCaptureInterp)                          │
│    - 用 graph_task_group 包裹需要更新的节点                   │
│    - 记录 UpdatedNodeInfo                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. 重放 (CapturedGraphUpdateAndReplay)                      │
│    - replay_graph.replay() 重放主图                          │
│    - graph_task_update 动态更新参数                          │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 决策树

```
对于每个输入参数：
│
├─ 是否包含 SymInt？
│  ├─ 否 → 跳过（常量参数）
│  └─ 是 → 继续
│
└─ 该参数被哪些节点使用？
   │
   ├─ 被不支持更新的算子使用
   │  └─ → Unupdated（参与 graph key）
   │
   └─ 仅被支持更新的算子使用
      │
      ├─ 参数名不在 updated_param_keys 中
      │  └─ → Unupdated（参与 graph key）
      │
      └─ 参数名在 updated_param_keys 中
         └─ → Updated（不参与 graph key，可动态更新）
```

### 5.3 关键数据流

```python
# 编译时
_REPLACE_FUNC_MAP
    ↓
get_unupdated_sym_input_index()
    ↓
unupdated_input_index = [0, 1, 2, 3, 4]  # 前5个输入
    ↓
get_updated_ops_rulers_param()
    ↓
ops_update_rulers = {
    "attention_node_1": {
        "actual_seq_lengths_kv": [("index", 5)]
    }
}
    ↓
UpdatedNodeCaptureInterp.run()
    ↓
updated_node_infos = [
    UpdatedNodeInfo(
        node_name="attention_node_1",
        updated_param_name=["actual_seq_lengths_kv"],
        handle=<task_group_handle>,
        ...
    )
]

# 运行时
args = (input_ids, attn_mask, pos_ids, kv_cache, actual_seq_lengths_kv)
    ↓
gen_unupdated_input_key(args[0:5])  # 只用前5个生成 key
    ↓
graph_key = "[1,1],None,[1,1],..."
    ↓
replay_graph.replay()  # 重放主图
    ↓
updated_input_func(args)  # 计算更新值
    ↓
updated_kwargs = {
    "attention_node_1": {
        "actual_seq_lengths_kv": [256, 260, 258]
    }
}
    ↓
graph_task_update_begin(handle)
attention_func(..., actual_seq_lengths_kv=[256, 260, 258])
graph_task_update_end()
```

---

## 6. 使用建议

### 6.1 添加新的可更新算子

如果要支持新算子的参数动态更新：

```python
# 1. 在 _REPLACE_FUNC_MAP 中注册
_REPLACE_FUNC_MAP[torch.ops.npu.your_new_op.default] = StaticWorkspaceReplaceFunc(
    get_workspace=torch.ops.npu._your_new_op_get_workspace.default,
    out_operator=torch.ops.npu.your_new_op.out,
    workspace_keys=["workspace"],
    output_keys=["output"],
    updated_param_keys=["your_dynamic_param"],  # ← 声明可更新参数
)

# 2. 在模型中使用
output = torch.ops.npu.your_new_op(
    input_tensor,
    static_param=value,       # 固定参数（参与 graph key）
    your_dynamic_param=value, # 动态参数（不参与 graph key）
)
```

### 6.2 调试技巧

```python
# 1. 查看哪些输入被识别为 Unupdated
import logging
logging.getLogger("torchair").setLevel(logging.DEBUG)

# 输出示例：
# DEBUG - In graph[xxx], all unupdated sym input index is [0, 1, 2, 3, 4].

# 2. 查看哪些节点需要更新
# DEBUG - All need to be updated node names dict_keys(['attention_node_1'])
#         param names dict_values([['actual_seq_lengths_kv']]).

# 3. 查看运行时更新
# DEBUG - In AclGraph running, all updated op_name and param:
#         {'attention_node_1': {'actual_seq_lengths_kv': [256, 260, 258]}}.
```

### 6.3 性能优化

```python
# ✅ 推荐：明确使用可更新参数
attn_output = torch.ops.npu.npu_incre_flash_attention(
    query, key, value,
    num_heads=32,
    scale_value=0.125,
    atten_mask=None,                    # 固定参数
    actual_seq_lengths=kv_len,          # 动态参数
    actual_seq_lengths_kv=kv_len,       # 动态参数
    kv_padding_size=kv_padding_size,
)

# ❌ 不推荐：所有参数都固定
attn_output = torch.ops.npu.npu_incre_flash_attention(
    query[:, :actual_len, :],  # 动态切片 → 每次 shape 变化 → 重新捕获
    key[:, :actual_len, :],
    value[:, :actual_len, :],
    ...
)
```

---

## 7. 源码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| StaticWorkspaceReplaceFunc | acl_graph.py | 26-42 |
| UpdatedNodeInfo | acl_graph.py | 45-65 |
| _REPLACE_FUNC_MAP | acl_graph.py | 68-84 |
| get_unupdated_sym_input_index | acl_graph.py | 167-211 |
| get_update_ruler | acl_graph.py | 218-244 |
| get_updated_ops_rulers_param | acl_graph.py | 267-299 |
| UpdatedNodeCaptureInterp | acl_graph.py | 345-403 |
| CapturedGraphUpdateAndReplay | acl_graph.py | 406-437 |
| DeepSeek V3 Attention | modeling_deepseek.py | 1097-1173 |
| prepare_inputs_for_generation | modeling_deepseek.py | 2201-2256 |

---

## 8. 总结

aclgraph 的参数更新识别机制通过以下步骤实现：

1. **白名单注册**：在 `_REPLACE_FUNC_MAP` 中明确声明哪些算子的哪些参数可以动态更新
2. **输入分析**：遍历图的所有输入，根据其使用情况分类为 Unupdated 或 Updated
3. **Graph Key 生成**：只使用 Unupdated 输入生成 key，Updated 参数不影响 key
4. **节点分析**：为每个使用可更新算子的节点生成 update ruler
5. **捕获记录**：使用 graph_task_group 记录需要更新的节点信息
6. **动态更新**：重放时通过 graph_task_update API 动态更新参数值

这种机制的核心优势：
- **减少重新捕获**：Updated 参数变化不触发重新捕获
- **保持灵活性**：可以在固定 shape 的前提下动态调整业务逻辑参数
- **性能优化**：避免 padding 计算浪费，提升实际推理效率

通过这种机制，torchair 的 reduce-overhead 模式实现了**图捕获的低开销**和**参数更新的灵活性**之间的平衡。
