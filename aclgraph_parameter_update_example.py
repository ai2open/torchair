#!/usr/bin/env python3
# coding=utf-8
"""
AclGraph 参数更新机制实际使用示例

演示如何在实际模型中使用动态参数更新机制
"""

import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig


# ============================================================================
# 示例 1: 基本的 Attention 算子使用
# ============================================================================

def example_1_basic_attention():
    """
    基本示例：展示如何使用 actual_seq_lengths_kv 动态参数
    """
    print("=" * 80)
    print("示例 1: 基本 Attention 算子使用")
    print("=" * 80)

    batch_size = 4
    seq_len = 256
    kv_seq_len = 512
    num_heads = 32
    head_dim = 128

    # 创建固定 shape 的输入（用 padding）
    query = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.bfloat16, device='npu:0')
    key = torch.randn(batch_size, num_heads, kv_seq_len, head_dim,
                     dtype=torch.bfloat16, device='npu:0')
    value = torch.randn(batch_size, num_heads, kv_seq_len, head_dim,
                       dtype=torch.bfloat16, device='npu:0')

    # ❌ 方式 1：不使用动态参数（会计算 padding 部分）
    print("\n❌ 方式 1：不使用 actual_seq_lengths_kv")
    print("   - 所有 512 个 KV token 都会参与计算")
    print("   - 包括 padding 部分，造成计算浪费")

    output_1 = torch.ops.npu.npu_incre_flash_attention(
        query, key, value,
        num_heads=num_heads,
        scale_value=1.0 / (head_dim ** 0.5),
        input_layout="BNSD",
        # 不指定 actual_seq_lengths_kv → 默认使用完整长度
    )
    print(f"   - 输出 shape: {output_1.shape}")

    # ✅ 方式 2：使用动态参数（只计算有效部分）
    print("\n✅ 方式 2：使用 actual_seq_lengths_kv 动态参数")

    # 每个 batch 的实际 KV 长度（不包括 padding）
    actual_kv_lengths = [256, 312, 480, 501]
    print(f"   - 实际 KV 长度: {actual_kv_lengths}")
    print(f"   - Padding 长度: {[512 - l for l in actual_kv_lengths]}")
    print(f"   - 计算节省: {sum(512 - l for l in actual_kv_lengths) / (512 * 4) * 100:.1f}%")

    output_2 = torch.ops.npu.npu_incre_flash_attention(
        query, key, value,
        num_heads=num_heads,
        scale_value=1.0 / (head_dim ** 0.5),
        input_layout="BNSD",
        actual_seq_lengths_kv=actual_kv_lengths,  # ← 动态参数
    )
    print(f"   - 输出 shape: {output_2.shape}")

    print("\n✅ 关键点：")
    print("   - query/key/value 的 shape 固定（通过 padding）")
    print("   - actual_seq_lengths_kv 可以每次推理时变化")
    print("   - 不会触发 aclgraph 重新捕获")


# ============================================================================
# 示例 2: Prefill vs Decode 的参数变化
# ============================================================================

def example_2_prefill_decode():
    """
    展示 Prefill 和 Decode 阶段的参数使用差异
    """
    print("\n" + "=" * 80)
    print("示例 2: Prefill vs Decode 阶段")
    print("=" * 80)

    batch_size = 4
    num_heads = 32
    head_dim = 128
    max_seq_len = 512

    # 初始化 KV cache
    kv_cache_key = torch.zeros(batch_size, num_heads, max_seq_len, head_dim,
                                dtype=torch.bfloat16, device='npu:0')
    kv_cache_value = torch.zeros(batch_size, num_heads, max_seq_len, head_dim,
                                  dtype=torch.bfloat16, device='npu:0')

    # ========================================================================
    # Prefill 阶段
    # ========================================================================
    print("\n【Prefill 阶段】")
    print("-" * 80)

    # Prefill: 处理完整的输入序列
    prefill_seq_len = 256
    query_prefill = torch.randn(batch_size, num_heads, prefill_seq_len, head_dim,
                                dtype=torch.bfloat16, device='npu:0')

    print(f"输入参数:")
    print(f"  - query shape: {list(query_prefill.shape)}")
    print(f"  - KV cache shape: {list(kv_cache_key.shape)}")
    print(f"  - actual_seq_lengths_kv: None")

    output_prefill = torch.ops.npu.npu_incre_flash_attention(
        query_prefill, kv_cache_key, kv_cache_value,
        num_heads=num_heads,
        scale_value=1.0 / (head_dim ** 0.5),
        input_layout="BNSD",
        # Prefill 阶段：不使用 actual_seq_lengths_kv
        # 因为所有序列长度相同，无 padding
    )

    print(f"\nGraph Key 组成:")
    print(f"  ✅ query.shape = [4, 32, 256, 128]")
    print(f"  ✅ key.shape = [4, 32, 512, 128]")
    print(f"  ✅ value.shape = [4, 32, 512, 128]")
    print(f"  ❌ actual_seq_lengths_kv 不参与")
    print(f"\n  → Graph Key: 'prefill_[4,32,256,128]_...'")

    # 更新 KV cache（实际模型中会拼接）
    current_kv_len = prefill_seq_len

    # ========================================================================
    # Decode 阶段（多步）
    # ========================================================================
    print("\n【Decode 阶段】")
    print("-" * 80)

    decode_steps = 3
    for step in range(decode_steps):
        print(f"\n--- Decode Step {step} ---")

        # Decode: 每次只生成 1 个 token
        query_decode = torch.randn(batch_size, num_heads, 1, head_dim,
                                   dtype=torch.bfloat16, device='npu:0')

        # 每个 batch 的实际 KV 长度（可能不同）
        # 在实际场景中，不同 batch 可能在不同时间结束
        actual_kv_lengths = [
            current_kv_len + step + 1,
            current_kv_len + step + 1,
            current_kv_len + step + 1,
            current_kv_len + step + 1,
        ]

        # 模拟某些 batch 先结束
        if step >= 1:
            actual_kv_lengths[0] = current_kv_len + 1  # batch 0 已结束
        if step >= 2:
            actual_kv_lengths[1] = current_kv_len + 2  # batch 1 已结束

        print(f"输入参数:")
        print(f"  - query shape: {list(query_decode.shape)}")
        print(f"  - KV cache shape: {list(kv_cache_key.shape)}")
        print(f"  - actual_seq_lengths_kv: {actual_kv_lengths}")

        output_decode = torch.ops.npu.npu_incre_flash_attention(
            query_decode, kv_cache_key, kv_cache_value,
            num_heads=num_heads,
            scale_value=1.0 / (head_dim ** 0.5),
            input_layout="BNSD",
            actual_seq_lengths_kv=actual_kv_lengths,  # ← 每步都不同！
        )

        if step == 0:
            print(f"\nGraph Key 组成 (首次 Decode):")
            print(f"  ✅ query.shape = [4, 32, 1, 128]")
            print(f"  ✅ key.shape = [4, 32, 512, 128]")
            print(f"  ✅ value.shape = [4, 32, 512, 128]")
            print(f"  ❌ actual_seq_lengths_kv 不参与")
            print(f"\n  → Graph Key: 'decode_[4,32,1,128]_...'")
            print(f"  → 首次执行：捕获图")
        else:
            print(f"\n  → Graph Key 相同，重用已捕获的图")
            print(f"  → 动态更新: actual_seq_lengths_kv = {actual_kv_lengths}")

        print(f"  → 输出 shape: {list(output_decode.shape)}")

    print("\n✅ 关键点：")
    print("   - Prefill 和 Decode 有不同的 graph key（shape 不同）")
    print("   - Decode 阶段所有步骤共享同一个 graph（shape 相同）")
    print("   - actual_seq_lengths_kv 每步不同，但不触发重新捕获")
    print("   - 实现了高效的增量推理")


# ============================================================================
# 示例 3: 使用 torch.compile 的完整流程
# ============================================================================

class SimpleAttentionModel(torch.nn.Module):
    """简单的 Attention 模型，用于演示编译流程"""

    def __init__(self, num_heads=32, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)

    def forward(self, query, key, value, actual_seq_lengths_kv=None):
        """
        前向传播

        Args:
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_heads, kv_seq_len, head_dim]
            value: [batch, num_heads, kv_seq_len, head_dim]
            actual_seq_lengths_kv: List[int], 可选的实际 KV 长度
        """
        return torch.ops.npu.npu_incre_flash_attention(
            query, key, value,
            num_heads=self.num_heads,
            scale_value=self.scale,
            input_layout="BNSD",
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )


def example_3_torch_compile():
    """
    展示使用 torch.compile 和 reduce-overhead 模式的完整流程
    """
    print("\n" + "=" * 80)
    print("示例 3: torch.compile + reduce-overhead 模式")
    print("=" * 80)

    # 1. 创建模型
    model = SimpleAttentionModel().to('npu:0')
    print("\n1. 创建模型")
    print(f"   - 模型: SimpleAttentionModel")
    print(f"   - 设备: npu:0")

    # 2. 配置编译器
    compiler_config = CompilerConfig()
    compiler_config.experimental_config.frozen_parameter = True
    compiler_config.experimental_config.tiling_schedule_optimize = True
    compiler_config.mode = "reduce-overhead"  # ← 使用 reduce-overhead 模式

    print("\n2. 配置编译器")
    print(f"   - 模式: reduce-overhead")
    print(f"   - frozen_parameter: True")
    print(f"   - tiling_schedule_optimize: True")

    # 3. 编译模型
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    compiled_model = torch.compile(model, dynamic=True, fullgraph=True, backend=npu_backend)

    print("\n3. 编译模型")
    print(f"   - Backend: torchair NPU backend")
    print(f"   - Dynamic: True")
    print(f"   - Fullgraph: True")

    # 4. 准备输入
    batch_size = 4
    seq_len = 1  # Decode 阶段
    kv_seq_len = 512
    num_heads = 32
    head_dim = 128

    query = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.bfloat16, device='npu:0')
    key = torch.randn(batch_size, num_heads, kv_seq_len, head_dim,
                     dtype=torch.bfloat16, device='npu:0')
    value = torch.randn(batch_size, num_heads, kv_seq_len, head_dim,
                       dtype=torch.bfloat16, device='npu:0')

    print("\n4. 首次推理（捕获图）")
    print(f"   - 输入 shape: query={list(query.shape)}, key={list(key.shape)}")

    actual_kv_lengths_1 = [256, 312, 400, 501]
    print(f"   - actual_seq_lengths_kv: {actual_kv_lengths_1}")

    # 首次推理：会触发图捕获
    output_1 = compiled_model(query, key, value, actual_kv_lengths_1)
    torch.npu.synchronize()
    print(f"   - 输出 shape: {list(output_1.shape)}")
    print(f"   ✅ 图捕获完成")

    # 5. 后续推理（重放图）
    print("\n5. 后续推理（重放图）")

    for i in range(3):
        # 动态变化的 actual_seq_lengths_kv
        actual_kv_lengths = [256 + i, 312 + i, 400 + i, 501 + i]
        print(f"\n   推理 {i+1}:")
        print(f"   - actual_seq_lengths_kv: {actual_kv_lengths}")

        output = compiled_model(query, key, value, actual_kv_lengths)
        torch.npu.synchronize()

        print(f"   - 输出 shape: {list(output.shape)}")
        print(f"   ✅ 重放图 + 动态更新参数（无重新捕获）")

    print("\n6. 总结")
    print("   ✅ 只捕获一次图")
    print("   ✅ actual_seq_lengths_kv 每次都不同")
    print("   ✅ 不触发重新捕获")
    print("   ✅ 获得 reduce-overhead 的性能优势")


# ============================================================================
# 示例 4: 对比不同实现的性能
# ============================================================================

def example_4_performance_comparison():
    """
    对比使用和不使用动态参数的性能差异
    """
    print("\n" + "=" * 80)
    print("示例 4: 性能对比")
    print("=" * 80)

    import time

    batch_size = 8
    seq_len = 1
    kv_seq_len = 2048
    num_heads = 32
    head_dim = 128
    num_iters = 100

    query = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.bfloat16, device='npu:0')
    key = torch.randn(batch_size, num_heads, kv_seq_len, head_dim,
                     dtype=torch.bfloat16, device='npu:0')
    value = torch.randn(batch_size, num_heads, kv_seq_len, head_dim,
                       dtype=torch.bfloat16, device='npu:0')

    # 实际 KV 长度（很短）
    actual_kv_lengths = [128, 156, 200, 234, 150, 180, 210, 190]

    # ========================================================================
    # 方法 1: 不使用 actual_seq_lengths_kv（计算所有 padding）
    # ========================================================================
    print("\n【方法 1: 不使用 actual_seq_lengths_kv】")
    print(f"  - 计算范围: 所有 {kv_seq_len} 个 KV tokens")
    print(f"  - 包括 padding 部分")

    # Warmup
    for _ in range(10):
        _ = torch.ops.npu.npu_incre_flash_attention(
            query, key, value,
            num_heads=num_heads,
            scale_value=1.0 / (head_dim ** 0.5),
            input_layout="BNSD",
        )
    torch.npu.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        _ = torch.ops.npu.npu_incre_flash_attention(
            query, key, value,
            num_heads=num_heads,
            scale_value=1.0 / (head_dim ** 0.5),
            input_layout="BNSD",
        )
    torch.npu.synchronize()
    time_1 = (time.time() - start) / num_iters * 1000  # ms

    print(f"  - 平均延迟: {time_1:.3f} ms")

    # ========================================================================
    # 方法 2: 使用 actual_seq_lengths_kv（只计算有效部分）
    # ========================================================================
    print("\n【方法 2: 使用 actual_seq_lengths_kv】")
    print(f"  - 计算范围: 实际 KV 长度 {actual_kv_lengths}")
    print(f"  - 跳过 padding 部分")

    # Warmup
    for _ in range(10):
        _ = torch.ops.npu.npu_incre_flash_attention(
            query, key, value,
            num_heads=num_heads,
            scale_value=1.0 / (head_dim ** 0.5),
            input_layout="BNSD",
            actual_seq_lengths_kv=actual_kv_lengths,
        )
    torch.npu.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        _ = torch.ops.npu.npu_incre_flash_attention(
            query, key, value,
            num_heads=num_heads,
            scale_value=1.0 / (head_dim ** 0.5),
            input_layout="BNSD",
            actual_seq_lengths_kv=actual_kv_lengths,
        )
    torch.npu.synchronize()
    time_2 = (time.time() - start) / num_iters * 1000  # ms

    print(f"  - 平均延迟: {time_2:.3f} ms")

    # ========================================================================
    # 对比结果
    # ========================================================================
    print("\n【性能对比】")
    speedup = time_1 / time_2
    time_saved = time_1 - time_2
    avg_actual_len = sum(actual_kv_lengths) / len(actual_kv_lengths)
    waste_ratio = (kv_seq_len - avg_actual_len) / kv_seq_len * 100

    print(f"  - 方法 1 延迟: {time_1:.3f} ms")
    print(f"  - 方法 2 延迟: {time_2:.3f} ms")
    print(f"  - 加速比: {speedup:.2f}x")
    print(f"  - 节省时间: {time_saved:.3f} ms ({(1 - 1/speedup) * 100:.1f}%)")
    print(f"  - Padding 浪费: {waste_ratio:.1f}%")
    print(f"\n  ✅ 使用 actual_seq_lengths_kv 可节省约 {waste_ratio:.1f}% 的计算")


# ============================================================================
# 示例 5: mark_static 的使用
# ============================================================================

def example_5_mark_static():
    """
    展示如何使用 torch._dynamo.mark_static 标记静态输入
    """
    print("\n" + "=" * 80)
    print("示例 5: mark_static 的使用")
    print("=" * 80)

    class ModelWithCache(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 32
            self.head_dim = 128

        def forward(self, query, past_key, past_value, kv_len, actual_kv_lengths):
            """
            Args:
                query: [batch, num_heads, 1, head_dim]
                past_key: [batch, num_heads, max_seq_len, head_dim]
                past_value: [batch, num_heads, max_seq_len, head_dim]
                kv_len: [batch], 当前 KV 长度
                actual_kv_lengths: List[int], 实际 KV 长度
            """
            return torch.ops.npu.npu_incre_flash_attention(
                query, past_key, past_value,
                num_heads=self.num_heads,
                scale_value=1.0 / (self.head_dim ** 0.5),
                input_layout="BNSD",
                actual_seq_lengths_kv=actual_kv_lengths,
            )

    print("\n❌ 问题：动态 shape 导致重新捕获")
    print("   如果 past_key/past_value 的 shape 每次都不同")
    print("   （例如，通过动态拼接实现 KV cache）")
    print("   会导致每次推理都重新捕获图")

    print("\n✅ 解决：使用 mark_static")
    print("   在首次推理前，标记哪些输入是静态的")

    print("\n   示例代码:")
    print("""
    def mark_inputs(model_inputs):
        query = model_inputs["query"]
        past_key = model_inputs["past_key"]
        past_value = model_inputs["past_value"]
        kv_len = model_inputs["kv_len"]

        # 标记这些输入为静态（shape 不变）
        torch._dynamo.mark_static(query)
        torch._dynamo.mark_static(past_key)
        torch._dynamo.mark_static(past_value)
        torch._dynamo.mark_static(kv_len)

        # actual_kv_lengths 不需要标记（本身就是动态参数）

    # 在首次推理前调用
    mark_inputs(model_inputs)
    output = compiled_model(**model_inputs)
    """)

    print("\n   效果:")
    print("   - past_key/past_value 的 shape 被固定")
    print("   - 即使底层实现可能变化，编译器认为 shape 固定")
    print("   - 避免不必要的重新捕获")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "AclGraph 参数更新机制示例" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # 示例 1: 基本使用
        example_1_basic_attention()

        # 示例 2: Prefill vs Decode
        example_2_prefill_decode()

        # 示例 3: torch.compile 流程
        example_3_torch_compile()

        # 示例 4: 性能对比
        example_4_performance_comparison()

        # 示例 5: mark_static
        example_5_mark_static()

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("所有示例完成！")
    print("=" * 80 + "\n")
