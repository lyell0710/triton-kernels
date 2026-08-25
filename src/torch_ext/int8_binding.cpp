// torch extension 绑定:零改动复用 Kernel_Optimazation(EXP-K01)的
// quantize_v4 CUDA kernel,接进 torch 生态做同尺寸对照(EXP-T03)。
// 讲解点:.cu 保持原样零改动——绑定层只做 张量校验 + scale 计算 + 指针透传。
//
// 接口契约:input (channels, hw) fp32 CUDA 张量 → {int8 q, fp32 per-channel
// scale}。v4 kernel 的签名要求 scale 预置,故 scale 只能在绑定层用 torch 算。
//
// 面试点——性能口径(EXP-T03,三数字不得混引,LEDGER 红线):裸 kernel
// 5.9µs(scale 预置口径)≠ 本绑定端到端 65.1µs:abs/max/div 三个前置
// torch launch + 分发开销吃掉了 kernel 的全部速度优势,被 Triton 单
// kernel 融合(41.6~52µs)反超——融合数(launch 数)比单 kernel 快慢
// 更重要,这就是"更快的 kernel 输掉端到端"的机理。
#include <torch/extension.h>

// 来自 Kernel_Optimazation 的原始符号(裸指针签名),链接期由 .cu 提供;
// 不改签名是"零改动复用"的边界所在
void quantize_v4(const float* input, const float* scales, int8_t* output,
                 int channels, int hw);

std::vector<torch::Tensor> int8_quantize_v4(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32,
                "expect fp32 CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "expect (channels, hw)");
    // kernel 以 (channels, hw) 扁平指针 + 行主序寻址,必须保证内存连续
    auto x = input.contiguous();
    // per-channel absmax/127 对称量化;clamp_min 防全零通道除 0。
    // 这一行展开即 abs→max→div 三次独立 kernel launch——端到端 65.1µs
    // 与裸 kernel 5.9µs 之间的差距主要在此(见文件头口径说明)
    auto scales = (std::get<0>(x.abs().max(1)) / 127.0f).clamp_min(1e-8f);
    auto out = torch::empty_like(x, torch::kInt8);
    quantize_v4(x.data_ptr<float>(), scales.data_ptr<float>(),
                out.data_ptr<int8_t>(), x.size(0), x.size(1));
    return {out, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_quantize_v4", &int8_quantize_v4,
          "per-channel INT8 quantize (CUDA v4 kernel from Kernel_Optimazation)");
}
