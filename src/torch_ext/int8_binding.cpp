// torch extension 绑定:复用 Kernel_Optimazation 的 quantize_v4 CUDA kernel。
// 讲解点:.cu 保持原样零改动——绑定层只做 张量校验 + scale 计算 + 指针透传。
#include <torch/extension.h>

void quantize_v4(const float* input, const float* scales, int8_t* output,
                 int channels, int hw);

std::vector<torch::Tensor> int8_quantize_v4(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32,
                "expect fp32 CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "expect (channels, hw)");
    auto x = input.contiguous();
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
