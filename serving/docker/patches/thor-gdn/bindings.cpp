#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <string>
using torch::stable::Tensor;
void fused_gdn_decode_post_conv_mtp(
    Tensor const&, Tensor const&, Tensor const&, Tensor const&, Tensor const&,
    Tensor const&, Tensor const&, Tensor const&, Tensor&, Tensor const&,
    Tensor const&, Tensor&, double, double, const std::string&);

STABLE_TORCH_LIBRARY(thor_gdn, m) {
  m.def("fused_gdn_decode_post_conv_mtp("
        "Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
        "Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens, "
        "Tensor! state, Tensor output_gate, Tensor norm_weight, Tensor! out, "
        "float scale, float norm_eps=1e-5, "
        "str output_gate_activation='silu') -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(thor_gdn, CUDA, m) {
  m.impl("fused_gdn_decode_post_conv_mtp", TORCH_BOX(&fused_gdn_decode_post_conv_mtp));
}
