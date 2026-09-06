"""Compile a pinned upstream GDN operator for Thor, in its own namespace."""
from pathlib import Path
from torch.utils.cpp_extension import load

root = Path('/opt/thor-gdn')
(root / 'build').mkdir(exist_ok=True)
load(name='thor_gdn', sources=[str(root/'bindings.cpp'),
     str(root/'csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu')],
     extra_cflags=['-O3', '-std=c++20', '-DUSE_CUDA'],
     extra_cuda_cflags=['-O3', '-std=c++20', '-DUSE_CUDA'],
     build_directory=str(root/'build'), is_python_module=False, verbose=True)
