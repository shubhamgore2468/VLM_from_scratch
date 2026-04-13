from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="custom_gelu",
    ext_modules=[
        CUDAExtension(
            name="custom_gelu",
            sources=["gelu_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)