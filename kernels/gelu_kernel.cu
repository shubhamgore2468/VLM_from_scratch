// Writing a FUSED KERNEL FOR GELU + LAYERNORM + LINEAR TRANSFORM

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include<torch/extension.h>

__global__ void gelu(const float * input, float * output, int n){

    // TODO: Implement the GELU activation function
    // The GELU function can be approximated as:
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i>=n) return;
    float x = input[i];
    output[i] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));

}

torch::Tensor gelu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    //allocate memory
    auto output = torch::empty_like(input);
    int n = input.numel(); // (1024, 768) → 1024 * 768 = 786,432. flattens the array to 1D
    int threadsPerBlock = 256;
    int blocksPergrid = ( N + threadsPerBlock - 1) / threadsPerBlock; // ceil(N / threadsPerBlock)

    gelu<<<blocksPerGrid, threadsPerBlock>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    )
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward (CUDA)");
}