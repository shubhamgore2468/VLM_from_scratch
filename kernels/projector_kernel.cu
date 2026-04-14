#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void layernorm_kernel(const float* input, float* output,
                                 const float* gamma, const float* beta, int dim) {
    extern __shared__ float sdata[];
    float* s_sum  = sdata;
    float* s_sum2 = sdata + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = input + row * dim;

    float local_sum = 0.0f, local_sum2 = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = row_in[i];
        local_sum  += val;
        local_sum2 += val * val;
    }
    s_sum[tid]  = local_sum;
    s_sum2[tid] = local_sum2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]  += s_sum[tid + stride];
            s_sum2[tid] += s_sum2[tid + stride];
        }
        __syncthreads();
    }

    float mean    = s_sum[0] / dim;
    float var     = s_sum2[0] / dim - mean * mean;
    float inv_std = rsqrtf(var + 1e-5f);

    float* row_out = output + row * dim;
    for (int i = tid; i < dim; i += blockDim.x) {
        row_out[i] = gamma[i] * ((row_in[i] - mean) * inv_std) + beta[i];
    }
}

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

torch::Tensor layernorm_forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta) {
    TORCH_CHECK(input.is_cuda() && gamma.is_cuda() && beta.is_cuda());
    TORCH_CHECK(input.is_contiguous());
    auto output = torch::empty_like(input);
    int dim  = input.size(-1);
    int rows = input.numel() / dim;
    int threads = 256;
    layernorm_kernel<<<rows, threads, 2 * threads * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(), dim);
    return output;
}

torch::Tensor gelu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    gelu_kernel<<<(n + threads - 1) / threads, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm (CUDA)");
    m.def("gelu_forward", &gelu_forward, "GELU (CUDA)");
}