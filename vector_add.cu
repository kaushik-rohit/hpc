
__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = threadIdx.x + blockIdx.x * blockIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int N) {
    int size = N * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<(N + 255) / 256, 256>>>(A_d, B_d, C_d, N);
    
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int N = 1000;
    float* A = (float*) malloc(N*sizeof(float)), *B = (float*) malloc(N*sizeof(float)), *C = (float*) malloc(N*sizeof(float));
    vecAdd(A, B, C, N);
}