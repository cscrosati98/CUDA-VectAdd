//C[i]= A[i]+B[i]
#include <stdio.h>
#include <cuda.h>


#define N 1024

__global__ void vector_add(float *A, float *B, float *C){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N){
        C[i]=A[i]+B[i];
    }

}

int main(){
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    float ker, htod, dtoh, total;
    cudaEvent_t start, stop;
    cudaEvent_t kstart, kstop;
    cudaEvent_t hstart, hstop;
    cudaEvent_t dstart, dstop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kstart);cudaEventCreate(&hstart);cudaEventCreate(&dstart);
    cudaEventCreate(&kstop);cudaEventCreate(&hstop);cudaEventCreate(&dstop);

    cudaEventRecord(start);
    size_t size=N*sizeof(float);

    h_A=(float*) malloc(size);
    h_B=(float*) malloc(size);
    h_C=(float*) malloc(size);

    for (int i=0; i<N; i++){
        h_A[i]=i;
        h_B[i]=2*i;
    }
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaEventRecord(hstart);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(hstop);
    cudaEventSynchronize(hstop);
    cudaEventElapsedTime(&htod, hstart,hstop);
    printf("DTOH Time: %f ms\n",htod);

    int threadsPerBlock =32;
    int blocks = (N+threadsPerBlock-1)/threadsPerBlock;
    cudaEventRecord(kstart);
    vector_add<<<blocks,threadsPerBlock>>>(d_A,d_B,d_C);
    cudaEventRecord(kstop);
    cudaEventSynchronize(kstop);
    cudaEventElapsedTime(&ker, kstart,kstop);
    printf("Kernel Execution Time: %f ms\n",ker);

    cudaEventRecord(dstart);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(dstop);
    cudaEventSynchronize(dstop);
    cudaEventElapsedTime(&dtoh, dstart,dstop);
    printf("HTOD Time: %f ms\n",dtoh);
    printf("Total Time: %f\n", ker+dtoh+htod);
    for(int i =0; i<N;i++){
        if (h_C[i]!=h_A[i]+h_B[i]){
            printf("C[%d]=%f\n", i, h_C[i]);
            break;
        }
        if(i+1==N){
            printf("All %d matched\n", N);
        }
    }
    free(h_A);free(h_B);free(h_C);
    cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total, start,stop);
    printf("Total Time: %f ms\n",total);
}