#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>

// CUDA kernel for RSA encryption\_
decryption (unchanged)
__global__ void rsa_kernel_off(unsigned long long *msg, unsigned long long *out, unsigned long long key, unsigned long long n, int offset, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        unsigned long long m = msg[offset + idx];
        unsigned long long c = 1;
        for (unsigned long long i = 0; i < key; i++) {
            c = (c * m) % n;
        }
        out[offset + idx] = c;
    }
}

int main() {
    unsigned long long n = 3233, e = 17, d = 2753; // Example RSA keys
    char paragraph[2048];

    printf("Enter paragraph: ");
    fgets(paragraph, sizeof(paragraph), stdin);

    int msg_len = strlen(paragraph);
    unsigned long long *h_msg = (unsigned long long *)malloc(msg_len * sizeof(unsigned long long));
    unsigned long long *h_enc = (unsigned long long *)malloc(msg_len * sizeof(unsigned long long));
    unsigned long long *h_dec = (unsigned long long *)malloc(msg_len * sizeof(unsigned long long));

    for (int i = 0; i < msg_len; i++) h_msg[i] = (unsigned long long)paragraph[i];

    unsigned long long *d_msg, *d_enc, *d_dec;
    cudaMalloc(&d_msg, msg_len * sizeof(unsigned long long));
    cudaMalloc(&d_enc, msg_len * sizeof(unsigned long long));
    cudaMalloc(&d_dec, msg_len * sizeof(unsigned long long));

    cudaMemcpy(d_msg, h_msg, msg_len * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    int num_streams = 4;
    cudaStream_t streams[4];
    for (int i = 0; i < num_streams; i++) cudaStreamCreate(&streams[i]);

    int total_size = msg_len;
    int chunk_size = total_size / num_streams;
    int threads = 256;

    // Timing setup
    auto start_enc = std::chrono::high_resolution_clock::now();

    // Encryption kernel per stream
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        int len = (i == num_streams - 1) ? (total_size - offset) : chunk_size;
        int blocks = (len + threads - 1) / threads;
        rsa_kernel_off<<<blocks, threads, 0, streams[i]>>>(d_msg, d_enc, e, n, offset, len);
    }
    cudaDeviceSynchronize();

    auto stop_enc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> enc_time = stop_enc - start_enc;

    // Copy encrypted back
    cudaMemcpy(h_enc, d_enc, msg_len * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    auto start_dec = std::chrono::high_resolution_clock::now();

    // Decryption kernel per stream
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        int len = (i == num_streams - 1) ? (total_size - offset) : chunk_size;
        int blocks = (len + threads - 1) / threads;
        rsa_kernel_off<<<blocks, threads, 0, streams[i]>>>(d_enc, d_dec, d, n, offset, len);
    }
    cudaDeviceSynchronize();

    auto stop_dec = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dec_time = stop_dec - start_dec;

    cudaMemcpy(h_dec, d_dec, msg_len * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("\nEncrypted text:\n");
    for (int i = 0; i < msg_len; i++) printf("%llu ", h_enc[i]);

    printf("\n\nDecrypted text:\n");
    for (int i = 0; i < msg_len; i++) printf("%c", (char)h_dec[i]);

    printf("\n\nExecution Time:\nEncryption: %.3f ms\nDecryption: %.3f ms\n", enc_time.count(), dec_time.count());

    for (int i = 0; i < num_streams; i++) cudaStreamDestroy(streams[i]);
    cudaFree(d_msg);
    cudaFree(d_enc);
    cudaFree(d_dec);
    free(h_msg);
    free(h_enc);
    free(h_dec);

    return 0;
}