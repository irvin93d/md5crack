#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "md5.hpp"


#define MAX_PASSWORD_LEN 256
#define DIGEST_SIZE 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void crackMD5(char* hash_in, char* pass_set, uint32_t len, char* pass_out) {
    // TODO
    int id = threadIdx.x;
    if(id == 0) {
        char * pass_test = pass_set + MAX_PASSWORD_LEN * id;
        char pass_cache[MAX_PASSWORD_LEN];
        int pass_len = 0;
        // Find the length of the string
        while(pass_test[pass_len]) {
            pass_cache[pass_len] = pass_test[pass_len];
            ++pass_len;
        }

        pass_cache[pass_len] = '\0';
        MD5 md5(pass_cache, pass_len);
              
        unsigned char result[DIGEST_SIZE]; // 128 bit
        md5.get_digest(result); // load the result
        
        memcpy(pass_out, result, DIGEST_SIZE);
    }
}

int main(int argc, char const ** argv) {
    // TODO load a file and stuff
    std::string hash = "5f4dcc3b5aa765d61d8327deb882cf99"; // 'password'
    std::string dict = "password";

    std::cout << hash << std::endl;

    char * d_pass_out;
    char * d_hash_in;
    char * d_passwords;

    gpuErrchk(cudaMalloc((void**) &d_pass_out, MAX_PASSWORD_LEN));
    gpuErrchk(cudaMalloc((void**) &d_hash_in, MAX_PASSWORD_LEN));
    gpuErrchk(cudaMalloc((void**) &d_passwords, 256));

    gpuErrchk(cudaMemcpy(d_hash_in, hash.c_str(), hash.length() + 1, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_passwords, dict.c_str(), dict.length() + 1 , cudaMemcpyHostToDevice));

    crackMD5<<<1,1>>>(d_hash_in, d_passwords, 1, d_pass_out);
    
    unsigned char result[MAX_PASSWORD_LEN];
    cudaMemcpy(result, d_pass_out, MAX_PASSWORD_LEN, cudaMemcpyDeviceToHost);
    
    
    std::cout << hexdigest(result) << std::endl;

    return 0;
}
