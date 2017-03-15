#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "md5.hpp"

#define PASSWORDS_PER_KERNEL 4
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

__global__ void crackMD5(unsigned char* hash_in, char* pass_set, uint32_t len, char* pass_out) {
    // TODO
	unsigned char hash_in_cache[DIGEST_SIZE];
	memcpy(hash_in_cache, hash_in, DIGEST_SIZE);

	// TODO CREATE LOOP WHER IF IS
    for(int id = threadIdx.x + blockIdx.x*blockDim.x ; id < len ; id += gridDim.x*blockDim.x) {
		// Init varibles for password test
        char * pass_test = pass_set + MAX_PASSWORD_LEN * id;
        char pass_cache[MAX_PASSWORD_LEN];
        int pass_len = 0;

		printf("Thread%d: %s\n", threadIdx.x, pass_set+id*MAX_PASSWORD_LEN);
        // Copy and find the length of the password to test
        while(pass_test[pass_len]) {
            pass_cache[pass_len] = pass_test[pass_len];
            ++pass_len;
        }
        pass_cache[pass_len] = 0;
       
	   	// Create hash for password to test
		MD5 md5(pass_cache, pass_len);
              
		// Retrieve created hash
        unsigned char result[DIGEST_SIZE]; // 128 bit
        md5.get_digest(result); // load the result
    
		// Test created hash against hash to crack
		int success = 1;
		for(int i = 0 ; i < DIGEST_SIZE ; ++i ) {
			if(result[i] != hash_in_cache[i]) {	
				success = 0;	
				break;
			}
		}

		// If crack is successful, return result
		if(success) {
			memcpy(pass_out, pass_cache, DIGEST_SIZE);
    	}
	}
}

int main(int argc, char const ** argv) {
    // TODO load a file and stuff

    std::string hash = "5f4dcc3b5aa765d61d8327deb882cf99"; // 'password'

    std::cout << hash << std::endl;

	unsigned char hash_in[17];
	strcpy( (char*) hash_in, hexencode(hash.c_str() ).c_str());

	// device declerations
    char * d_pass_out;
    unsigned char * d_hash_in;
    char * d_passwords;

	// device memory allocations
    gpuErrchk(cudaMalloc((void**) &d_pass_out, MAX_PASSWORD_LEN));
    gpuErrchk(cudaMalloc((void**) &d_hash_in, 16));
    gpuErrchk(cudaMalloc((void**) &d_passwords, PASSWORDS_PER_KERNEL * MAX_PASSWORD_LEN));

	//TODO load all passwords with padding, passwords size = PASSWORDS_PER_KERNEL*MAX_PASSWORD_LEN
    const char * dictionary[] = {"kattj'vel", "passwor", "password", "passwords"};
	char passwords[PASSWORDS_PER_KERNEL * MAX_PASSWORD_LEN] = {0};
	for(int p = 0 ; p < PASSWORDS_PER_KERNEL ; ++p) {
		strcpy(passwords+p*MAX_PASSWORD_LEN, dictionary[p]);	
	}


	// device variable initializing
    gpuErrchk(cudaMemcpy(d_hash_in, hash_in, 16, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_passwords, passwords, PASSWORDS_PER_KERNEL*MAX_PASSWORD_LEN , cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(d_pass_out, 0, MAX_PASSWORD_LEN));

	// run crack
	dim3 grid_dim(1,0,0);
	dim3 block_dim(1,0,0); // TODO fix number of threads, make grid dynamic
    crackMD5<<<3,3>>>(d_hash_in, d_passwords, PASSWORDS_PER_KERNEL, d_pass_out);
   
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		printf("ERROR: %s\n", err);
    }


	// retrieve result
    unsigned char result[MAX_PASSWORD_LEN] = {0};
    cudaMemcpy(result, d_pass_out, MAX_PASSWORD_LEN, cudaMemcpyDeviceToHost);
  

  	// free device memory
	gpuErrchk(cudaFree(d_pass_out));
	gpuErrchk(cudaFree(d_hash_in));
	gpuErrchk(cudaFree(d_passwords));

	// TODO test if there's a result


	// print result
    std::cout << hexdigest(hash_in) << std::endl;
    std::cout << "Password is: " << result << std::endl; 
    return 0;
}
