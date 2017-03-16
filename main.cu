#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "md5.hpp"
#include <fstream>
#include <errno.h>

#define PASSWORDS_PER_KERNEL 20480
#define MAX_PASSWORD_LEN 256
#define DIGEST_SIZE 16
#define BLOCK_DIM 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ bool password_found = false;

__global__ void crackMD5(unsigned char* hash_in, char* pass_set, uint32_t len, char* pass_out) {
	unsigned char hash_in_cache[DIGEST_SIZE];
	memcpy(hash_in_cache, hash_in, DIGEST_SIZE);

    for(int id = threadIdx.x + blockIdx.x*blockDim.x ; id < len && !password_found ; id += gridDim.x*blockDim.x) {
		// Init varibles for password test
        char * pass_test = pass_set + MAX_PASSWORD_LEN * id;
        char pass_cache[MAX_PASSWORD_LEN];
        int pass_len = 0;

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

		if(password_found)
			break;
		// If crack is successful, return result
		if(success) {
			password_found = true;
			memcpy(pass_out, pass_cache, DIGEST_SIZE);
    	}
	}
}

int main(int argc, char const ** argv) {

	

    char const * hash  = argv[1]; // 'password'

	std::ifstream file("crackstation-human-only.txt");
	if(!file) {
		std::cerr << "Error: " << strerror(errno) << std::endl;
		return(-1);
	}

    std::cout << "Hash: " << hash << std::endl;
    
	unsigned char result[MAX_PASSWORD_LEN] = {0};
	// Convert the hex representation of the hash
    unsigned char hash_in[17];
	strcpy( (char*) hash_in, hexencode(hash).c_str());

	// device declerations
    char * d_pass_out;
    unsigned char * d_hash_in;
    char * d_passwords;

	// device memory allocations
    gpuErrchk(cudaMalloc((void**) &d_pass_out, MAX_PASSWORD_LEN));
    gpuErrchk(cudaMalloc((void**) &d_hash_in, 16));
    gpuErrchk(cudaMalloc((void**) &d_passwords, PASSWORDS_PER_KERNEL * MAX_PASSWORD_LEN));

	int password_found = 0; 
	while(!password_found) {

		//load a chunk of passwords. Passwords are null terminated
		char passwords[PASSWORDS_PER_KERNEL * MAX_PASSWORD_LEN] = {0};
		std::string str;
        for(int p = 0 ; p < PASSWORDS_PER_KERNEL ; ++p) {
            if(!std::getline(file, str)) {
                password_found = -1;
                break;
            }

			strcpy(passwords+p*MAX_PASSWORD_LEN, str.c_str()); // load file row to padded password list
			passwords[p*MAX_PASSWORD_LEN + str.length()-1] = 0; // exchange last character '\n' to '\0'
		}

		// device variable initializing
		gpuErrchk(cudaMemcpy(d_hash_in, hash_in, 16, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_passwords, passwords, PASSWORDS_PER_KERNEL*MAX_PASSWORD_LEN , cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemset(d_pass_out, 0, MAX_PASSWORD_LEN));

		// run crack
		crackMD5<<<(PASSWORDS_PER_KERNEL+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM>>>(d_hash_in, d_passwords, PASSWORDS_PER_KERNEL, d_pass_out);
		cudaError_t err = cudaGetLastError();
		if(err != cudaSuccess) {
			printf("ERROR: %s\n", err);
		}

		// retrieve result
		cudaMemcpy(result, d_pass_out, MAX_PASSWORD_LEN, cudaMemcpyDeviceToHost);

		if(result[0]){
			password_found = 1;		
		}
	}

	// free device memory
	gpuErrchk(cudaFree(d_pass_out));
	gpuErrchk(cudaFree(d_hash_in));
	gpuErrchk(cudaFree(d_passwords));

	// print result
	if(password_found == 1)
    	std::cout << "Password is: " << result << std::endl; 
   	else
    	std::cout << "Password not found" << std::endl; 

	return 0;
}
