#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "md5.hpp"
#include <fstream>
#include <errno.h>

#define PASSWORDS_PER_KERNEL 8192
#define MAX_PASSWORD_LEN 256
#define DIGEST_SIZE 16
#define BLOCK_DIM 256
#define ASCII_MIN 32
#define ASCII_MAX 255

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

enum STATE {DICTIONARY, BRUTEFORCE, DONE};

__device__ int password_found = 0;

static size_t password_count = 0;

__device__ int digest_equal(unsigned char const * a, unsigned char const * b){
	for(int i = 0 ; i < DIGEST_SIZE ; ++i ) {
		if(a[i] != b[i]) {	
			return 0;
		}
	}
	return 1;
}

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
    
		// If crack is successful, return result
		if(digest_equal(hash_in_cache, result)) {
			// If 2 different passwords gives same (and correct) hash,
			// only one password is returned
			int old = atomicAdd(&password_found,1);
			if(!old){
				memcpy(pass_out, pass_cache, DIGEST_SIZE);
    		}
		}
	}
}

int next_pass(char * pass) {
    if(pass[0] == 0) return 0; // last character, done
    if(pass[0] == (char) ASCII_MAX) {
        pass[0] = (char) ASCII_MIN;
        return next_pass(pass + sizeof(char));
    }
    pass[0]++;
    return 1;

}

int cpu_brute_force(unsigned char const * hash_in, unsigned char * res) {
    char gen_pass[MAX_PASSWORD_LEN];
    for(int i = 1; i < 6; ++i) {
        memset(gen_pass, ASCII_MIN, i);
        gen_pass[i] = 0;
        do {
            MD5 md5(gen_pass, i);
            password_count++;
            unsigned char result[DIGEST_SIZE];
            md5.get_digest(result);
            
            // Compare result with input hash
            bool eq = true;
            for(int j = 0; j < DIGEST_SIZE; j++) {
                if(hash_in[j] != result[j]) {
                    eq = false;
                    break;
                }
            }
            if(eq) {
                memcpy(res, gen_pass, i + 1);
                return 1;
            }
        } while(next_pass(gen_pass));
    }
    return -1;
}

// Must run with griddim == blockdim == ASCII_MAX-ASCII_MIN and len >= 3
__global__ void gpu_brute_crack_md5(unsigned char const * hash, int len, char * pass_out){
	unsigned char hash_cache[DIGEST_SIZE];
	memcpy(hash_cache, hash, DIGEST_SIZE);

	unsigned char pass[MAX_PASSWORD_LEN] = {0};
	for(int i = 0 ; i < len-2 ; ++i) {
		pass[i] = ASCII_MIN;
	}

	// set last character to threadid and let this be constant to avoid divergence
	pass[len-1] = threadIdx.x + ASCII_MIN;
	pass[len-2] = blockIdx.x + ASCII_MIN; 

	while(pass[len-3] != ASCII_MAX) {
		pass[0]++;
		if(pass[0] == ASCII_MAX) {
			pass[0] = ASCII_MIN;
			for(int i = 1 ; i < len-2 && pass[i]++ == ASCII_MAX ; ++i) {
				pass[i] = ASCII_MIN;
			};
		}
	   	// Create hash for password to test
		MD5 md5((char*) pass, len);
              
		// Retrieve created hash
        unsigned char result[DIGEST_SIZE]; // 128 bit
        md5.get_digest(result); // load the result
    
		// If crack is successful, return result
		if(digest_equal(hash_cache, result)) {
			// If 2 different passwords gives same (and correct) hash,
			// only one password is returned
			printf("KERNEL SAYS YEY\n");
			int old = atomicAdd(&password_found,1);
			if(!old){
				memcpy(pass_out, pass, DIGEST_SIZE);
    		}
		}
	}
}

int gpu_brute_force(unsigned char const * hash, int len, unsigned char * res) {

    unsigned char * d_hash;
    char * d_pass_out;
    gpuErrchk(cudaMalloc((void**) &d_hash, 16));
    gpuErrchk(cudaMalloc((void**) &d_pass_out, MAX_PASSWORD_LEN));
	
	gpuErrchk(cudaMemcpy(d_hash, hash, 16, cudaMemcpyHostToDevice));
	
	gpu_brute_crack_md5<<<ASCII_MAX-ASCII_MIN, ASCII_MAX-ASCII_MIN>>>(d_hash, len, d_pass_out);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		printf("ERROR: %s\n", err);
	}
	
	cudaMemcpy(res, d_pass_out, MAX_PASSWORD_LEN, cudaMemcpyDeviceToHost);
	
	cudaFree(d_hash);
	cudaFree(d_pass_out);

	if(res[0]) {
		return 1;
	}
	return 0;
}

int cpu_crack(unsigned char const * hash_in, std::ifstream* file, unsigned char * res) {
    std::string s;
    // Read each line
    while(std::getline(*file, s)){
        MD5 md5(s.c_str(), s.length());
        password_count++;
        unsigned char result[DIGEST_SIZE];
        md5.get_digest(result);
        
        // Compare result with input hash
        bool eq = true;
        for(int i = 0; i < DIGEST_SIZE; i++) {
            if(hash_in[i] != result[i]) {
                eq = false;
                break;
            }
        }
        if(eq) {
            memcpy(res, s.c_str(), s.length() + 1);
            return 1;
        }
    }
    return -1;
}

int gpu_crack(unsigned char const * hash_in, std::ifstream * file, unsigned char * result) {
	// device declerations
    char * d_pass_out;
    unsigned char * d_hash_in;
    char * d_passwords;
    int password_found = 0;
	// device memory allocations
    gpuErrchk(cudaMalloc((void**) &d_pass_out, MAX_PASSWORD_LEN));
    gpuErrchk(cudaMalloc((void**) &d_hash_in, 16));
    gpuErrchk(cudaMalloc((void**) &d_passwords, PASSWORDS_PER_KERNEL * MAX_PASSWORD_LEN));
	while(!password_found) {

		//load a chunk of passwords. Passwords are null terminated
		char passwords[PASSWORDS_PER_KERNEL * MAX_PASSWORD_LEN] = {0};
		std::string str;
        size_t current_password_count = 0;
        for(int p = 0 ; p < PASSWORDS_PER_KERNEL ; ++p) {
            if(!std::getline(*file, str)) {
                password_found = -1;
                break;
            }
           

            ++current_password_count;
			strcpy(passwords+p*MAX_PASSWORD_LEN, str.c_str()); // load file row to padded password list
			passwords[p*MAX_PASSWORD_LEN + str.length()] = '\0'; // exchange last character '\n' to '\0'
			//	std::cout << passwords + p*MAX_PASSWORD_LEN << std::endl;
		}

		// device variable initializing
		gpuErrchk(cudaMemcpy(d_hash_in, hash_in, 16, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_passwords, passwords, PASSWORDS_PER_KERNEL*MAX_PASSWORD_LEN , cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemset(d_pass_out, 0, MAX_PASSWORD_LEN));
        password_count += current_password_count;
		// run crack
		crackMD5<<<(PASSWORDS_PER_KERNEL+BLOCK_DIM-1)/BLOCK_DIM,BLOCK_DIM>>>(d_hash_in, d_passwords, current_password_count, d_pass_out);
		//crackMD5<<<1024,256>>>(d_hash_in, d_passwords, current_password_count, d_pass_out);
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
    return password_found;
}



int main(int argc, char const ** argv) {
	if(argc < 2) {
		std::cout << "missing input" << std::endl;
		exit(-1);
    }

    char const * hash  = argv[1]; // 'password'

	std::ifstream file("crackstation-human-only.txt");
	if(!file) {
		std::cerr << "Error: " << strerror(errno) << std::endl;
		return(-1);
	}

    std::cout << "Hash: " << hash << std::endl;
    
	unsigned char result[MAX_PASSWORD_LEN] = {0};
	// Convert the hex representation of the hash into its numerical value
    unsigned char hash_in[17];
    hexencode(hash, hash_in);
    
	int password_found = 0; 
    bool use_gpu = true;
    bool use_brute = true;
    if(argc > 2) {
        for(int i = 2; i < argc; i++) {
            std::string arg(argv[i]);
            if(arg == "-cpu") {
                use_gpu = false;
            } else if (arg == "-nb") {
                use_brute = false;
            }
        }
    }

    if(use_gpu) {
        std::cout << "Running crack on GPU" << std::endl;
        password_found = gpu_crack(hash_in, &file, result);
    } else {
        std::cout << "Running crack on CPU" << std::endl;
        password_found = cpu_crack(hash_in, &file, result);
    }

	// print result
	if(password_found)
    	std::cout << "Password is: " << result << std::endl; 
   	else
    	std::cout << "Password not found in dictionary" << std::endl; 
    std::cout << "Tried " << password_count << " passwords" << std::endl;

    if(!password_found && use_brute) {
        std::cout << "Starting brute force" << std::endl;
        if(use_gpu){
			password_found = gpu_brute_force(hash_in, 4, result);
        } else {
            password_found = cpu_brute_force(hash_in, result);
			std::cout << "Tried " << password_count << " passwords" << std::endl;
        }

        if(password_found == 1)
            std::cout << "Password is: " << result << std::endl;

    }

	return 0;
}

