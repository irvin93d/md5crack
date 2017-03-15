#include <string>
#include "md5.hpp"


__global__ void bruteforceMD5(char* input, char** dictionary) {
    // TODO
}

int main(int argc, char const ** argv) {
    // TODO load a file and stuff
    std::string hash = "5f4dcc3b5aa765d61d8327deb882cf99"; // 'password'
    std::cout << hash << std::endl;
    return 0;
}
