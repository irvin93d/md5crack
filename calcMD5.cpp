#include <string>
#include "md5.hpp"

using namespace std;

int main(int argc, char ** argv) {
    if(argc > 1) {
        string in = "";
        for(int i = 1; i < argc; i++) {
            in += argv[i];
            if(i + 1 != argc) {
                in += " ";
            }
        } 
        cout << "MD5 of '" << in  << "'" << endl;

        cout << md5(in)  << endl;
    } else {
        cout << "Missing argument" << endl;
    }
}
