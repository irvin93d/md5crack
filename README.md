# md5crack
GPU accelerated cracker for md5

## Compile the code
This project was built using C++11, and CUDA v8.0.
To build the project, you must have make installed. To compile the program, just run `make`.

## To download the dictionary files used, and compile the program
```
make setup
make
```
Will produce the program in a.out.


## Utility program
We included a utility program that creates an MD5 hash from its input arguments.
```
make calcMD5
./md5 hash me
```
will give you the hash of 'hash me'.


## Thanks to
[Original MD5 implementation](http://www.zedwood.com/article/cpp-md5-function)

