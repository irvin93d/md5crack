CC = nvcc
FLAGS = -x cu

main: md5.o main.o
	$(CC) main.o md5.o

main.o: main.cu
	$(CC) -dc main.cu

calcMD5: md5.o calcMD5.o
	$(CC) calcMD5.o md5.o -o md5 

calcMD5.o: calcMD5.cpp
	$(CC) $(FLAGS) -dc calcMD5.cpp 

md5.o: md5.cu md5.hpp
	$(CC) $(FLAGS) -dc md5.cu

setup:
	wget https://crackstation.net/files/crackstation-human-only.txt.gz
	wget https://wiki.skullsecurity.org/images/b/b5/List-cain.txt
	wget https://wiki.skullsecurity.org/images/5/53/List-john.txt
	gunzip crackstation-human-only.txt.gz
