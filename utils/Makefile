all: test

oims.o: oims.cpp oims.hpp
	g++ -c -o oims.o oims.cpp

test.o: test.cpp
	g++ -c -o test.o -I/opt/homebrew/include/ test.cpp

test: test.o oims.o
	g++ -o test test.o oims.o -L/opt/homebrew/lib/ -lcfitsio
