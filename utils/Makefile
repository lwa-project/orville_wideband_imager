all: convert_to_fits

oims.o: oims.cpp oims.hpp
	g++ -c -o oims.o oims.cpp

convert_to_fits.o: convert_to_fits.cpp
	g++ -c -o convert_to_fits.o -I/opt/homebrew/include/ convert_to_fits.cpp

convert_to_fits: convert_to_fits.o oims.o
	g++ -o convert_to_fits convert_to_fits.o oims.o -L/opt/homebrew/lib/ -lcfitsio
