#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <fstream>
#include <thread>
#include "Board.h"
#include "uci.h"
#include "NN.h"
#include "ConvLayer.h"

class a
{
    std::array<int, 1> abcd;
};

template <typename _Ty>
void incrementArray(_Ty& arr)
{
    for (auto& val : arr)
    {
        val++;
    }
}
typedef std::array<int, 20> array_type;



int main()
{
    std::cout << "engine started\n";
    search::init();
    uci::selfplay(100000,4);
    uci::rate();
    uci::Loop();
}
