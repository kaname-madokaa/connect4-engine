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

void pause_loop()
{
    while (1)
    {
        std::string line;
        std::cin >> line;
        if (line == "pause")
        {
            is_paused = !is_paused;
            if (is_paused) std::cout << "paused\n";
            else std::cout << "resumed\n";
        }
    }
}



int main()
{
    std::thread a(pause_loop);
    std::cout << "engine started\n";
    search::init();
    uci::selfplay(2400 * 25,4);
    //uci::rate();
    uci::Loop();
}
