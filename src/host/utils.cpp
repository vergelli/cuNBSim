#include <cstdio>
#include <iostream>

void printProgress(int current, int total) {
    float progress = (float)current / total;
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100) << " %\r";
    std::cout.flush();
}
