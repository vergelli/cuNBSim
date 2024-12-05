#include <cstdio>

void printProgress(int current, int total) {
    float progress = (float)current / total;
    int barWidth = 70;

    printf("[");
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d%%\r", int(progress * 100));
    fflush(stdout);
}
