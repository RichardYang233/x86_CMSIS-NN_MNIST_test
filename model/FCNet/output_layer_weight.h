#ifndef OUTPUT_LAYER_WEIGHT_H
#define OUTPUT_LAYER_WEIGHT_H

#include <stdint.h>

#define OUTPUT_LAYER_WEIGHT   5120

// 10 * 512
int8_t output_layer_weight[OUTPUT_LAYER_WEIGHT] = {
-31, -52, 7, -7, -16, -3, 18, 1, -65, -13, -7, 23, 13, -15, -22, 5, 14, -28, 7, 18, 30, -1, 7, -55, -28, 12, -5, -1, 1, -51, 23, 8, 14, -25, -18, -9, -20, 3, -8, 4, -46, 12, 19, 8, 9, 15, 15, 20, 16, -40, 
10, -17, -19, 12, -20, -10, -77, -4, -15, 6, 2, 26, 12, 11, 23, -12, 19, -18, -19, 22, -21, -54, 5, 7, -53, -7, -11, -20, 12, 5, -5, 4, 28, -14, 22, 21, -6, 11, 6, -47, -28, 28, -19, -1, 5, 10, -48, 5, 4, 21, 
-12, -12, -29, -46, 20, -20, 30, -32, -8, 19, -22, 5, -29, 0, 5, 21, 16, 20, 26, -100, 21, -82, -81, 19, -5, -46, 23, -48, 4, -3, -13, -13, 13, 5, 2, 20, 16, 11, 9, -25, 6, 9, -14, 22, -5, -13, -19, -42, -3, -1, 
-10, -36, 7, -42, -7, 14, -32, -18, -15, -13, 14, 10, -28, -32, 23, -6, 22, 33, -26, -17, -23, -8, 18, 24, -21, -62, 25, -72, -9, -20, -48, -35, 21, -28, -5, 19, -30, -42, -24, -16, 18, -9, 2, -15, 7, 8, -18, -14, -16, 9, 
-33, 17, -29, -14, 21, 11, 3, -60, -40, -46, -20, 21, -22, -25, -29, 25, 22, 19, -48, -3, -8, -48, -40, 9, 7, -7, 25, -1, 25, 14, 18, 28, 24, 14, -23, 6, 14, -11, -7, 8, -5, -5, 22, -25, -4, 14, 23, -17, -20, -25, 
4, -26, 11, -17, -2, -36, 12, -9, -2, 0, 16, 22, -9, 20, 11, -15, 2, -49, 18, 2, 4, -2, -34, -1, 30, -34, -29, 17, 21, -45, -28, 1, -57, -11, -72, -52, 18, 16, 18, -23, 8, -27, -20, 24, 16, -35, -9, 0, 3, 5, 
15, 11, -18, -38, -12, -42, -2, 17, 12, -2, -42, -12, 2, 14, 6, -27, 4, -18, 12, 8, 15, -33, -19, 6, 8, 21, -56, -26, -12, -22, 24, 13, -4, -34, -81, -17, 7, 26, -16, 11, 17, 21, 8, -40, 5, 13, 15, -2, -41, -24, 
5, 3, -42, 23, -2, 10, -4, 7, 14, -75, 3, 13, -15, 5, 26, 13, -11, -19, -15, -49, -38, 11, 0, 19, 5, -20, -64, 6, 26, -10, 21, 16, -39, 6, -7, -7, 8, -58, 25, 6, -5, 15, 15, 15, -20, 28, -24, -27, 8, 26, 
9, -37, 22, 4, -10, 35, 18, -46, 19, -37, 11, 27, 15, -15, 17, 19, 15, 14, 12, -40, -42, -25, 12, -6, -26, -35, 6, -8, -31, -1, 4, 10, -4, -22, 32, 20, 18, -6, -17, -1, 12, -23, 19, 25, -11, 14, -12, 22, 6, -9, 
0, -29, -59, 7, -37, 20, 25, -10, -45, -24, 10, 23, -45, -30, 8, -21, 20, -50, -86, -8, 15, 16, -10, -100, -32, -1, 12, -24, 20, -31, 4, -48, 5, -30, 16, -54, 0, 8, -5, -23, -35, 18, 14, -7, -100, 13, 2, 18, 22, 5, 
-14, -9, 24, -2, -20, -35, -30, -50, -49, 2, -34, 27, 23, -46, 8, -8, 13, 3, 24, 8, 11, 20, 21, 45, 29, -11, -39, 17, -36, 13, 20, -19, -29, -21, 16, 21, 10, -9, 14, 14, -27, 2, -23, -20, 13, -37, -3, -12, 20, -40, 
1, 5, 6, -6, 69, -19, 16, 27, -29, -46, -13, 24, -13, -45, 15, -17, -10, -35, 1, -47, 21, -14, -34, -31, 44, -28, 8, -17, 3, -34, -27, -6, 6, 1, -24, -44, 1, -29, -30, 35, 10, -32, 20, -6, -5, -26, 20, -11, 23, -19, 
16, 24, -3, 23, -34, -5, 20, -36, 14, 21, 51, 8, -18, -13, -10, 4, 2, 36, -8, 27, 12, -13, 20, -12, 21, -30, -49, 14, -15, -24, 40, -37, 2, 17, 17, 4, 26, 2, -3, 15, 13, 14, -7, -42, -5, 28, -14, -9, -33, 20, 
-47, 13, -47, -34, 21, -30, -18, -7, 20, 9, -8, 4, -21, -2, -1, 12, 2, 16, -14, -25, -27, 18, 39, 21, -12, 11, -27, 5, -6, 21, -38, -48, 35, 23, -12, -12, 12, 22, -29, -4, 19, 24, 22, 2, 3, -9, 22, 21, 21, -5, 
-57, 17, 23, -47, -34, -22, -28, -28, 19, 3, -35, 16, -8, -47, 8, -34, -20, -11, 9, 23, 22, 0, -24, -15, -14, -29, 16, -35, -2, 26, 20, 15, -24, 21, 8, -39, 8, 38, -9, -19, 7, 9, -26, -27, -27, 13, 12, -11, -22, -37, 
-17, -48, -26, -5, -32, 18, -1, -39, 35, 12, -10, 14, -48, -32, -9, 20, 13, 13, -1, -25, 22, -26, -40, -18, 2, -12, -37, 7, 23, 4, 5, -41, 5, -24, 9, -37, 8, -4, 8, -28, -19, -46, 4, -52, 11, -35, 30, 15, -1, 0, 
17, -7, 19, 18, 37, -23, 18, 7, 30, 17, 22, -8, 9, 21, 2, 18, 5, 35, -28, -14, -42, -31, 56, 10, -33, 21, -45, -1, 5, 21, 16, -16, -14, -30, 15, 33, -4, -35, 10, 19, 6, -5, 9, -21, 11, 63, -32, -6, -5, 15, 
11, 3, -34, -35, -25, 5, -38, 21, 8, 6, 51, 15, 15, 84, 21, 38, 22, -27, 0, -37, -1, 18, -22, 8, -27, -11, -28, -26, -25, -32, 3, 27, 3, -24, 7, -26, 17, -41, 51, -28, -19, -28, -45, -41, 9, -43, -3, 22, -35, -39, 
-29, 17, 3, 15, 77, -21, -25, -1, 11, 22, 9, -29, 20, -28, -22, 38, -26, -2, 7, -34, -36, 16, 9, -39, 6, -33, -10, -18, 14, -5, -40, 6, -1, 21, -12, -32, 26, 6, -27, 22, 13, 20, -47, 9, -18, 22, 35, -39, -23, 30, 
9, 20, 16, -56, -24, 20, -30, 7, 32, 12, -31, -5, 0, -1, 16, -46, -12, 41, -29, -49, 21, -31, -23, -20, 15, 11, 26, 21, -7, 2, 23, -43, 44, 12, -43, 3, -13, -20, 10, 20, -29, 13, -18, 7, 4, -3, -9, -1, -23, -51, 
32, 11, 18, -27, -24, 62, -2, -37, -7, 0, 21, -3, -21, 6, 29, -18, 13, 5, 17, -30, 6, -21, 4, -44, -19, -11, 13, 34, 11, -7, -21, -25, 19, -90, 8, -4, 21, 8, -5, 10, 16, 21, 26, 2, 15, 9, -72, 8, 8, 15, 
23, 28, -3, 18, 35, 6, -46, -22, -14, 16, -52, 13, 19, 17, 13, -25, -18, -13, -28, 26, 10, -30, -29, 7, -12, 16, 0, 13, 10, 21, -26, -16, 3, 28, 4, -5, 31, 7, 29, 9, -29, 3, 21, 16, 28, -2, -26, 11, -20, 15, 
6, 35, 30, -31, 24, 21, 8, 19, 5, 12, 3, 2, 20, 24, 4, 32, -19, 12, -3, -27, -1, 4, -36, -45, 7, 4, -21, -37, 2, -29, 26, 20, -30, 25, 39, 10, 6, -23, -1, 13, -1, 2, -33, -65, 1, -21, 7, -13, 32, 11, 
-29, 29, -3, 30, -9, -15, -1, -61, -8, 9, 10, 10, 6, -23, -56, -7, -29, 11, -14, 7, -16, -28, 34, 17, -4, 36, 3, -78, -19, -14, 21, -57, 11, 29, -3, -3, -9, -21, -18, 17, 26, 18, 28, -5, 32, -11, -11, -36, -11, -25, 
13, -15, 20, 34, -50, 44, -15, -8, 33, -59, -12, -7, 4, -26, 1, 30, -17, 42, 19, -4, 8, 18, 68, 27, -31, -11, 40, 16, 14, 16, -9, 24, -2, -15, -32, -5, -19, 23, 13, 11, 19, 12, -27, 5, -29, 5, 14, 38, 9, 31, 
-53, 12, 2, 18, -15, 1, -20, 10, -8, 3, 21, -5, -86, -9, 21, 18, 8, 13, 24, 3, -17, -17, -9, -4, 5, 37, 17, -15, -24, 13, 28, 17, 15, -19, 17, -6, -37, -95, -81, -78, -15, -20, 4, -44, 33, 14, 1, -21, -68, -14, 
12, 20, 11, -10, 8, 21, -4, -30, 48, 26, -25, -4, 8, 25, 9, 19, -74, 19, -32, 11, -19, 23, -54, 10, 22, -37, 23, 18, 16, 0, 19, -18, -13, -13, -16, 13, -14, 3, -16, 24, 1, -22, 12, 12, 12, 10, -9, -63, 13, 3, 
27, 11, 7, 18, 16, 21, -95, -21, -25, 14, -20, 1, 24, 7, 15, -2, -18, 23, 2, 25, -22, 24, -81, 21, -23, -39, 5, 4, 42, 41, 11, -2, 35, 11, 21, 22, -17, -24, -6, 30, -6, 38, -12, 35, 19, -1, 43, 12, -6, 21, 
-8, 18, 26, -28, -8, -25, -5, -12, -21, 33, 1, 10, 16, 24, 0, 11, -32, -32, -13, -4, -2, 33, 20, -12, 14, -32, -19, 12, -5, -31, 32, -50, 4, 6, 22, 28, 53, 14, 10, 7, 23, 10, 0, 12, -12, -16, 24, -77, -8, 58, 
-17, 16, -14, 5, 4, -81, 8, 6, -60, 8, 9, -55, 14, -36, 16, -24, -3, 29, 11, -14, 24, -3, -13, 10, 16, -45, -15, -24, -1, -31, 22, -69, -25, 33, 10, 15, -15, -21, 36, 22, -40, 26, 10, 8, -46, 15, -28, -40, 18, -21, 
19, 22, 1, -2, 21, -27, -1, 8, -45, 19, -13, -7, 25, -23, -39, 0, 3, -80, -86, 16, -63, -7, 1, 21, 19, -7, -25, 11, 3, 8, -15, -28, -12, -6, -40, 13, 2, 35, 15, -16, 15, -47, 52, 16, -3, -49, 18, -32, 8, -37, 
21, 11, 13, 8, 16, -25, 1, 31, -36, -16, -2, 0, 2, -10, 23, 19, 4, -7, 8, 13, 8, -12, 25, -19, 14, -18, 20, 20, -10, -41, -6, 9, 7, -24, 16, 15, -36, -20, -73, -25, 30, -18, 13, -6, 24, -43, -19, -24, -23, 16, 
26, 0, 8, 21, -45, 17, 25, 18, 14, 8, 6, -4, 3, -82, -16, 1, 19, -20, -11, 22, 2, -33, 7, 16, 13, 36, -13, -12, 27, -25, 3, 17, 17, 5, -42, 6, -38, 10, 18, -27, 19, -30, -1, 16, -26, 25, -17, -9, 6, 18, 
-6, -49, 14, -13, 26, 29, 29, 39, 32, -10, -15, 27, -17, 17, 2, -49, 19, -4, 18, -70, -34, -14, -27, 12, 8, 35, 14, 5, 21, -16, -19, 22, -18, 14, -48, 15, 27, -28, 19, -65, -12, -3, 22, 32, -18, 20, -28, 8, -59, -8, 
-22, 7, -56, -16, -5, -45, 43, 12, 45, 9, -12, 24, -11, 5, -21, -36, -73, -26, 3, -57, 14, -48, -7, 5, 11, -26, 23, -11, 6, 3, 21, -24, 0, 3, -23, -23, -8, -10, 48, 23, -25, -14, -22, 31, 37, 2, 20, -15, 9, -47, 
-9, 10, 11, -7, 25, -48, -9, 17, -10, -29, 20, 23, 27, 6, 23, 22, 15, -7, -17, 6, -5, -22, -21, -19, 10, -14, 10, 14, 2, 0, 17, -20, -50, 4, -19, 15, -30, -32, 1, 13, -12, 20, -26, 19, 5, -9, -25, 9, -33, 29, 
-28, 27, -59, 17, -3, -51, -19, 3, 5, 20, -13, -1, 11, 13, -9, 14, 8, -21, 26, -43, -34, -7, 5, -24, -10, 20, 13, -11, 27, 2, -12, 7, -35, 24, 27, 10, 4, -55, -21, 7, -16, 29, 9, 4, 6, -1, -55, 13, 6, 17, 
-14, -2, -29, 32, -33, 17, -19, 19, 8, -23, -22, 7, -7, -1, 20, 13, -41, 12, -27, -10, 23, -18, -4, -7, 4, -54, -34, -22, -18, 11, 16, 15, -6, 8, -24, 25, 16, -24, -41, 24, -57, 3, 11, -4, -11, -17, -18, -19, 14, -3, 
20, -2, -5, 7, -26, -9, -28, 12, -39, 7, 18, -6, 23, -3, 2, 2, -33, -2, 5, 4, -45, -50, -9, -39, 16, 22, 3, -50, -26, -44, -7, 19, 26, -10, -3, -14, 18, 17, 10, -28, -10, -13, -9, 33, 8, 21, 26, 23, -18, -15, 
17, -25, 7, -31, 5, 18, 15, -5, 28, 13, 22, -18, 18, 22, 15, -10, 4, -23, -11, -26, 2, -23, -26, -63, -16, 34, 25, -53, -3, 6, -45, 10, 37, -56, -19, 17, 2, -51, 21, 1, -19, -37, -18, 32, 23, -22, 13, 12, -2, 11, 
-47, -28, -18, -32, 6, -37, 15, 13, -22, 52, -3, -14, 8, 7, 9, 10, -29, 8, 22, -32, -26, 4, -11, -9, -14, -62, -1, -3, -23, -65, 40, -15, 13, 30, 13, 7, 23, -31, -22, 19, 0, 4, -14, 15, -3, 27, -12, 11, 22, 13, 
5, -6, 20, 16, 8, -34, 4, 6, -25, -22, 42, -1, -6, -33, -31, -56, -17, 9, 24, -56, 21, -37, 4, -20, -29, 8, -42, -43, -74, 14, -34, 15, -14, 17, -23, 8, -12, -18, -7, -8, 3, 21, -26, -6, 8, 15, 5, -5, 19, 20, 
17, -30, 8, -13, 6, 2, -20, 11, 11, -15, 30, 18, -56, 7, 12, 11, 27, -43, -49, 18, 6, 18, 30, -16, 14, -1, -46, 5, 31, 22, -16, -22, -47, -14, 15, -3, -6, -69, -16, -1, 3, 12, -58, -17, 13, -34, 36, -72, 14, 12, 
7, 19, -42, -30, -76, 16, 18, 17, 3, -3, -28, -31, -5, 7, -24, -15, -3, -10, -12, 25, -27, -57, 60, -30, -63, 34, -11, 13, 12, -1, -41, 11, -33, -29, -7, 14, -11, 50, 4, 16, 33, 3, -7, -38, -20, 4, 18, 6, 22, -25, 
-33, 32, -2, -24, 18, -44, -3, -49, -43, -33, 24, -10, 5, 1, 34, 24, 22, -8, -32, -20, -1, -15, 9, 36, -5, 26, -25, -24, -2, 11, -28, 32, 8, -19, -26, -10, 4, 24, -65, 1, 23, 14, -51, 13, 15, 12, -8, -2, 23, -2, 
-10, -1, 2, -5, 5, -32, -5, 1, 16, -15, -3, -4, -98, -64, -18, -54, 6, -1, -11, 5, -38, -31, -50, -14, 32, 23, -78, -46, -61, -23, 16, -29, 10, 24, 9, 8, -83, -10, -35, -58, -27, -13, 15, -25, -26, 8, 22, 4, 3, 11, 
-23, -28, -22, -8, -20, -10, -30, 2, 11, 0, 23, 4, -11, 21, 14, 6, -57, 40, 3, 27, 6, 19, 26, -10, -7, 0, -4, 15, 5, -3, -18, 5, -36, -7, -18, 18, -58, -7, 7, -7, -10, -47, -3, 1, 4, 6, 0, -47, -72, -36, 
15, 16, 27, 25, -1, 9, 7, 21, 26, 12, 17, -58, 18, 3, 1, 43, 13, -42, -24, -10, -36, -53, -48, -1, -41, -27, -15, 3, -17, 1, 5, 24, 15, 20, -35, 21, -79, -19, 27, 1, 23, -14, -5, -39, -1, -15, -3, -127, 10, 68, 
43, -15, 25, 22, 23, -16, 18, -18, -26, 1, -15, 8, 1, -3, -2, -91, 17, -27, 17, -73, -52, -84, -23, 12, 49, 18, -105, 13, -12, 18, 11, 9, 15, 17, -7, -42, -58, -48, 1, 22, 10, 16, 4, 5, 3, 16, -31, -17, 4, 25, 
9, -24, 23, -45, -99, 15, -21, -24, 14, -31, -13, 0, -101, 9, -85, 21, 14, -43, 18, -32, 11, 30, 40, -9, 10, -49, -4, -24, 16, 9, 17, 18, -21, 21, 23, -47, -32, 30, -62, 14, 24, -15, 17, -30, 30, 47, 7, 8, -17, 32, 
3, -1, -10, 24, 5, -8, 29, -25, -2, 4, -2, -29, -15, 0, -66, -38, 18, 22, 20, -9, 8, 22, 14, -4, -3, -51, 18, 5, 31, 27, -8, -59, -3, -82, 22, -17, -20, 21, 16, 19, 0, 25, 18, 10, 14, -62, -57, -22, -13, 17, 
23, -43, -6, 0, -11, 26, -12, -33, -3, -55, -13, 4, 13, -30, -19, 10, -4, 11, 32, 19, -4, -24, -75, -1, -2, -24, 12, 20, -51, -16, -33, 10, 51, 13, -10, 16, -49, -90, 20, 2, -41, 16, 6, -3, -3, -40, 18, -11, -14, -17, 
35, -7, 25, 8, 10, -74, -20, 5, 14, -19, -7, 28, 8, 60, -69, -2, 30, 22, -13, -39, -19, 12, -37, 5, -30, 7, 16, 1, 31, -26, 39, 4, -21, 8, 18, 3, -6, -32, -85, 10, -84, -8, -20, -4, -27, 20, -71, -2, 10, -5, 
-3, -24, -34, -5, 22, 19, -31, 1, -8, 33, 0, 1, 9, 13, 12, 28, 20, 10, 12, -11, 24, -16, 7, 8, 9, 6, 4, 19, -65, 40, -4, 41, 32, 28, 39, 4, 31, 27, 23, 26, 7, -6, 15, 10, 2, 2, 11, 1, 13, -12, 
16, -15, -50, -9, -16, -2, 17, 12, -31, 15, -32, 11, 22, 39, -32, 9, -15, 3, 56, -14, 26, 0, 16, -15, -12, 4, -37, -67, 19, 3, 12, -18, -17, 18, -21, 12, -15, 20, 9, 13, 41, 3, 3, 49, -9, 20, -2, -10, -11, 20, 
26, -12, 15, 28, 5, -9, -35, -12, -21, -95, 10, 14, -24, -18, -26, 10, -10, 18, -40, -1, 20, 17, 12, 26, 41, 27, -1, -29, 11, 13, 24, 7, -8, 6, 22, -92, -31, 15, 21, -16, 18, -1, -16, 14, -16, -26, -27, -17, 62, 9, 
20, 26, -5, 8, -67, 11, -35, 20, -11, 4, -15, 18, -23, -14, 12, 13, 20, 24, -13, -31, -31, 11, 10, 25, 26, -31, 35, 11, -11, 37, -11, 11, -26, -11, -39, -26, 11, 13, -13, -7, -26, 19, 38, 32, 8, 17, -4, -13, -20, -22, 
6, 6, 34, 17, -43, -14, -34, 5, -36, 11, 17, 55, 6, -18, -24, -65, -5, 11, 3, -19, -5, -10, -10, 22, -2, -23, -18, -93, -64, -32, 33, 4, 13, 39, -18, 3, 26, 11, -11, 22, -48, 1, 29, -6, 42, -6, -21, -68, 22, 12, 
5, -30, -12, 41, -2, -40, -23, -2, 58, 60, 8, -9, -15, -14, -77, 6, 14, -25, -31, 31, -1, 1, 11, 32, -7, -6, -3, 25, -47, 28, 15, -20, -38, 23, 19, -34, -44, -8, 26, 15, 11, -5, 14, 12, 29, 10, -10, 79, 16, 10, 
-41, -33, 5, -3, -27, -19, -17, 19, 2, -33, 11, 24, -25, -43, -19, 25, -18, 11, 3, 12, -54, -33, 62, 9, 41, -10, -19, 2, -22, 15, 6, -44, 5, 15, -97, -29, -9, -11, -8, 15, 3, 51, -22, -65, 16, 1, 12, 53, -13, -76, 
-6, 17, -8, -36, 22, -46, -68, -44, -11, -28, -18, 17, -4, -1, -6, 6, -18, 30, -55, -6, 12, 6, -7, -6, -86, 33, 35, 26, 34, 3, 11, -55, 2, 38, 3, 11, 17, 38, -5, -28, 12, 14, -31, 44, 33, 22, 7, 14, -52, 41, 
20, -5, -20, 9, 51, -11, -4, -6, 20, 21, -8, -14, -50, -6, 21, 15, -2, 35, 22, -35, 45, 22, 3, -56, 22, -19, 22, 40, 15, -35, -4, 5, -12, 14, -23, 24, 23, 11, 4, 7, -31, -7, 1, -15, 27, -36, -5, 22, -30, -14, 
-30, -28, -52, 12, -15, -14, 14, -27, 3, 4, 18, -1, -38, -25, 21, 9, 17, -38, 0, 25, -15, 11, 21, 18, 14, -21, -38, -30, -36, 22, 8, 16, -36, -2, -8, -8, -12, -7, -31, -35, -2, -57, 19, -50, -7, -19, -37, 6, 20, -9, 
22, 20, -21, 25, 2, 8, 4, -1, -42, -50, 11, -13, 17, -7, -52, 28, -15, -21, -48, 11, 44, 17, 27, -8, -51, 23, -2, 14, -6, -10, 26, 9, 31, -9, 2, -33, -55, 5, 0, -14, 30, -19, 19, -55, -49, -20, -65, 25, 21, -17, 
3, 16, -37, -17, -12, -33, 24, 2, -52, -58, -28, -39, 11, 19, 23, 10, -18, 3, -13, 2, 20, 2, 32, -28, 2, -22, -51, -20, -23, 2, -67, -61, 10, -14, 26, 18, -12, -44, -22, -15, 18, 12, -43, 20, 6, 16, 8, -56, 9, -38, 
-7, 16, 48, 22, -8, -61, 26, 40, 9, -55, -43, 20, -40, 24, -4, -66, -67, 19, 7, 16, -25, 12, -23, -30, -54, -28, 14, -6, -43, -28, -29, -30, -42, 8, 22, 21, 4, -35, 14, -8, -25, 15, 17, -19, 12, -59, -25, -68, 29, 16, 
10, -22, 4, 17, -5, 15, 16, 11, 22, 4, 49, -10, -60, 5, -2, 17, 21, 9, 17, 23, 22, -53, 21, -56, 14, -4, -28, -17, -27, -78, 19, 26, 20, 5, -4, 7, 27, -29, 20, -24, -33, 20, -30, 2, -24, -5, 20, 11, 8, 18, 
21, 2, -1, -34, -73, 2, -11, -23, 14, 7, -28, 5, 5, -1, 38, 0, 16, -6, -20, 20, -17, 12, -36, 35, -18, 16, -16, 32, -1, -21, -22, -3, 16, 0, 11, -43, -21, -11, 25, 0, 34, 16, 6, 43, 4, -16, -7, -16, 0, 18, 
21, 20, 20, -68, 26, 12, -47, 6, -28, 8, -39, -24, 12, -29, 5, 1, -8, -69, -3, 4, 22, -47, -28, -8, -4, 9, -39, -1, -23, -11, -5, 8, 10, 4, -43, 17, 22, -16, 25, 1, -4, -59, -2, -35, -32, 1, 19, -9, -65, 9, 
2, 1, -3, -63, 4, -9, 45, 23, 18, 27, 4, -59, -46, 9, 25, 31, 19, -22, -14, -3, 22, 17, 4, -60, -25, 27, 2, -19, -5, -68, -11, -60, 3, -2, -55, 18, -59, -7, -17, -2, 20, 3, -58, 28, -7, -25, -36, 23, -4, -3, 
-45, 10, -5, 8, 3, -51, 21, 2, 13, 41, -4, 0, -67, 10, -19, -68, 19, 9, -53, 6, -18, 17, 17, 2, -52, 1, -71, 19, -9, -28, -14, -56, 16, 1, 8, 13, 4, 16, 15, 9, -3, 14, 14, -70, 13, -69, -7, -32, 1, -8, 
-15, 19, 1, -36, -34, -39, -19, -25, 26, 12, -30, -43, -47, -41, 21, -35, -66, 0, 4, 18, -12, 5, -54, 14, -25, -76, -44, -1, 3, -78, 0, -31, -26, -30, -12, 4, -33, 5, -17, 16, 14, 13, -9, -36, -24, 32, 4, -45, 14, -2, 
21, 0, -1, 19, -23, 14, -9, 14, -16, -29, -11, 19, 7, 8, 28, -20, 10, 2, 20, 4, 28, 10, -17, -30, -6, -13, 44, -8, -9, 3, 8, 12, 2, -14, -24, 8, 8, 35, 8, 7, -72, 9, -20, 32, 21, -16, -48, -17, -6, -11, 
7, -5, 27, 30, -48, 20, 11, -8, -11, -35, 13, 21, 21, 18, 31, -54, -44, -23, -11, -32, 5, 18, -28, 34, 23, 45, 12, -55, -36, 7, 8, -16, 20, 12, -40, -24, 13, -39, 26, 31, -16, 21, -20, -24, 19, -6, -31, -2, 14, -40, 
-23, -38, -20, -52, -8, 18, -39, 17, 10, 26, -6, 31, -29, -22, 15, 8, -56, 17, -31, -21, 9, -15, 17, 36, 9, -15, 3, 11, -13, 15, -13, 12, 13, -22, -6, 6, -69, -17, 5, -25, 11, 19, 28, 44, 50, -9, -10, -1, -12, 19, 
21, 0, 16, 11, 51, -13, 32, 6, -22, -9, -8, -42, -7, 16, 1, -54, -27, 21, 28, -33, -24, 8, 17, -19, 16, -42, -16, 33, 18, -8, -16, -31, 4, 26, 4, 19, 36, 6, 19, -27, -8, 51, 27, 17, -8, -34, -34, -46, -3, 10, 
-26, 8, -25, -8, -61, -34, 49, 17, 2, 23, 13, -11, 25, -9, -21, -31, -13, -37, -18, -20, -57, 4, -2, -2, -27, -58, -3, -33, 24, 6, 11, -44, 5, 20, 24, 1, -11, -12, 5, 20, 8, 10, -22, 14, -48, -11, 25, -30, -18, -17, 
30, 12, 17, 15, -33, -32, 12, 11, -9, 17, -34, 26, -5, 21, 0, 15, -64, -24, 0, 26, -32, -26, 13, 30, -39, -30, -34, -29, 22, -32, 23, -37, -37, -19, 11, -22, -36, 2, -20, 26, -19, -12, 2, 0, 29, -11, 15, 3, -2, -1, 
-46, 22, 23, 7, -45, 24, -12, 0, 13, 18, 48, -40, 10, -10, -16, 33, -37, 42, -38, -6, 25, 25, 2, -1, -34, 1, -9, 12, 13, 17, -10, -31, -13, -51, -7, -13, 43, -41, 17, -31, -43, 9, -6, -8, -11, -32, -5, 33, -32, 11, 
-27, 24, 0, 26, -11, 9, 33, -18, -61, 26, 46, -26, 9, 15, -6, 20, -14, 20, -34, -24, 12, 22, -14, 7, 11, 30, 35, -31, 9, 3, 26, -37, -14, -34, -24, 12, 24, 17, 17, 65, 20, -14, 20, -35, 16, -12, -10, 6, 10, 25, 
13, -20, 13, -12, 17, -1, 23, -25, 25, -21, -12, 10, -25, -46, 15, -36, 18, -15, 26, 12, -13, -34, 19, 11, 11, -13, -26, 22, -25, 9, 3, 51, -6, -24, -19, -21, -56, 19, 26, 26, -37, -5, -9, 28, -25, -19, 18, 16, -4, 47, 
-10, 55, -18, -23, 28, 24, 12, -23, 32, 18, 55, 34, -10, 7, -49, -4, 22, 18, -7, 15, -23, -41, 49, -4, 11, 0, -32, 55, -52, 15, -6, -10, -2, -55, 6, -24, 25, 24, -10, -35, 2, 24, 15, 7, 56, -29, 7, 9, 20, 23, 
4, 32, -52, 2, -23, -7, 21, -58, -39, 14, -3, 17, -47, 15, 22, 11, -54, -39, -41, 7, 24, 7, 13, 7, 5, 6, -32, 9, 21, 22, 19, 9, -19, -25, -10, 20, -10, -7, -57, 19, 9, 11, 0, 8, -20, -2, 0, -21, 20, -46, 
-3, 13, 29, 2, 14, -38, -3, 0, -41, 2, -10, -13, -18, 21, 22, -22, -37, 13, 11, 2, 0, 12, 12, -63, 7, 12, -14, 9, -8, 8, 12, -7, 6, -3, 15, -28, 20, -100, -9, 13, -2, 11, 16, 23, -14, 1, 15, 20, 8, 18, 
-48, -23, -26, -71, -3, -21, -35, 8, -6, -4, 8, 13, 11, 19, -43, -27, -50, -47, -15, -13, -36, -16, -7, -51, 8, 14, 19, 6, -3, -4, 31, 20, -11, 7, -5, -26, 1, 10, 7, 10, -7, 13, 6, 19, 5, 13, -59, 25, 20, -18, 
17, 15, 18, -2, -47, -61, -20, 14, 25, -7, 1, 18, -16, -4, -56, -8, -58, 21, -56, -21, 21, -17, 7, 15, 20, -24, 7, -41, 13, -49, -27, 0, 16, -9, -27, -85, -27, -18, 21, -60, -15, 8, 5, 11, 6, -11, -2, -8, -27, 17, 
18, 19, 16, -56, -38, 11, 23, 3, 10, 2, -47, -11, 20, 5, 13, 10, -5, 8, 4, 0, 14, -33, -60, 4, -7, 17, 3, -21, 4, 16, 24, 12, -15, 17, -52, 10, 5, -5, 17, 20, -25, 13, 34, 22, 0, 1, -1, -42, -7, 13, 
13, -18, 16, -24, 13, -7, -10, 16, -7, -2, 21, 3, -62, 12, 15, 32, -24, 15, 10, 12, 0, 13, 14, -14, 4, -31, -3, -9, -10, 27, 12, 3, 19, 19, -19, 20, 13, 20, -72, -26, -8, 19, -8, -43, 9, 17, -5, -36, 20, -8, 
1, -31, -5, -5, 11, 1, -25, 17, 15, -3, -8, 17, 3, -39, -45, -1, -33, -21, 10, -58, 0, -12, -30, 9, -1, -16, 9, -23, 4, -14, 4, 24, 8, -53, 11, 23, -29, 10, 13, -18, 14, -8, 4, -12, -61, -103, 14, -93, 19, -57, 
-15, -97, 19, 11, -53, -18, 0, -16, 15, -71, -6, 16, 18, 1, -13, 10, 23, 9, -27, -6, 2, -3, -9, 21, 11, 16, -8, -34, 16, 2, -51, -62, 15, -26, -2, 3, 14, -55, -41, 4, -2, -18, 20, 19, -28, 11, -1, 10, 4, -15, 
16, -28, 4, -41, 8, -30, -6, -26, -71, 0, -43, 11, -2, 21, 3, 16, 2, 14, -38, 6, -7, -5, 16, 17, 9, 25, -14, -9, 17, 15, -33, 2, -9, -11, -5, -13, 1, 35, 22, -41, 30, -54, -29, -31, 19, -6, 9, 20, 31, -11, 
-3, -24, 8, -61, 10, -20, -24, 2, -33, 3, -31, -15, -3, -8, -29, 20, -54, 0, 25, -42, 18, 12, -69, -47, 3, -3, -13, 23, 3, -30, -49, 4, 10, 14, -23, -30, 0, 4, 1, -54, -94, -4, -21, -20, 5, 8, 18, 13, -61, 5, 
22, 19, 21, -24, -8, 12, -11, -5, 4, 16, -32, 6, -9, -72, -53, -9, 15, 43, -20, 3, -27, -2, 12, -38, 15, -46, 1, 21, 43, 16, 19, 18, 19, -6, 14, 18, -1, -67, 0, -31, 8, 4, 4, 11, 27, 16, 7, -11, -41, 16, 
-55, -17, -20, 24, -6, 11, 11, 11, -45, -81, 1, -1, 12, 17, 12, 7, -29, -2, 20, -20, -18, -9, 12, -60, 10, 9, -11, 11, -28, -32, 12, 17, -13, -21, 12, 31, 21, -3, -43, -25, -13, 7, -10, 15, 17, 19, -35, -46, -58, -16, 
10, 0, -24, 5, -17, 2, -86, -8, -38, -13, -21, 24, 0, 5, -19, 19, 1, -72, -77, -1, 4, 5, 13, -13, -22, -4, 20, 28, 5, -11, 1, -24, -46, -43, -3, -81, -34, -41, 19, 0, 1, 22, 25, -18, 20, 9, -4, -62, -8, -4, 
-8, 1, 15, -54, -16, 12, 8, -34, 11, -72, -10, 29, 19, 0, -11, 20, 2, -70, -55, -26, -25, 21, 19, 23, 8, 23, 8, 8, -55, -23, -29, 11, 22, -14, 15, 0, -42, -78, 2, 35, 15, 6, -96, 14, 6, -11, 16, 29, 18, -2, 
14, 0, 11, -11, 13, 8, -66, 19, 11, 5, -60, 15, -23, -12, 6, -94, 16, 21, 3, 10, 14, -4, 45, -1, 1, 27, -15, -45, -18, -30, -72, -13, -33, -22, -5, -49, -29, -11, 4, 23, -14, 9, -16, -11, 8, 19, -16, 13, 24, -13, 
-22, 5, 14, 15, -30, -23, 9, 8, -20, 20, 12, -18, 11, -84, -4, -20, 11, 8, 8, -24, 8, 23, 28, 27, 20, 13, -12, -39, 32, -19, 5, 3, 8, 18, -3, -48, -18, 16, 18, 1, -3, 18, -33, -58, -11, -18, 33, -72, -11, 7, 
15, -18, -1, 16, 21, -71, 16, 4, -45, 3, 19, 11, -1, -28, 12, -10, -1, 15, 9, 9, 8, -18, 3, -65, -10, -11, 9, 29, 38, 10, -15, 15, 6, 5, -103, -10, -62, -60, 17, 1, 32, -44, 4, -28, 2, 19, 19, 25, 6, 9, 
22, -84, 13, 27, 0, -27, 20, -7, -6, -58, -35, 18, -12, 26, -2, 17, -77, 15, -12, -39, 11, -27, -18, -70, 0, -37, 8, -43, -61, -5, -5, -26, -17, 21, -58, -19, -16, 17, 20, 25, 5, 17, 16, 33, 18, -32, 2, -32, 22, 9, 
-43, 15, 9, -3, -24, -64, -57, 10, -13, 17, -7, -28, 15, -60, 8, 18, -2, -11, -12, 20, 14, 2, -4, -25, -64, -17, -39, 20, 16, 16, -47, -2, -13, 21, 0, 11, -17, -1, -15, -26, -17, 24, -25, 16, -14, 28, -86, 6, 4, -1, 
10, -2, -44, 15, -35, -30, 2, -10, -3, 2, 19, 20, 16, -1, 13, 27, -13, 1, -78, -27, 22, -2, 38, -67, 31, -60, 13, 3, -12, 15, -9, 20, -85, 7, -66, -59, 13, 20, -20, 1, 22, -1, 43, 11, -20, 8, -69, -2, 21, 14, 
-18, -2, 52, 13, 9, -23, -5, -28, 15, 13, 31, -7, 12, 0, 22, 2, 22, 17, -66, 3, };

#endif
