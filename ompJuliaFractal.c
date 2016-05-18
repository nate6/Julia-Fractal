#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "png_util.h"

/**
 * The Julia Fractal Equation to set the color for the given pixel.
 */
float iterate(float newZR, float newZI, float cR, float cI, int iter) {
  
  float oldZR, oldZI;
  int i, count = 0;

  for (i = 0; i < iter; i++) {
    oldZR = newZR;
    oldZI = newZI;

    newZR = oldZR*oldZR - oldZI*oldZI + cR;
    newZI = (2.f)*oldZR*oldZI + cI;
    
    if ((newZR*newZR + newZI*newZI) <= 4.f) {
      count++;
    }
  }

  float color = sqrtf((float)count);
  return color;
}

/**
 * Starts OpenMP iteration through array of the image pixels.
 */
void draw(float cR, float cI, int size, int iter, float *julia, int n) {

  float newZR, newZI;
  int x,y;

  /* Preferable to run with 12 processing cores */
  omp_set_num_threads(12);

#pragma omp parallel for num_threads(n) private(x)
  for (y = 0; y < size; y++) {
    for (x = 0; x < size; x++) {
      newZR = (float) (1.5*(x-size*0.5)/(size*0.5));
      newZI = (float) ((y-size*0.5)/(size*0.5));
      julia[y+x*size] = (float) iterate(newZR, newZI, cR, cI, iter);
    }
  }
}

/**
 * Sets up fractal variables.
 * Calls method to start the equation for a Julia Fractal.
 * Creates Image from the data.
 */
int main(int argc, char** argv) {
  
  /* Acquire and Set Variables */
  int size = atoi(argv[1]);
  int iter = atoi(argv[2]);
  int n = atoi(argv[3]);
  float cR = -0.778;
  float cI = -0.116;
  
  /* Allocate Memory */
  float *julia = (float*) calloc(size*size, sizeof(float));
  double timeA = clock();

  /* Start Equation Iterations */
  draw(cR, cI, size, iter, julia, n);
  
  /* Clock in the time it took to iterate */
  double timeB = clock();
  double elapsed = (timeB-timeA)/CLOCKS_PER_SEC;
  printf("Thread: %d, Elapsed time: %f\n", n, elapsed);

  /* Creates the Image */
  FILE *png = fopen("OmpJuliaFractal.png", "w");
  write_hot_png(png, size, size, julia, 0, 80);
  fclose(png);

  return 0;
}
