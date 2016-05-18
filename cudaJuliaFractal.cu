#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <stdint.h>

extern "C" {
  #include "png_util.h"
}

#define T 16

/**
 * Uses kernel to iterate through image pixels.
 * Calculates the color for each pixel from the equation for a Julia Fractal.
 */
__global__ void cudaDrawKernel(double cR, double cI, int size, int iter, int64_t *c_julia) {

  double newZR, newZI;
  int x,y;

  x = (threadIdx.x + blockDim.x*blockIdx.x);
  y = (threadIdx.y + blockDim.y*blockIdx.y);

  if (x<size && y<size) {
    newZR = 1.5*(x-size*0.5)/(size*0.5);
    newZI = (y-size*0.5)/(size*0.5);

    double oldZR, oldZI, color;
    int i, count = 0;

    for (i = 0; i < iter; i++) {
      oldZR = newZR;
      oldZI = newZI;

      newZR = oldZR*oldZR - oldZI*oldZI + cR;
      newZI = (2.f)*oldZR*oldZI + cI;
    
      if ((newZR*newZR + newZI*newZI) <= 4.0) {
	count++;
      }
    }

    color = sqrt((double)count);
    c_julia[y+x*size] = color;
  }
}

/**
 * Sets image variables.
 * Starts Cuda kernel.
 * Creates Image from data.
 */
int main(int argc, char** argv) {
  
  int size = atoi(argv[1]);
  int iter = atoi(argv[2]);
  double cR = -0.778;
  double cI = -0.116;
  
  
  /* Start Cuda Time */

  cudaEvent_t tic, toc;
  cudaEventCreate(&tic);
  cudaEventCreate(&toc);
  cudaEventRecord(tic, 0);
  
  
  /* Allocate and Copy Memory for Cuda */

  int64_t *h_julia = (int64_t*) calloc(size*size, sizeof(int64_t));
  int64_t *c_julia;
  cudaMalloc(&c_julia, size*size*sizeof(int64_t));
  cudaMemcpy(c_julia, h_julia, size*size*sizeof(int64_t), cudaMemcpyHostToDevice);

  
  /* Run the kernel */
  
  int g = (size+T-1)/T;
  dim3 gDim(g, g);
  dim3 bDim(T, T);
  cudaDrawKernel <<< gDim, bDim >>> (cR, cI, size, iter, c_julia);

  
  /* Copy Memory back from Cuda */
  
  cudaMemcpy(h_julia, c_julia, size*size*sizeof(int64_t), cudaMemcpyDeviceToHost);

  
  /* End Cuda Time */
  
  cudaEventRecord(toc, 0);
  cudaEventSynchronize(toc);
  float elapsed;
  cudaEventElapsedTime(&elapsed, tic, toc);
  printf("Elapsed time: %g\n", elapsed/1000.0);

  
  /* Image Creation */
  
  double timeA = clock();

  FILE *png = fopen("CudaJuliaFractal.png", "w");
  write_hot_png(png, size, size, h_julia, 0, 80);
  fclose(png);

  double timeB = clock();
  double elapsedPic = (timeB-timeA)/CLOCKS_PER_SEC;
  printf("Image creation time: %f\n", elapsedPic);

  return 0;
}
