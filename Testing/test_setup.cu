#include <vector_types.h>
#include <Testing/test_setup.h>
#include <Testing/test_setup.cuh>

const unsigned int BLOCKDIM_X = 32;
const unsigned int BLOCKDIM_Y = 32;

__global__ void test_kernel(uchar4 *dst, const int gridWidth, const int numBlocks,const  dim3 grid) {
	//unsigned int blockX = blockIdx.x ;
	//unsigned int blockY = blockIdx.y ;

	//// process this block
	//const int ix = blockDim.x * blockX + threadIdx.x;
	//const int iy = blockDim.y * blockY + threadIdx.y;

	

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	//int offset = 

	dst[ix + iy * gridWidth].x = (unsigned char)((float)ix / blockDim.x * 256);
	dst[ix + iy *gridWidth].y = (unsigned char)((float)iy / blockDim.y * 256);
}

void run_test(uchar4 *dst, const int numSMs, const int imageW, const int imageH)
{
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 num_blocks = dim3(imageW/BLOCKDIM_X,imageH/ BLOCKDIM_Y);

	int numWorkerBlocks = numSMs;

	test_kernel << <num_blocks, threads >> > (dst, imageW, imageH, num_blocks);
	
	
} // RunMandelbrot1