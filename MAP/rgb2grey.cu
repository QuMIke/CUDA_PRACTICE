#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <opencv.hpp>
#include <string>
#include <stdio.h>

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
	uchar4 **d_rgbaImage, unsigned char **d_greyImage,
	const std::string &filename) {

	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	//allocate memory for the output
	imageGrey.create(image.rows, image.cols, CV_8UC1);

	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);		
	}

	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();
	//allocate memory on the device for both input and output
	cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels);
	cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels);
	cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)); //make sure no memory is left laying around

																	//copy input array to the GPU
	cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
	const int numPixels = numRows() * numCols();
	//copy the output back to the host
	cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);

	//output the image
	cv::imwrite(output_file.c_str(), imageGrey);

	//cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
}

__global__ void rgba_to_greyscale(const uchar4* const rgbaImage, unsigned char* const greyImage, int numRows, int numCols)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIndex < numCols && yIndex < numRows)
	{
		uchar4 rgb = rgbaImage[yIndex * numCols + xIndex];

		greyImage[yIndex * numCols + xIndex] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
	}
}

void your_rgba_to_greyscale(uchar4 * const d_rgbaImage,
	unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
	//You must fill in the correct sizes for the blockSize and gridSize
	//currently only one block with one thread is being launched
	int m = 32;
	const dim3 blockSize(m, m, 1);
	const dim3 gridSize(numCols / m + 1, numRows / m + 1, 1);
	rgba_to_greyscale <<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
	cudaDeviceSynchronize();
}


int main(int argc, char **argv) {
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;

	std::string input_file = "2-31.jpg";
	std::string output_file = "2_31.jpg";

	//load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

	//call the students' code
	your_rgba_to_greyscale(d_rgbaImage, d_greyImage, numRows(), numCols());
	cudaDeviceSynchronize();
	printf("\n");

	//check results and output the grey image
	postProcess(output_file);

	return 0;
}
