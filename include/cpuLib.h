

#ifndef CPU_LIB_H
#define CPU_LIB_H

	#include <iostream>
	#include <cstdlib>
	#include <ctime>
	#include <random>
    #include <iomanip>
	#include <chrono>
	#include <cstring>
	#include <cstdarg>
	#include <fstream>
	#include <vector>
	#include <algorithm>
	
	// Uncomment this to suppress console output
	// #define DEBUG_PRINT_DISABLE

	extern void dbprintf(const char* fmt...);


	extern void vectorInit(float* v, int size);
	extern int verifyVector(float* a, float* b, float* c, float scale, int size);
	extern void printVector(float* v, int size);
	
	extern void saxpy_cpu(float* x, float* y, float scale, uint64_t size);

	extern int runCpuSaxpy(uint64_t vectorSize);

	extern int runCpuMCPi(uint64_t iterationCount, uint64_t sampleSize);


	////    Lab 2    ////

	typedef struct ImageDim_t
	{
		uint32_t width;
		uint32_t height;
		uint32_t channels;
		uint32_t pixelSize;
	} ImageDim;

	extern std::ostream& operator<< (std::ostream &o,ImageDim imgDim);
	
	/**
	 * @brief 
	 * 
	 * @param bytesFilePath 
	 * @param imgDim 
	 * @param imgData 
	 * @return int 
	 */
	extern int loadBytesImage(std::string bytesFilePath, ImageDim &imgDim, uint8_t ** imgData);

	extern int writeBytesImage(std::string outPath, ImageDim &imgDim, uint8_t * outData);

	typedef struct MedianFilterArgs_t {
		uint32_t filterH;
		uint32_t filterW;
	} MedianFilterArgs;

	extern int medianFilter_cpu(uint8_t inPixels, ImageDim imgDim, 
		uint8_t outPixels, MedianFilterArgs args);

	extern int runCpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args);

	enum class PoolOp{MaxPool, AvgPool, MinPool};

	extern std::ostream& operator<< (std::ostream &o,PoolOp op);

	typedef struct TensorShape_t {
		uint32_t width;		//	Width = # cols
		uint32_t height;	//	Height = # rows
                uint32_t channels;	//	
	} TensorShape;

	typedef struct PoolLayerArgs_t {
		PoolOp opType;
		uint32_t poolH;		//	pooling rows
		uint32_t poolW;		//	pooling cols
		uint32_t strideH;
		uint32_t strideW;
	} PoolLayerArgs;


	/**
	 * @brief 
	 * 
	 * @param input 
	 * @param inShape 
	 * @param output 
	 * @param outShape 
	 * @param args 
	 * @return int 
	 */
	extern int poolLayer_cpu (float * input, TensorShape inShape, 
		float * output, TensorShape outShape, PoolLayerArgs args);


	extern int runCpuPool (TensorShape inShape, PoolLayerArgs poolArgs);
	

#endif
