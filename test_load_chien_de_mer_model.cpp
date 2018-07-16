#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include <string>
#include <string.h>
#include "test_load.hpp"

const bool True = true;
const bool False = false;

static TF_Buffer* readFile(const char* filename); 
static TF_SessionOptions* pSessOpts = NULL;
static TF_Graph* pGraph = NULL;
static TF_Status* pStatus = NULL;
static TF_Buffer* pRunOptsBuff = NULL;
static TF_Buffer* pMetaGraphBuff = NULL; 
static TF_Session* pSess=NULL;
static TF_Operation *pInpOp=NULL, *pSizeOp=NULL, *pOutOp=NULL;


static bool isInitialized = false;


static void freeBuffer(void* data, size_t ) {
        free(data);
}


static void freeData(void* data, size_t , void*){
	free(data);
}

static void  freeT(void* data, size_t, void*){
}


/** 
 * @brief init initialize for whole lib file
 */
bool init(const char* filePath){
	int i = 0;
	
	printf("\n%d: Initializing Tensorflow module\n", i++);
	if (isInitialized) return true;
	pStatus = TF_NewStatus();
	pSessOpts = TF_NewSessionOptions();
	pGraph = TF_NewGraph();
	pRunOptsBuff = TF_NewBufferFromString("", 0);
	//pRunOptsBuff = TF_NewBuffer();
	pMetaGraphBuff = TF_NewBuffer();
	const char* const tags = {"serve"};
	printf("%d: Allocated Tensorflow's Objects\n", i++);
	pSess = TF_LoadSessionFromSavedModel(
			pSessOpts, pRunOptsBuff, filePath, &tags, 1,
		 pGraph, pMetaGraphBuff,	pStatus);
	if(TF_GetCode(pStatus) != TF_OK){
		// print some errors here...
		fprintf(stderr, "ERROR: Unable to create session %s", TF_Message(pStatus));
		return False;
	}
	isInitialized = True;
	pInpOp = TF_GraphOperationByName(pGraph, "X");
	pSizeOp = TF_GraphOperationByName(pGraph, "T");
	pOutOp = TF_GraphOperationByName(pGraph, "softmax");
	if (pInpOp == NULL || pSizeOp == NULL || pOutOp == NULL){
		fprintf(stderr, "ERROR: Unable to load operations");
	}

	printf("%d: Getting input & output params\n", i++);
	int numDims = TF_GraphGetTensorNumDims(pGraph, {pInpOp, 0}, pStatus);
	int64_t* pDims = NULL;
	int j = 0; 
	if (TF_GetCode(pStatus) != TF_OK){
		fprintf(stderr, "ERROR: Unable to read number of dimmension of op - %s\n", TF_Message(pStatus));
	}
	printf("Input Params %d: \n", j++);
	printf("- Input Data Size Dim: %d\n", numDims);
	pDims = (int64_t*) malloc(numDims*sizeof(int64_t));
	TF_GraphGetTensorShape(pGraph, {pInpOp, 0}, pDims, numDims, pStatus);
	printf("- Input Data Shape: ");
	for (int k = 0; k < numDims; k++) printf("%ld, ", pDims[k]);
	printf("\n");


	printf("Input Params %d: \n", j++);
	numDims = TF_GraphGetTensorNumDims(pGraph, {pSizeOp, 0}, pStatus);
	if (TF_GetCode(pStatus) != TF_OK){
		fprintf(stderr, "ERROR: Unable to read number of dimmension of op - %s\n", TF_Message(pStatus));
	}
	printf("- Input Data Size Dim: %d\n", numDims);
	pDims = (int64_t*) realloc(pDims, numDims*sizeof(int64_t));
	TF_GraphGetTensorShape(pGraph, {pSizeOp, 0}, pDims, numDims, pStatus);
	printf("- Input Data Shape: ");
	for (int k = 0; k < numDims; k++) printf("%ld, ", pDims[k]);
	printf("\n");


	printf("Output Params %d: \n", j++);
	numDims = TF_GraphGetTensorNumDims(pGraph, {pOutOp, 0}, pStatus);
	if (TF_GetCode(pStatus) != TF_OK){
		fprintf(stderr, "ERROR: Unable to read number of dimmension of op - %s\n", TF_Message(pStatus));
	}
	printf("- Output Data Size Dim: %d\n", numDims);
	pDims = (int64_t*) realloc(pDims, numDims*sizeof(int64_t));
	TF_GraphGetTensorShape(pGraph, {pOutOp, 0}, pDims, numDims, pStatus);
	printf("- Output Data Shape: ");
	for (int k = 0; k < numDims; k++) printf("%ld, ", pDims[k]);
	printf("\n");

	free(pDims);
	isInitialized = True;
	return isInitialized;
}


static void runSession(TF_Tensor* pInpTensor, TF_Tensor* pSizeTensor, TF_Tensor** ppOutputTensors){
	if (!isInitialized) init("cho_dat");
	TF_Output inps[] = {{pInpOp, 0}, {pSizeOp, 0}};
	TF_Tensor* pInpVals[] = {pInpTensor, pSizeTensor};
	TF_Output outs[] = {{pOutOp, 0}};
	TF_SessionRun(pSess,
		 	NULL,
		 	inps, pInpVals, 2,
		 	outs, ppOutputTensors, 1,
		 	NULL, 0, NULL, pStatus);
	if(TF_GetCode(pStatus) != TF_OK){
		// print some errors here...
		fprintf(stderr, "ERROR: Unable to run session %s\n", TF_Message(pStatus));
	}

}


void predictTF(int64_t numSample, int32_t T, float* inpData, float* outputBuffer){
	int64_t pInpDims[] = {numSample, (int64_t)T, 39, 1};
	int i = 0;
	printf("\n%d: Start predicting using C API...\n", i++);
	TF_Tensor* pInpTensor = TF_NewTensor(TF_FLOAT, pInpDims, 4, inpData, sizeof(float)*numSample*T*39, freeData, NULL);
	int64_t pInpDimsT[] = {1};
	printf("%d: Initialized First tensor\n", i++);
	TF_Tensor* pInpSizeTensor = TF_NewTensor(TF_INT32, pInpDimsT, 1, &T, sizeof(int32_t), freeT, NULL);
	printf("%d: Initialized Second tensor\n", i++);
	TF_Tensor* pOutputTensor = NULL;
	printf("%d: Running session...\n", i++);
	runSession(pInpTensor, pInpSizeTensor, &pOutputTensor);
	// TF_DeleteTensor(pInpTensor);
	// TF_DeleteTensor(pInpSizeTensor);
	printf("%d: Finished running session\n", i++);
	printf("- Output parameters: \n");
	printf("Number of dimmension - %d\n", TF_NumDims(pOutputTensor));
	printf("Output Tensor Size in number of floats - %ld\n", TF_TensorByteSize(pOutputTensor)/sizeof(float));
	// memcpy(outputBuffer, (void*)TF_TensorData(pOutputTensor), 2*sizeof(float));
	// TF_DeleteTensor(pOutputTensor);
	printf("%d\n", i++);
}


void closeTF(){
	TF_CloseSession(pSess, pStatus);
}


static TF_Buffer* readFile(const char* filename){
	FILE *f = fopen(filename, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  //same as rewind(f);
  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);
  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = freeBuffer;
  return buf;
}
