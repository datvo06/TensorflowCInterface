#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include <string>
#include <string.h>

const bool True = true;
const bool False = false;

TF_Buffer* readFile(const char* filename); 
TF_SessionOptions* pSessOpts = NULL;
TF_Graph* pGraph = NULL;
TF_Status* pStatus = NULL;
TF_Buffer* pRunOptsBuff = NULL;
TF_Buffer* pMetaGraphBuff = NULL; 
TF_Session* pSess=NULL;
TF_Operation *pInpOp=NULL, *pSizeOp=NULL, *pOutOp=NULL;


bool isInitialized = false;


void freeBuffer(void* data, size_t ) {
        free(data);
}


void freeData(void* data, size_t , void*){
	free(data);
}


/** 
 * @brief init initialize for whole lib file
 */
bool init(const char* filePath){
	if (isInitialized) return true;
	pStatus = TF_NewStatus();
	pSessOpts = TF_NewSessionOptions();
	pGraph = TF_NewGraph();
	pRunOptsBuff = TF_NewBuffer();
	pMetaGraphBuff = TF_NewBuffer();
	pSess = TF_LoadSessionFromSavedModel(
			pSessOpts, pRunOptsBuff, filePath, NULL, 0,
		 pGraph, pMetaGraphBuff,	pStatus);
	if(TF_GetCode(pStatus) != TF_OK){
		// print some errors here...
		fprintf(stderr, "ERROR: Unable to create session %s", TF_Message(pStatus));
		return False;
	}
	isInitialized = True;
	pInpOp = TF_GraphOperationByName(pGraph, "a");
	pSizeOp = TF_GraphOperationByName(pGraph, "a");
	pOutOp = TF_GraphOperationByName(pGraph, "a");
	return isInitialized;
}


void runPrediction(TF_Tensor* pInpTensor, TF_Tensor* pSizeTensor, TF_Tensor* pOutputTensor){
	TF_Output inps[] = {{pInpOp, 0}, {pSizeOp, 1}};
	TF_Tensor* pInpVals[] = {pInpTensor, pSizeTensor};
	TF_Output outs[] = {{pOutOp, 0}};
	TF_SessionRun(pSess, NULL, inps, pInpVals, 2, outs, &pOutputTensor, 1, &pOutOp, 1, NULL, pStatus);
}

TF_Buffer* readFile(const char* filename){
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
