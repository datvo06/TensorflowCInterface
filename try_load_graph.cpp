#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>

TF_Buffer* readFile(const char* filename); 


void freeBuffer(void* data, size_t ) {
        free(data);
}


void freeData(void* data, size_t , void*){
	free(data);
}


int main(int argc, char** argv){
	// 1. Set session options
	TF_SessionOptions* pSessOpts = TF_NewSessionOptions();
	// 2. Graph
	TF_Buffer* pGraphDef = readFile("models/graph.pb");                      
	TF_Graph* pGraph = TF_NewGraph();

	// check status
	TF_Status* pStatus = TF_NewStatus();
	TF_ImportGraphDefOptions* pGraphOpts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(pGraph, pGraphDef, pGraphOpts, pStatus);
	if(TF_GetCode(pStatus) != TF_OK){
		fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(pStatus));
	}
	fprintf(stdout, "Successfully imported graph");
	TF_DeleteBuffer(pGraphDef);

	// New session
	TF_Session* pSess = TF_NewSession(pGraph, pSessOpts, pStatus);
	if(TF_GetCode(pStatus) != TF_OK){
		// print some errors here...
		fprintf(stderr, "ERROR: Unable to create session %s", TF_Message(pStatus));
	}

	// Setup inputs and outputs
	int64_t dims[] = {1};
	float *pDataA = (float*) malloc(sizeof(float));
	*pDataA = 3.0;
	TF_Tensor* pTensorA = TF_NewTensor(TF_FLOAT, dims, 1, (void*)pDataA, sizeof(float), freeData, (void*) NULL);
	TF_Operation* pOpA = TF_GraphOperationByName(pGraph, "a");

	float *pDataB = (float*) malloc(sizeof(float));
	*pDataB = 3.0;
	TF_Tensor* pTensorB = TF_NewTensor(TF_FLOAT, dims, 1, (void*)pDataB, sizeof(float), freeData, (void*) NULL);
	TF_Operation* pOpB = TF_GraphOperationByName(pGraph, "b");

	TF_Operation* pOpC = TF_GraphOperationByName(pGraph, "c");

	TF_Tensor* pTensorC = TF_NewTensor(TF_FLOAT, dims, 1, (void*)pDataA, sizeof(float), freeData, (void*) NULL);
	//
	TF_Tensor* pInpVals[] = {pTensorA, pTensorB};
	TF_Output inps[] = {{pOpA, 0}, {pOpB, 0}};
	TF_Output graphOut = {pOpC, (int)0};
	TF_SessionRun(pSess, NULL, inps, pInpVals, 2,
			&graphOut, &pTensorC, 1, &pOpC, 1, NULL, pStatus);

	if(TF_GetCode(pStatus) != TF_OK){
		// print some errors here...
		fprintf(stderr, "ERROR: Unable to run session %s", TF_Message(pStatus));
	}
	float* pData = (float*)TF_TensorData(pTensorC);
	printf("\nOutput: %f", *pData);

	TF_DeleteStatus(pStatus);
	return 0;
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
