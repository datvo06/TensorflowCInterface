#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include <string>
#include <string.h>
#include <stack>
#include <vector>
#include "StatusSingleton.hpp"
#include <map>


const bool True = true;
const bool False = false;

typedef struct {
	TF_SessionOptions* pSessOpts;
	TF_Graph* pGraph;
	TF_Buffer* pRunOptsBuff;
	TF_Buffer* pMetaGraphBuff; 
	TF_Session* pSess;
	std::map<std::string, TF_Operation*> inpDict;
	std::map<std::string, TF_Operation*> outDict;
} TFModelUnit;

static TFModelUnit allModel = {
	NULL, NULL, NULL, NULL, NULL,
 	std::map<std::string, TF_Operation*>(),
 	std::map<std::string, TF_Operation*>(),
};

static TFModelUnit cnnModel = {
	NULL, NULL, NULL, NULL, NULL,
 	std::map<std::string, TF_Operation*>(),
 	std::map<std::string, TF_Operation*>(),
};
static TFModelUnit rnnModel = {
	NULL, NULL, NULL, NULL, NULL,
 	std::map<std::string, TF_Operation*>(),
 	std::map<std::string, TF_Operation*>()};

static TF_Buffer* readFile(const char* filename); 

static bool isInitialized = false;


static void freeBuffer(void* data, size_t ) {
        free(data);
}


static void freeData(void* data, size_t , void*){
	free(data);
}

static void  freeT(void* data, size_t, void*){
}


static void printTFOpParam(TF_Graph* pGraph, std::string name, TF_Operation* pOp){
	printf("name - %s", name.c_str());
	int numDims = TF_GraphGetTensorNumDims(
			pGraph, {pOp, 0}, TFStatusSingleton::instance().getStatus());
	int64_t* pDims = NULL;
	if (TF_GetCode(TFStatusSingleton::instance().getStatus()) != TF_OK){
		fprintf(stderr, "ERROR: Unable to read number of dimmension of op - %s\n", TF_Message(TFStatusSingleton::instance().getStatus()));
	}
	printf("\nParams: \n");
	printf("- Data Size Dim: %d\n", numDims);
	pDims = (int64_t*) malloc(numDims*sizeof(int64_t));
	TF_GraphGetTensorShape(pGraph, {pOp, 0}, pDims, numDims, TFStatusSingleton::instance().getStatus());
	printf("- Data Shape: ");
	for (int k = 0; k < numDims; k++) printf("%ld, ", pDims[k]);
	printf("\n");
	free(pDims);

}


bool initModel(const char* filePath, TFModelUnit* pModelUnit, const std::vector<std::string>& inputNames, const std::vector<std::string>& outputNames){
	int i = 0;
	TF_Status* pStatus = TFStatusSingleton::instance().getStatus();
	pModelUnit->pSessOpts = TF_NewSessionOptions();
	pModelUnit->pGraph = TF_NewGraph();

	pModelUnit->pRunOptsBuff = TF_NewBufferFromString("", 0);
	pModelUnit->pMetaGraphBuff = TF_NewBuffer();
	TF_Buffer* pGraphDef = readFile(filePath) ;
	TF_ImportGraphDefOptions* pGraphOpts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(pModelUnit->pGraph, pGraphDef, pGraphOpts, pStatus);

	if(TF_GetCode(pStatus) != TF_OK){
		// print some errors here...
		fprintf(stderr, "ERROR: Unable Load GraphDef: %s", TF_Message(pStatus));
		return False;
	}

	pModelUnit->pSess = TF_NewSession(pModelUnit->pGraph, pModelUnit->pSessOpts, pStatus);

	if(TF_GetCode(pStatus) != TF_OK){
		// print some errors here...
		fprintf(stderr, "ERROR: Unable to create session %s", TF_Message(pStatus));
		return False;
	}
	for(size_t i = 0; i < inputNames.size(); i++){
		pModelUnit->inpDict[inputNames[i]] = TF_GraphOperationByName(pModelUnit->pGraph, inputNames[i].c_str());
	}
	for(size_t i = 0; i < outputNames.size(); i++){
		pModelUnit->outDict[std::string(outputNames[i])] = TF_GraphOperationByName(pModelUnit->pGraph, outputNames[i].c_str());
	}
	printf("%d: Getting input & output params\n", i++);
	printf("\n- Inputs: ");
	for (auto i = pModelUnit->inpDict.begin(); i != pModelUnit->inpDict.end(); i++){
		if (i->second == NULL) {
			fprintf(stderr, "ERROR: Unable to load operations");
			return False;
		}
		printTFOpParam(pModelUnit->pGraph, i->first, i->second);
	}
	printf("\n- Outputs: ");
	for (auto i = pModelUnit->outDict.begin(); i != pModelUnit->outDict.end(); i++){
		if (i->second == NULL) {
			fprintf(stderr, "ERROR: Unable to load operations");
			return False;
		}
		printTFOpParam(pModelUnit->pGraph, i->first, i->second);
	}

	return True;
}

/** 
 * @brief init initialize for whole lib file
 */
bool initTF(const char* cnnFilePath, const char* rnnFilePath, const char* allFilePath){
	int i = 0;
	printf("\n%d: Initializing Tensorflow module\n", i++);
	if (isInitialized) return true;

	std::vector<std::string> inputNamesCNN; inputNamesCNN.push_back("X");
	std::vector<std::string> outputNamesCNN; outputNamesCNN.push_back("X_conv_relu");

	std::string rnnInpNamesArray[] = {"X_conv_input", "T"};
	std::vector<std::string> inputNamesRNN(rnnInpNamesArray,
		 	rnnInpNamesArray+sizeof(rnnInpNamesArray)/sizeof(std::string));
	std::vector<std::string> outputNamesRNN; outputNamesRNN.push_back("softmax");

	if (allFilePath != NULL){
		std::string allInpNamesArray[] = {"X", "T"};
		std::vector<std::string> inputNamesALL(allInpNamesArray,
				allInpNamesArray+sizeof(allInpNamesArray)/sizeof(std::string));

		initModel(allFilePath, &allModel, inputNamesALL, outputNamesRNN);
		printf("\n%d: Initialized ALL Model\n", i++);
	}


	initModel(cnnFilePath, &cnnModel, inputNamesCNN, outputNamesCNN);
	printf("\n%d: Initialized CNN Model\n", i++);
	
	initModel(rnnFilePath, &rnnModel, inputNamesRNN, outputNamesRNN);
	printf("\n%d: Initialized RNN Model\n", i++);
	isInitialized = True;

	return isInitialized;
}


static float getElement(TF_Tensor* pTensor, const std::vector<int>& index){
	float* pData = (float*)TF_TensorData(pTensor);
	size_t offset = 0;
	size_t coeff = 1;
	for(int i = index.size()-2; i >= 0; i--){
		coeff *= TF_Dim(pTensor, i+1);
		offset += coeff*index[i];
	}
	offset += index.back();
	return pData[offset];
}


static void recursivePrint(TF_Tensor* pTensor, int currentDim, std::vector<int> offsets){
	if (currentDim == 0) printf("[\n");
	if (currentDim !=  TF_NumDims(pTensor) - 1){
		for(int i = 0; i < TF_Dim(pTensor, currentDim); i++){
			if (currentDim < TF_NumDims(pTensor) -2){
				printf("\n");
			}
			if (currentDim < TF_NumDims(pTensor) -1)
				for(int i = 0; i < currentDim+1; i++) printf("\t");
			printf("[");
			std::vector<int> newOffsets = offsets;
			newOffsets.push_back(i);
			recursivePrint(pTensor, currentDim+1, newOffsets);
			if (currentDim < TF_NumDims(pTensor) -2){
				printf("\n");
			}
			if (currentDim < TF_NumDims(pTensor) -1)
				for(int i = 0; i < currentDim; i++) printf("\t");
			printf("]");
			if (i < TF_Dim(pTensor, currentDim)	-1) printf(",");
			printf("\n");
		}
	}
	else{
		for(int i=0; i < TF_Dim(pTensor, currentDim) - 1; i++){
			offsets.push_back(i);
			printf("%f, ", getElement(pTensor, offsets));
			offsets.pop_back();
		}
		offsets.push_back(TF_Dim(pTensor, currentDim) - 1);
		printf("%f", getElement(pTensor, offsets));
	}
	if (currentDim == 0) printf("]\n");
}


void printTFTensor(TF_Tensor* pTensor){
	recursivePrint(pTensor, 0, std::vector<int>());
}


std::vector<int> getTFTensorDim(TF_Tensor* pTensor){
	std::vector<int> dims;
	for(int i = 0; i < TF_NumDims(pTensor); i++){
		dims.push_back(TF_Dim(pTensor, i));
	}
	return dims;
}


TF_Tensor* predictTFWithData(TFModelUnit* pModel, std::map<std::string, float*> inpDataDict,
		std::map<std::string, std::vector<int64_t>> inpSizeDict,
		std::map<std::string, TF_DataType> inpTypeDict, std::vector<std::string> outputOpsString){
	TF_Output* inps = (TF_Output*)malloc(inpDataDict.size()*sizeof(TF_Output));
	TF_Tensor* pOutputTensor = NULL;
	TF_Output* outs = (TF_Output*)malloc(outputOpsString.size()*sizeof(TF_Output));
	TF_Tensor** pAllInpTensors = (TF_Tensor**) malloc(inpDataDict.size()*sizeof(TF_Tensor*));
	std::vector<std::string> keyVecs;
	for (const auto& keyValPair: inpDataDict){
		keyVecs.push_back(keyValPair.first);
	}
	for (auto keyIt = keyVecs.begin(); keyIt != keyVecs.end(); keyIt ++){
		int bufferSize = 1;
		for (auto dimSize: inpSizeDict[(*keyIt)]){
			bufferSize *= dimSize;
		}
		// Use freeT instead so tensorflow will not free our data, we manage it our self
		pAllInpTensors[keyIt-keyVecs.begin()] = TF_NewTensor(
				inpTypeDict[(*keyIt)], (const int64_t*)inpSizeDict[(*keyIt)].data(), (int)inpSizeDict[(*keyIt)].size(),
			 	inpDataDict[(*keyIt)], sizeof(float)*bufferSize, freeT, NULL);
		inps[keyIt-keyVecs.begin()] = {pModel->inpDict[(*keyIt)], 0};
	}
	for (auto strIt = outputOpsString.begin(); strIt != outputOpsString.end(); strIt++){
		outs[strIt-outputOpsString.begin()] = {pModel->outDict[*strIt], 0};
	}
	TF_SessionRun(pModel->pSess,
			NULL,
			inps, pAllInpTensors, (int)keyVecs.size(),
			outs, &pOutputTensor, (int)outputOpsString.size(),
			NULL, 0, NULL, TFStatusSingleton::instance().getStatus());
	for (size_t i = 0; i < keyVecs.size(); i++){
		TF_DeleteTensor(pAllInpTensors[i]);
	}
	if (TF_OK != TF_GetCode(TFStatusSingleton::instance().getStatus())){
		fprintf(stderr, "\nERROR: Failed to run TF model - %s", TF_Message(TFStatusSingleton::instance().getStatus()));
	}
	return pOutputTensor;
}


TF_Tensor* predictTF(float* inpData, int32_t inpSize){
	int64_t pInpDims[] = {1, inpSize, 39, 1};
	int64_t pInpDimsT[] = {1};

	float* buffer = (float*)malloc(39*sizeof(float)*inpSize);
	memcpy(buffer, inpData, inpSize*39*sizeof(float));

	TF_Tensor* pInpTensor = TF_NewTensor(TF_FLOAT, pInpDims, 4, buffer, sizeof(float)*39*inpSize, freeData, NULL);
	TF_Tensor* pInpSizeTensor = TF_NewTensor(TF_INT32, pInpDimsT, 1, &inpSize, sizeof(int32_t), freeT, NULL);
	TF_Tensor* pOutputTensor = NULL;
	TF_Tensor* pALLInpTensors[] = {pInpTensor, pInpSizeTensor};
	
	TF_Output inps[] = {{allModel.inpDict["X"], 0}, {allModel.inpDict["T"], 0}};
	TF_Output outs[] = {{allModel.outDict["softmax"], 0}};
	TF_SessionRun(allModel.pSess,
		 	NULL,
			inps, pALLInpTensors, 2,
			outs, &pOutputTensor, 1,
			NULL, 0, NULL, TFStatusSingleton::instance().getStatus());
	TF_DeleteTensor(pInpTensor);
	if (TF_OK != TF_GetCode(TFStatusSingleton::instance().getStatus())){
		fprintf(stderr, "\nERROR: Failed to run CNN model - %s", TF_Message(TFStatusSingleton::instance().getStatus()));
	}
	return pOutputTensor;
}


TF_Tensor* predictTFCNN(float* inpData, int32_t inpSize){
	int64_t pInpDims[] = {1, inpSize, 39, 1};

	float* buffer = (float*)malloc(39*sizeof(float)*inpSize);
	memcpy(buffer, inpData, inpSize*39*sizeof(float));

	TF_Tensor* pInpTensor = TF_NewTensor(TF_FLOAT, pInpDims, 4, buffer, sizeof(float)*39*inpSize, freeData, NULL);
	TF_Tensor* pOutputTensor = NULL;
	
	TF_Output inps[] = {{cnnModel.inpDict["X"], 0}};
	TF_Output outs[] = {{cnnModel.outDict["X_conv_relu"], 0}};
	TF_SessionRun(cnnModel.pSess,
		 	NULL,
			inps, &pInpTensor, 1,
			outs, &pOutputTensor, 1,
			NULL, 0, NULL, TFStatusSingleton::instance().getStatus());
	TF_DeleteTensor(pInpTensor);
	if (TF_OK != TF_GetCode(TFStatusSingleton::instance().getStatus())){
		fprintf(stderr, "\nERROR: Failed to run CNN model - %s", TF_Message(TFStatusSingleton::instance().getStatus()));
	}
	return pOutputTensor;
}



TF_Tensor* predictTFRNN(float* inpData, int32_t T){
	int64_t pInpDims[] = {1, int64_t((T-15)/6)+1, 48*18};
	int64_t pInpDimsT[] = {1};

	float* buffer = (float*) malloc(48*18*T*sizeof(float));
	memcpy(buffer, inpData, 48*18*sizeof(float)*(int64_t((T-15)/6)+1));

	TF_Tensor* pInpSizeTensor = TF_NewTensor(TF_INT32, pInpDimsT, 1, &T, sizeof(int32_t), freeT, NULL);
	TF_Tensor* pInpTensor = TF_NewTensor(TF_FLOAT, pInpDims, 3, buffer, sizeof(float)*48*18*(int64_t((T-15)/6)+1), freeData, NULL);
	TF_Tensor* pOutputTensor = NULL;

	TF_Output inps[] = {{rnnModel.inpDict["X_conv_input"], 0}, {rnnModel.inpDict["T"], 0}};
	TF_Output outs[] = {{rnnModel.outDict["softmax"], 0}};
	TF_Tensor* pRNNInpTensors[] = {pInpTensor, pInpSizeTensor};

	TF_SessionRun(rnnModel.pSess,
		 	NULL,
			inps, pRNNInpTensors, 2,
			outs, &pOutputTensor, 1,
			NULL, 0, NULL, TFStatusSingleton::instance().getStatus());
	if (TF_OK != TF_GetCode(TFStatusSingleton::instance().getStatus())){
		fprintf(stderr, "\nERROR: Failed to run RNN model - %s", TF_Message(TFStatusSingleton::instance().getStatus()));
	}

	TF_DeleteTensor(pInpTensor);
	TF_DeleteTensor(pInpSizeTensor);
	return pOutputTensor;
}


void closeTF(){
	TF_CloseSession(cnnModel.pSess, TFStatusSingleton::instance().getStatus());
	TF_CloseSession(rnnModel.pSess, TFStatusSingleton::instance().getStatus());
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
