#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include <string>
#include <string.h>
#include <stack>
#include <DatCustomUtils/Tensorflow/StatusSingleton.hpp>
#include <DatCustomUtils/Tensorflow/TFModelUnit.hpp>
#include <memory>

const bool True = true;
const bool False = false;


static std::shared_ptr<DatCustom::Tensorflow::TFModelUnit> pAllModel;
static std::shared_ptr<DatCustom::Tensorflow::TFModelUnit> pCnnModel;
static std::shared_ptr<DatCustom::Tensorflow::TFModelUnit> pRnnModel;

static bool isInitialized = false;

static void freeBuffer(void* data, size_t ) {
        free(data);
}


/** 
 * @brief init initialize for whole lib file
 */
bool initTF(const char* cnnFilePath, const char* rnnFilePath, const char* allFilePath){
	int i = 0;
	printf("\n%d: Initializing Tensorflow module\n", i++);
	if (isInitialized) return true;
	if (allFilePath != NULL){
		// Note: Vector initialization with array only happens since c++11
		// RAII
		pAllModel = std::shared_ptr<DatCustom::Tensorflow::TFModelUnit>(
				new DatCustom::Tensorflow::TFModelUnit(allFilePath, {"X", "T"}, {"softmax"}));
		printf("\n%d: Initialized ALL Model\n", i++);
	}
	pCnnModel = std::shared_ptr<DatCustom::Tensorflow::TFModelUnit> (
			new DatCustom::Tensorflow::TFModelUnit(cnnFilePath, {"X"}, {"X_conv_relu"}));
	printf("\n%d: Initialized CNN Model\n", i++);
	pRnnModel = std::shared_ptr<DatCustom::Tensorflow::TFModelUnit> (
			new DatCustom::Tensorflow::TFModelUnit(rnnFilePath, {"X_conv_input", "T"}, {"softmax"}));
	printf("\n%d: Initialized RNN Model\n", i++);
	isInitialized = True;
	return isInitialized;
}


TF_Tensor* predictTF(float* inpData, int32_t inpSize){
	return pAllModel->run({{"X", (void*)inpData}, {"T", (void*)&inpSize}},
			{{"X", {1, inpSize, 39, 1}}, {"T", {1}}}, 
			{{"X", sizeof(float)*inpSize*39}, {"T", sizeof(int32_t)}},
			{{"X", TF_FLOAT}, {"T", TF_INT32}},
			{"softmax"});
}


TF_Tensor* predictTFCNN(float* inpData, int32_t inpSize){
	return pCnnModel->run({{"X", (void*)inpData}}, {{"X", {1, inpSize, 39, 1}}},
			{{"X", sizeof(float)*inpSize*39}}, {{"X", TF_FLOAT}}, {"X_conv_relu"});
}


TF_Tensor* predictTFRNN(float* inpData, int32_t T){
	return pRnnModel->run({{"X_conv_input", (void*)inpData}, {"T", (void*)&T}},
			{{"X_conv_input", {1, int64_t((T-15)/6)+1, 48*18}}, {"T", {1}}},
			{{"X_conv_input", sizeof(float)*48*18*(int64_t((T-15)/6)+1)}, {"T", sizeof(int32_t)}},
			{{"X_conv_input", TF_FLOAT}, {"T", TF_INT32}}, {"softmax"});
}


void closeTF(){
	TF_CloseSession(pCnnModel->pSess, TFStatusSingleton::instance().getStatus());
	TF_CloseSession(pRnnModel->pSess, TFStatusSingleton::instance().getStatus());
}
