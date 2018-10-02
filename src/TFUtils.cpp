#include <DatCustomUtils/Tensorflow/TFUtils.hpp>
#include <DatCustomUtils/Tensorflow/StatusSingleton.hpp>
#include <vector>

void DatCustom::Tensorflow::printTFOpParam(TF_Graph* pGraph, std::string name, TF_Operation* pOp){
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


std::vector<int> DatCustom::Tensorflow::getTFTensorDim(TF_Tensor* pTensor){
	std::vector<int> dims;
	for(int i = 0; i < TF_NumDims(pTensor); i++){
		dims.push_back(TF_Dim(pTensor, i));
	}
	return dims;
}


void DatCustom::Tensorflow::printTFTensor(TF_Tensor* pTensor){
	recursivePrint(pTensor, 0, std::vector<int>());
}
