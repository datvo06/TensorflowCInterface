#include <DatCustomUtils/Tensorflow/TFModelUnit.hpp>
#include <DatCustomUtils/Tensorflow/StatusSingleton.hpp>
#include <DatCustomUtils/FileUtils/FileUtils.hpp>
#include <DatCustomUtils/Tensorflow/TFUtils.hpp>


static void freeBuffer(void* data, size_t ) {
        free(data);
}


static void freeT(void*, size_t, void*){
}


static TF_Buffer* readFile(const char* filePath){
	size_t fSize;
	void* data = DatCustom::FileUtils::readWholeFileAtOnce(filePath, &fSize);
	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fSize;
	buf->data_deallocator = freeBuffer;
	return buf;
}

static bool initModel(const char* filePath, DatCustom::Tensorflow::TFModelUnit* pModelUnit, const std::vector<std::string>& inputNames, const std::vector<std::string>& outputNames){
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
		return false;
	}

	pModelUnit->pSess = TF_NewSession(pModelUnit->pGraph, pModelUnit->pSessOpts, pStatus);

	if(TF_GetCode(pStatus) != TF_OK){
		// print some errors here...
		fprintf(stderr, "ERROR: Unable to create session %s", TF_Message(pStatus));
		return false;
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
			return false;
		}
		DatCustom::Tensorflow::printTFOpParam(pModelUnit->pGraph, i->first, i->second);
	}
	printf("\n- Outputs: ");
	for (auto i = pModelUnit->outDict.begin(); i != pModelUnit->outDict.end(); i++){
		if (i->second == NULL) {
			fprintf(stderr, "ERROR: Unable to load operations");
			return false;
		}
		DatCustom::Tensorflow::printTFOpParam(pModelUnit->pGraph, i->first, i->second);
	}

	return true;
}



DatCustom::Tensorflow::TFModelUnit::TFModelUnit(const char* filePath, const std::vector<std::string>& inputNames,
		const std::vector<std::string>& outputNames){
	printf("Inside TFModelUnit constructor - Here\n");
	initModel(filePath, this, inputNames, outputNames);
}


TF_Tensor* DatCustom::Tensorflow::TFModelUnit::run(std::map<std::string, void*> inpDataDict,
		std::map<std::string, std::vector<int64_t>> inpDimsDict,
		std::map<std::string, size_t> inpDataSizeInByteDict,
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
		for (auto dimSize: inpDimsDict[(*keyIt)]){
			bufferSize *= dimSize;
		}
		// Use freeT instead so tensorflow will not free our data, we manage it our self
		pAllInpTensors[keyIt-keyVecs.begin()] = TF_NewTensor(
				(TF_DataType)inpTypeDict[(*keyIt)], (const int64_t*)inpDimsDict[(*keyIt)].data(),
			 	(int)inpDimsDict[(*keyIt)].size(),
			 	(void*)inpDataDict[(*keyIt)], (size_t)inpDataSizeInByteDict[(*keyIt)],
			 	freeT, NULL);
		inps[keyIt-keyVecs.begin()] = {this->inpDict[(*keyIt)], 0};
	}
	for (auto strIt = outputOpsString.begin(); strIt != outputOpsString.end(); strIt++){
		outs[strIt-outputOpsString.begin()] = {this->outDict[*strIt], 0};
	}
	TF_SessionRun(this->pSess,
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
	free(pAllInpTensors);
	free(inps);
	free(outs);
	return pOutputTensor;
}


DatCustom::Tensorflow::TFModelUnit::~TFModelUnit(){
	//TF_DeleteGraph(this->pGraph);
	//TF_CloseSession(this->pSess, TFStatusSingleton::instance().getStatus());
}
