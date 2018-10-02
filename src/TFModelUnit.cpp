#include <DatCustomUtils/Tensorflow/TFModelUnit.hpp>
#include <DatCustomUtils/Tensorflow/StatusSingleton.hpp>
#include <DatCustomUtils/FileUtils/FileUtils.hpp>



static void freeBuffer(void* data, size_t ) {
        free(data);
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
		printTFOpParam(pModelUnit->pGraph, i->first, i->second);
	}
	printf("\n- Outputs: ");
	for (auto i = pModelUnit->outDict.begin(); i != pModelUnit->outDict.end(); i++){
		if (i->second == NULL) {
			fprintf(stderr, "ERROR: Unable to load operations");
			return false;
		}
		printTFOpParam(pModelUnit->pGraph, i->first, i->second);
	}

	return True;
}



DatCustom::Tensorflow::TFModelUnit::TFModelUnit(const char* filePath, const std::vector<std::string>& inputNames,
		const std::vector<std::string>& outputNames){
	TFModelUnit();
	initModel(filePath, this, inputNames, outputNames);
}
