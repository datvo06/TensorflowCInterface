#ifndef __TF_C_API_WRAPPER_HPP__
#define __TF_C_API_WRAPPER_HPP__
#include <tensorflow/c/c_api.h>
#include <vector>
#include <map>
typedef struct TFModelUnit{
	TF_SessionOptions* pSessOpts;
	TF_Graph* pGraph;
	TF_Buffer* pRunOptsBuff;
	TF_Buffer* pMetaGraphBuff; 
	TF_Session* pSess;
	std::map<std::string, TF_Operation*> inpDict;
	std::map<std::string, TF_Operation*> outDict;
	TFModelUnit(){
		pSessOpts = NULL;
		pGraph = NULL;
		pRunOptsBuff = NULL;
		pMetaGraphBuff = NULL;
		pSess = NULL;
		inpDict =  	std::map<std::string, TF_Operation*>();
		outDict =  	std::map<std::string, TF_Operation*>();
	}
} TFModelUnit;
bool initModel(const char* filePath, TFModelUnit* pModelUnit, const std::vector<std::string>& inputNames, const std::vector<std::string>& outputNames);

#endif
