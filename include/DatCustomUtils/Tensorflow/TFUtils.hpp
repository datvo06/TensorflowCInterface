#ifndef __DAT_CUSTOM_TF_UTILS_HPP__
#define __DAT_CUSTOM_TF_UTILS_HPP__
#include <tensorflow/c/c_api.h>
#include <string>
namespace DatCustom{
	namespace Tensorflow{
		void printTFOpParam(TF_Graph* pGraph, std::string name,
				TF_Operation* pOp);
	}
}
#endif
