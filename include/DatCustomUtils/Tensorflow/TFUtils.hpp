#ifndef __DAT_CUSTOM_TF_UTILS_HPP__
#define __DAT_CUSTOM_TF_UTILS_HPP__
#include <tensorflow/c/c_api.h>
#include <string>
#include <vector>
namespace DatCustom{
	namespace Tensorflow{
		/**
		 * @brief Print tensorflow operation informations
		 */
		void printTFOpParam(TF_Graph* pGraph, std::string name,
				TF_Operation* pOp);
		/**
		 * @brief Print a TF_Tensor data
		 */
		void printTFTensor(TF_Tensor* pTensor);
		/**
		 * @brief Get TF_Tensor dimmensions
		 */
		std::vector<int> getTFTensorDim(TF_Tensor* pTensor);
	}
}
#endif
