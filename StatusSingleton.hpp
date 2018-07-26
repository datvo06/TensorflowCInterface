#ifndef __TF_STATUS_SINGLETON_HPP__
#define __TF_STATUS_SINGLETON_HPP__
#include <tensorflow/c/c_api.h>
class TFStatusSingleton{
	public:
		static TFStatusSingleton& instance();
		virtual ~TFStatusSingleton();
		TF_Status* getStatus() const;
	private:
		TF_Status* pStatus;
	protected:
		explicit TFStatusSingleton();
		TFStatusSingleton& operator=(const TFStatusSingleton& rhs);
};
#endif
