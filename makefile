all:
	g++ -fPIC -shared -ltensorflow -std=c++11 test_load_chien_de_mer_model.cpp StatusSingleton.cpp -o libload_current_model.so
