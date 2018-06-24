TF_TYPE="cpu"
OS="linux"
TARGET_DIRECTORY="/usr/local"
 curl -L \
	    "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.8.0.tar.gz" |
    sudo tar -C $TARGET_DIRECTORY -xz
	
sudo ldconfig
