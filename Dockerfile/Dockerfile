FROM continuumio/anaconda3:2019.03

LABEL "maintainer"="shiningsurya"

RUN conda install --override-channels -c conda-forge gcc_linux-64 pep517

RUN conda install --override-channels -c anaconda numpy pandas scipy matplotlib scikit-learn scikit-image tqdm pyyaml=3.13 h5py cython pip && \
	conda install --override-channels -c anaconda tensorflow==1.13.1 keras

RUN apt-get update && apt-get -y install gcc g++ gfortran libboost-all-dev libfftw3-dev libopenblas-dev libopenmpi-dev vim git parallel rsync 

RUN git clone https://github.com/FRBs/sigpyproc3.git && cd sigpyproc3 && python -m pip install .

RUN git clone https://github.com/shiningsurya/fetch.git && cd fetch && python setup.py install

RUN get_all_weights.py -v -m a && \
	get_all_weights.py -v -m b && \
	get_all_weights.py -v -m c && \
	get_all_weights.py -v -m d && \
	get_all_weights.py -v -m e && \
	get_all_weights.py -v -m f && \
	get_all_weights.py -v -m g && \
	get_all_weights.py -v -m h && \
	get_all_weights.py -v -m i && \
	get_all_weights.py -v -m j && \
	get_all_weights.py -v -m k

ENV HDF5_USE_FILE_LOCKING=FALSE
