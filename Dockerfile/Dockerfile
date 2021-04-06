FROM continuumio/anaconda3:2019.03

RUN conda install -c anaconda tensorflow==1.13.1 keras scikit-learn pandas scipy numpy matplotlib scikit-image tqdm numba pyyaml=3.13 h5py cython pip && \
		pip install git+https://github.com/FRBs/sigpyproc3 && \
    git clone https://github.com/shiningsurya/fetch.git && \
    cd fetch && \
    python setup.py install && \
