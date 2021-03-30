from setuptools import setup
from Cython.Build import cythonize

import numpy as np

setup(
    name='fetch',
    version='0.1.8',
    packages=['fetch'],
    scripts=['bin/predict.py', 'bin/train.py', 'bin/candmaker.py'],
    package_dir={'fetch':'fetch'},
    package_data={'fetch':['models/model_list.csv', 'models/*/*yaml']},
    ext_modules=cythonize([
        "fetch/incoherent.pyx",
        "fetch/fdmt.pyx",
        ]),
    include_dirs=[np.get_include()],
    url='https://github.com/devanshkv/fetch',
    license='GNU General Public License v3.0',
    author=['Devansh Agarwal', 'Kshitij Aggarwal', 'Suryarao Bethapudi'],
    author_email=['devansh.kv@gmail.com', 'ka0064@mix.wvu.edu', 'shining.surya.d8@gmail.com'],
    description='FETCH (Fast Extragalactic Transient Candidate Hunter)',
    classifiers=[
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Scientific/Engineering :: Astronomy']
)

