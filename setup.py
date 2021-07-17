try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy
import glob
import os


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

#### setup chamferdist
package_name = 'chamferdist'
version = '0.3.0'
requirements = [
    'Cython',
    'torch>1.1.0',
]
long_description = 'A pytorch module to compute Chamfer distance \
                    between two point sets (pointclouds).'

setup(
    name='chamferdist',
    version=version,
    description='Pytorch Chamfer distance',
    long_description=long_description,
    requirements=requirements,
    ext_modules=[
        CUDAExtension('chamferdistcuda', [
            'chamferdist/chamfer_cuda.cpp',
            'chamferdist/chamfer.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


### build pointnet2
try:
    import builtins
except:
    import __builtin__ as builtins

builtins.__POINTNET2_SETUP__ = True
import pointnet2

_ext_src_root = "pointnet2/_ext-src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

requirements = ["etw_pytorch_utils==1.1.1", "h5py", "pprint", "enum34", "future"]

setup(
    name="pointnet2",
    version=pointnet2.__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

