''' How to Install

You can build locally with setuputils/distutils:

    pip install -r requirements.txt
    python setup.py install

Or you can grab the library online:

    pip install expm

To run the test suite any of the following will work:

    python setup.py test
    nosetests test/
    pytest test/
    make test
'''
# non-controversial imports
from setuptools import setup, find_packages
from setuptools.command.install import install
from glob import glob
from os import name as os_name
from sys import argv, platform
from expm import __version__

# Ideally, you install Cython *before* you install expm, so you can compile it.
try:
    from distutils.extension import Extension
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    cmdclass={'install': install, 'build_ext': build_ext}
    has_cython = True
except:
    print('WARNING: You do not have Cython installed. Installation preceeding without Cython.')
    cmdclass={'install': install}
    has_cython = False
# You will also need NumPy to compile this Cython
try:
    import numpy as np
except:
    print('WARNING: You do not have NumPy installed. Installation preceeding without NumPy.')
    has_cython = False

# find all the extension modules in the project, for a Cython build
if has_cython:
    FORCE_FLAGS = ['-f', '--f', '--force']
    FORCE_REBUILD = True if any([f in argv for f in FORCE_FLAGS]) else False
    IS_WINDOWS = True if (os_name.lower() == 'nt' or 'win' in platform.lower()) else False
    COMP_DIRS = {'language_level': 3, 'boundscheck': False, 'initializedcheck': False, 'cdivision': True}
    sep = '\\' if IS_WINDOWS else '/'
    ext_modules = [Extension(p[:-4].replace(sep, '.'), [p, p[:-2] + 'y'])
                   for p in glob(sep.join(['expm', '*.pxd']))]
    ext_modules += [p for p in glob(sep.join(['expm', '*.pyx']))]
    ext_modules_list = cythonize(ext_modules, annotate=False, force=FORCE_REBUILD, compiler_directives=COMP_DIRS)
else:
    ext_modules_list = []


# perform the actual build/install
setup(
    cmdclass=cmdclass,
    name='expm',
    version=__version__,
    description='Faster matrix exponentials in Python',
    url='https://github.com/theJollySin/expm',
    keywords = "mathematics algorithms algorithm",
    author='John Stilley',
    classifiers=['Development Status :: 3 - Alpha',
                 'Topic :: Software Development :: Libraries :: Python Modules',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Natural Language :: English'],
    python_requires='>=3.5, <3.8',
    license='GPLv3',
    long_description='Trying to improve performance for matrix exponentials in Python.',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'expm': ['*.pxd']},
    ext_modules=ext_modules_list,
    platforms='any',
    test_suite="test",
    setup_requires=["numpy>=1.13.1,<=1.16.4"],
    install_requires=["cython>=0.27.0,<=0.29.16",
                      "numpy>=1.13.1,<=1.16.4"],
    include_dirs=[np.get_include()],
    zip_safe=False)

