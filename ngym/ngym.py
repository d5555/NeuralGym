import setuptools

#with open("README.md", "r") as fh:
    #long_description = fh.read()


requirements = ["spacy-lookups-data>=1.0.0", 'PyQt5', "spacy>=3.0.0", 'matplotlib']#["spacy>=3.0.0"]

setuptools.setup(
    name="ngym",                     # This is the name of the package
    version="1.1",                        # The initial release version
    author="d5555",                     # Full name of the author
    author_email="gitprojects5@gmail.com",
    description="Tool for training spaCy pipeline",
    #long_description=long_description,      # Long description read from the the readme file
    #long_description_content_type="text/markdown",
    url='https://github.com/d5555/NeuralGym',
    packages=setuptools.find_packages(),    # List of all python modules to be installed #include=('images*',)
    #data_files=[('./images', ['images/expand.png'])],
    package_data={'ngym': ['images/*.png', 'images/*.ico', '*.pyd']},
    #setup_requires=['images'],
    #package_dir={'NGym': 'NGym/images'},
    #include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.7',                # Minimum version requirement of the package
    #py_modules=["NGym"],             # Name of the python package
    #package_dir={'':'NGym', 'NGym': 'NGym/images'},     # Directory of the source code of the package
    install_requires=requirements  ,                 # Install other dependencies if any
    entry_points={ 'console_scripts':['ngym = ngym.__main__:main'] }
)
#https://towardsdatascience.com/how-to-publish-a-python-package-to-pypi-7be9dd5d6dcd


#E:\my_env\Scripts\activate.bat
#python setup.py sdist
#http://transit.iut2.upmf-grenoble.fr/doc/python-setuptools-doc/html/setuptools.html
#http://transit.iut2.upmf-grenoble.fr/doc/python-setuptools-doc/html/setuptools.html#including-data-files

#(my_env) E:\NeuralGym>
#python setup.py sdist bdist_wheel
#python -m ngym