import setuptools

#with open("README.md", "r") as fh:
    #long_description = fh.read()


requirements = ["spacy-lookups-data>=1.0.0", 'PyQt5', "spacy>=3.0.0", 'matplotlib']

setuptools.setup(
    name="ngym",                    
    version="1.1",                    
    author="d5555",                   
    author_email="gitprojects5@gmail.com",
    description="Tool for training spaCy pipeline",
    url='https://github.com/d5555/NeuralGym',
    packages=setuptools.find_packages(),    
    package_data={'ngym': ['images/*.png', 'images/*.ico', '*.pyd']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                     
    python_requires='>=3.7',              
    install_requires=requirements  ,      
    entry_points={ 'console_scripts':['ngym = ngym.__main__:main'] }
)
