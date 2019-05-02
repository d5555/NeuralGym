## NeuralGym
python app for training spaCy models
#### Installation
**Option 1**: No installation required.
Download and unpack(extract) [**NGymSFX.exe**](https://github.com/d5555/NeuralGym/raw/master/NGymSFX.exe)<br/>
Launch ng.exe <br/>
**Option 2**: Download **NGym** folder with python files. Run **ng.pyw** (You will need pyqt5, spaCy and matpotlib to be installed on your PC). In this mode you can use spacy.prefer_gpu() option and it has an extra feature 'chart'.<br/>
#### How to use
1. Create an output directory where the trained model will be saved.<br/>
2. Select a training data file. Training data should be in spaCy data format. You can use [**TagEditor**](https://github.com/d5555/TagEditor) to create your training data. See axample of training data in file [**train_data.txt**](train_data.txt). Make sure your training data is utf-8 encoded otherwise spaCy may raise an error.<br/> 
3. Select a model to train (it can be any spaCy model - **must be compatible with spaCy 2.1.3**) or create a blank model. Labels in the training data should match labels in the original model otherwise start from blank model.<br/>
4. Check on **Parameter averaging** to save the model with parameter averaging.
5. Press **Start**. You can disrupt training process at any time by clicking **stop** or **stop and save**. 

![alt text](https://github.com/d5555/NeuralGym/blob/master/NGym.png)
![alt text](https://github.com/d5555/NeuralGym/blob/master/NGymChart.png)

#### If you want to contribute to a project and make it better, your help is very welcome.
