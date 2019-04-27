## NeuralGym
python app for training spaCy models
#### Installation
**Option 1**: No installation required.
Download and unpack(extract) [**NGymSFX.exe**](https://github.com/d5555/NeuralGym/raw/master/NGymSFX.exe)<br/>
Launch shortcut Ng_start <br/>
**Option 2**: Download **NGym** folder with python files. Run **ng.pyw** (You will need pyqt5 and spaCy to be installed on your PC). In this mode you can use spacy.prefer_gpu() option.<br/>
#### How to use
Create an output directory where the trained model will be saved.<br/>
Select training data file. Training data should be in spaCy data format. You can use [**TagEditor**](https://github.com/d5555/TagEditor) to create your training data. See axample of training data in file [**train_data.txt**](train_data.txt)<br/>
Select a model to train (it can be any spaCy model - **must be compatible with spaCy 2.1.3**) or create blank model. Labels in the training data should match labels in the model otherwise start from blank model.<br/>
Press **Start**. You can disrupt training process at any time by clicking **stop** or **stop and save**. 
![alt text](https://github.com/d5555/NeuralGym/blob/master/NGym.png)
