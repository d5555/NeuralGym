## NeuralGym
python app for training spaCy models<br/>
Check out [**TagEditor**](https://github.com/d5555/TagEditor) for creating training data
#### Installation
<!---**Option 1**: No installation required.--->
Download and unpack zip archive [**NGym.7z.001**](https://github.com/d5555/NeuralGym/raw/master/NGym.7z.001) and  [**NGym.7z.002**](https://github.com/d5555/NeuralGym/raw/master/NGym.7z.002).<br/>
Launch ng.exe <br/>
<!---**Option 2**: Download **NGym** folder with python files. Run **ng.pyw** (You will need pyqt5, spaCy and matpotlib to be installed on your PC). In this mode you can use spacy.prefer_gpu() option.<br/>--->
#### How to use
1. Create an output directory where the trained model will be saved.<br/>
2. Select a training data file. Training data should be in spaCy data format. You can use [**TagEditor**](https://github.com/d5555/TagEditor) to create your training data. 
3. Select a model to train (it can be any spaCy model compatible with spaCy 3.0+) or create a blank model. Labels in the training data should match labels in the original model otherwise start from blank model.<br/>
4. Check on **Parameter averaging** so the model to be saved with parameter averaging after training is done.
5. Press **Start**. You can disrupt training process at any time by clicking **stop**. 

![alt text](https://github.com/d5555/NeuralGym/blob/master/NGym.png)

![alt text](https://github.com/d5555/NeuralGym/blob/master/NGymChart.png)


#### *If you want to contribute to a project and make it better, your help is very welcome.
