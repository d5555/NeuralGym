## NeuralGym v1.1
python app for training spaCy models<br/>
Check out [**TagEditor**](https://github.com/d5555/TagEditor) for creating training data
#### Installation
**Option 1**: No installation required.
Download zip archive files [**NGym.7z.001**](https://github.com/d5555/NeuralGym/raw/master/NGym.7z.001) and  [**NGym.7z.002**](https://github.com/d5555/NeuralGym/raw/master/NGym.7z.002) into the same folder, and unzip **NGym.7z.001**.<br/>
Launch ng.exe 

**Option 2**: 
```
git clone https://github.com/d5555/NeuralGym
pip install neuralgym/.
```
To run application: `python -m ngym` or 
```
python
import ngym
```

#### How to use
1. Create an output directory where the trained model will be saved.<br/>
2. Select `train` and `dev` data files in spaCy format. You can use [**TagEditor**](https://github.com/d5555/TagEditor) to create your training dataset. For demonstration purposes there are 2 dataset files, `imdb_train.spacy` (400 docs) and `imdb_dev.spacy` (100 docs) annotated with POS ,Dependencies, NER and Textcategories.
3. Select a source model (it can be any spaCy model compatible with spaCy 3.0+) for training from source. You can specify either a source model name, eg en_core_web_sm or select a folder with model. If you specify the model name without full path, the model should be placed into the application's main folder (including model's dist-info folder) or add path to the Python folder where spaCy models are installed by pushing button `Add sys path`. Usually it is Python...\Lib\site-package. For example ... "C:\Python39\Lib\site-packages" <br/> To train from source check on Training options `From source` respectively or uncheck them to start from blank model. <br/>Labels in the training data should match labels in the original model otherwise start from blank model. 
4. Check on **Use averages** so the model to be saved with parameter averaging after training is done.
5. Press **Start** to initialize training. You can disrupt training process at any time by clicking **stop**. 
6. After training is completed there will be 2 folders in the output directory, **'Best model'** and **'Last model'**.
7. Button **Reset** allows to restore default settings in case of an error. Or delete 'config.cfg' in the main folder.

<img src="https://github.com/d5555/NeuralGym/blob/master/NGym.png" width="550" >

![alt text](https://github.com/d5555/NeuralGym/blob/master/NGymChart.png)


#### *If you want to contribute to a project and make it better, your help is very welcome.<br>**gitprojects5@gmail.com**
