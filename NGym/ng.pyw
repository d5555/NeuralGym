import os, sys

from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,  QAction, QMessageBox, QFileDialog
from PyQt5 import QtCore  
from PyQt5.QtCore import pyqtSlot  
from form1 import *
from PyQt5.QtGui import QIcon, QDoubleValidator, QIntValidator, QColor
from ast import literal_eval
import configparser
import spacy, random
import threading

import time
from spacy.util import set_data_path, get_model_meta, load_model_from_init_py, load_model_from_path, load_model, get_data_path

from pathlib import Path
from spacy.util import minibatch, compounding
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import codecs

spacy.prefer_gpu()

import matplotlib
matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def try_except(function):
        def wrapper( self,*args, **kwargs):
            try: 
                function(self, *args, **kwargs)
            except Exception as e:
                self.log_.emit('*Error in the training loop occured*', self.red)
                self.log_.emit(str(e), self.red)
                self.disrupt_thread()
        return wrapper

class OutputWrapper(QtCore.QObject):
    outputWritten = QtCore.pyqtSignal(object)#, object)

    def __init__(self, parent, stdout=True):
        QtCore.QObject.__init__(self, parent)
        if stdout:
            self._stream = sys.stdout
            sys.stdout = self
        else:
            self._stream = sys.stderr
            sys.stderr = self
        self._stdout = stdout

    def write(self, text):
        #self._stream.write(text)  -  output in console window
        self.outputWritten.emit(text) 

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __del__(self):
        try:
            if self._stdout:
                sys.stdout = self._stream
            else:
                sys.stderr = self._stream
        except AttributeError:
            pass

def save_trainconfig():
    with open('train.ini', 'w') as configfile:
        config.write(configfile)



def evaluate(model, examples):
    scorer = Scorer()
    temp=['words', 'tags' if 'tagger' in model.pipe_names else None , 'deps' if 'parser' in model.pipe_names else None, 
    'heads' if 'parser' in model.pipe_names else None, 'entities' if 'ner' in model.pipe_names else None]

    for input_, annot in examples:
        doc_gold_text =  model.make_doc(input_)
        temp={j:annot.get(j, None) for j in temp if j}
        gold = GoldParse(doc_gold_text,**temp)
        pred_value = model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores
from PyQt5 import QtCore

class MyForm(QDialog):
    progressChanged = QtCore.pyqtSignal(int)
    log_= QtCore.pyqtSignal([str, object])
    red=QtCore.Qt.red
    green= QtCore.Qt.green
    blue= QtCore.Qt.cyan 


    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint| QtCore.Qt.WindowMaximizeButtonHint)
        self.setWindowIcon(QIcon('web.png'))
        self.ui.setupUi(self)
        finish = QAction("Quit", self)
        finish.triggered.connect(self.closeEvent)
        self._err_color = QtCore.Qt.red
        self.stdcolor=QColor('#EEEEEE')
        self.train_config=config['opt']
        #--------------Figure---------
        self.fig = Figure(figsize=(3, 3))
        self.cnv = FigureCanvas(self.fig )
        self.ui.verticalLayout_.addWidget(self.cnv)
        self.ui.pushChart.clicked.connect(self.graph_show) 
        self.ui.pushButton_2.clicked.connect(self.clear_output) 



        self.ui.modname.setText(self.train_config.get('New model name', 'NewModel'))
        self.set_lineedit_validator(self.ui.modname, "[a-zA-Z-0-9_]+")
        start_output_path=self.train_config.get('Output folder', os.getenv("SystemDrive")+'/')
        self.ui.outputpath.setText(start_output_path)
        self.ui.open_output.clicked.connect(lambda:self.browsefunc(self.ui.outputpath, 'Output folder'))
        #---------Training DATA
        self.ui.traindatapath.setText(self.train_config.get('traindatapath', 'data file path...'))
        self.ui.but_traindatapath.clicked.connect(lambda:self.browsefilefunc(self.ui.traindatapath, 'traindatapath'))


        #---------Select Model to train
        self.ui.existmodelpath.setText(self.train_config.get('Model path folder', 'Model path...'))
        self.ui.but_existmodelpath.clicked.connect(lambda:self.browsefunc(self.ui.existmodelpath,  'Model path folder'))
        
        #-------------------Training options
        self.n_iter=     self.train_config.get('n_iter', 100)
        self.learn_rate= self.train_config.get('learn_rate', 0.001)
        self.drop_ind=   self.train_config.get('drop_ind', 0.2)
        self.batch_start=self.train_config.get('batch_start', 4)
        self.batch_stop=self.train_config.get('batch_stop', 32)
        self.batch_fac=self.train_config.get('batch_factor', 1.001)   


        self.set_lineedit_validator(self.ui.eval_entry, r"^(0\.0*[1-9](\d+)?)$" )
        self.ui.iter_entry.setValidator(QIntValidator())
        self.set_lineedit_validator(self.ui.lr_entry, r"^(0\.0*[1-9](\d+)?)$" )
        self.set_lineedit_validator(self.ui.drop_entry, r"^(0\.0*[1-9](\d+)?)$" )
        self.ui.comp_start.setValidator(QIntValidator( ))
        self.ui.comp_stop.setValidator(QIntValidator( ))
        self.ui.comp_fac.setValidator(QDoubleValidator())

        self.ui.iter_entry.setText(self.n_iter)
        self.ui.lr_entry.setText(self.learn_rate)
        self.ui.drop_entry.setText(self.drop_ind)

        self.ui.comp_start.setText(self.batch_start)
        self.ui.comp_stop.setText(self.batch_stop)
        self.ui.comp_fac.setText(self.batch_fac)

        self.ui.start_but.clicked.connect(self.initiate_data)
        self.ui.disrupt_but.clicked.connect(self.disrupt_thread)
        self.ui.disrupt_save_but.clicked.connect(self.disrupt_thread_save)
        self.ui.mbatch.clicked.connect(self.disablebatchframe)

        self.progressChanged.connect(self.ui.progressBar.setValue)
        self.log_.connect(self.log)

        self.output_dir=self.train_config['Output folder']
        self.thread_flag=0
        self.save_model_flag=0

        stdout = OutputWrapper(self, True)
        stdout.outputWritten.connect(self.handleOutput)
        stderr = OutputWrapper(self, False)
        stderr.outputWritten.connect(self.handleErrorOutput)

        self.loss=[]
 
    def graph_show(self):
        if self.ui.pushChart.isChecked():
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_2)
            loss=self.loss
            ax=self.fig.add_subplot(111)
            ax.plot([epoch.get('tagger') for epoch in loss ],  label='tagger')
            ax.plot([epoch.get('parser') for epoch in loss ],  label='parser')
            ax.plot([epoch.get('ner') for epoch in loss ],  label='ner')
            ax.plot([epoch.get('textcat') for epoch in loss ],  label='textcat')
            self.fig.legend()
            self.cnv.draw()
        else:
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_1)
            self.fig.clf() 

    def clear_output(self):
        self.loss=[]
        if self.ui.pushChart.isChecked(): self.ui.pushChart.setChecked(False); self.graph_show()

    def set_lineedit_validator(self, wid, val_str):
        regex=QtCore.QRegExp(val_str)
        validator = QtGui.QRegExpValidator(regex)
        wid.setValidator(validator)

    def handleOutput(self, text):
        self.ui.textfield.moveCursor(QtGui.QTextCursor.End)
        self.ui.textfield.insertPlainText(text)


    def handleErrorOutput(self, text):
        self.ui.textfield.moveCursor(QtGui.QTextCursor.End)
        self.ui.textfield.setTextColor(self._err_color)
        self.ui.textfield.insertPlainText(text)
        self.ui.textfield.setTextColor(self.stdcolor)
        self.ui.textfield.insertPlainText('\n')
        self.disrupt_thread()

    def log(self, txt, col=None):
        if not col: col=self.stdcolor
        self.ui.textfield.moveCursor(QtGui.QTextCursor.End)
        self.ui.textfield.setTextColor(col)
        self.ui.textfield.insertPlainText(txt)

        self.ui.textfield.setTextColor(self.stdcolor)
        self.ui.textfield.insertPlainText('\n')
        self.ui.textfield.moveCursor(QtGui.QTextCursor.End)

    def initiate_data(self):
        try: self.initiate_()
        except Exception as e: 
                self.log_.emit(f'Initializing variables error:\n{e}', self.red)
    def initiate_(self):

        self.loss=[]

        if self.ui.pushChart.isChecked(): self.ui.pushChart.setChecked(False); self.graph_show()
        self.progressChanged.emit(0)
        self.reinit_entries()
        save_trainconfig()
        if self.thread_flag: return

        self.log(f'spaCy{spacy.__version__}', col=QColor(0, 255, 0))

        print('Variables initialization...')
        self.output_dir=self.train_config['Output folder']; print( f'Output_dir:\"{self.output_dir}\"')#+self.train_config['new model name']
        
        if self.ui.model_rb1.isChecked()==True: 
            model=self.train_config['Model path folder']#self.existmodelpath['text']
            try:
                set_data_path(model)
                print('Loading model...')
                
                nlp = spacy.load(model)
                print(f'Model has been loaded: \"{model}\"')
            except Exception as e: 
                self.log(str(e), col=self.red)
                print('Trying to load from __init__.py...')
                try:
                    nlp = load_model_from_init_py(model +'\\__init__.py')
                    self.log(f'Success, Model has been loaded from __init__.py: \"{model +"__init__.py"}\"', col=self.green)
                except Exception as e:
                    self.log(str(e), col=self.red)
                    return
        elif self.ui.model_rb2.isChecked()==True:
            nlp = spacy.blank(self.ui.lang_combo.currentText())
            print(f"Creating blank '{self.ui.lang_combo.currentText()}' model")
        #-------loading  data

        datapath=self.train_config['traindatapath']
        print('Loading training data...')
        if datapath=='data file path...': self.log('missing data file path',self.red); return
        datapath = Path(datapath)
        if not datapath.is_file(): self.log(f'incorrect data file path: {datapath}',self.red); return
        try:
            with codecs.open(datapath, 'r', encoding='utf-8', errors='ignore') as f:
                fileContents = f.read()
                TRAIN_DATA=literal_eval( fileContents)
            print(f'TRAIN_DATA loaded from path: \"{datapath}\"')
            print(f'Number of examples: {len(TRAIN_DATA)}')
        except Exception as e: 
            self.log(f'Failed to load TRAIN_DATA from path \"{datapath}\"\n{e}', self.red)
            return

        if self.ui.eval_ck.isChecked()==True:
            self.eval_ind=self.trynum(self.ui.eval_entry.text(), 0.2, "split evaluation data index", integ=False, limits=(0, 1))
        
        self.n_iter=self.trynum(self.ui.iter_entry.text(), 100, "n_iter", limits=(0, float('inf')))    
        self.learn_rate=self.trynum(self.ui.lr_entry.text(), 0.001, "learn_rate", integ=False, limits=(0, 1) ) 
        self.drop_ind=self.trynum(self.ui.drop_entry.text()  , 0.2, "drop", integ=False, limits=(0, 1)) 
        if self.ui.mbatch.isChecked():
            self.batch_start= self.trynum(self.ui.comp_start.text()  , 4, "batch_start", integ=False) 
            self.batch_stop= self.trynum(self.ui.comp_stop.text()  , 32, "batch_stop", integ=False) 
            self.batch_fac= self.trynum(self.ui.comp_fac.text()  , 1.001, "batch_compound", integ=False) 
        #global nlp
        check_pipe = [self.ui.Tagger.isChecked(), self.ui.Parser.isChecked(), self.ui.Ner.isChecked(), self.ui.Cat.isChecked()]
        componets = ['tagger' , 'parser' ,'ner', 'textcat']
        pipeline = {componets[i]:set()  for i in range (4) if check_pipe[i]}
        if not pipeline: raise Exception (f"No pipeline components were selected")
        print('Selected pipeline components:', list(pipeline) )

        for _, annotations in TRAIN_DATA:
            if 'tagger' in pipeline: pipeline['tagger'] |= set(annotations.get('tags', [])); 
            if 'parser' in pipeline: pipeline['parser'] |= set(annotations.get('deps',[]))
            if 'ner'    in pipeline: pipeline['ner']    |= set( [j[2] for j in annotations.get('entities',[]) ]) 
            if 'textcat'in pipeline: pipeline['textcat']|= set( [j for j in annotations.get('cats',dict()).keys()])
        
        for name in pipeline.keys():
            if not pipeline[name]: 
                self.log(f"No \'{name}\' labels found in training data.", self.red) 
                if name =='textcat' and 'textcat' not in nlp.pipe_names: 
                    self.log(f"'textcat' disabled.", self.red)
                    continue
            if name in nlp.pipe_names: component = nlp.get_pipe(name)#nlp.create_pipe(name)# 
            else:
                component = nlp.create_pipe(name)
                nlp.add_pipe(component)

            for lab in pipeline[name]: component.add_label(lab)

        if not set(nlp.pipe_names) & set(pipeline): 
            raise Exception (f"Empty pipeline!")

        if self.ui.mbatch.isChecked(): 
            print('*Training with minibatches*')
            self.t = threading.Thread(target=self.training_with_minibatch, args=[pipeline.keys(), nlp, TRAIN_DATA] )
        else: self.t = threading.Thread(target=self.start_training, args=[pipeline.keys(), nlp, TRAIN_DATA])

        self.save_model_flag=1
        self.thread_flag=1
        self.t.start()

    def disrupt_thread(self):
        self.save_model_flag=0
        self.thread_flag=0
    def disrupt_thread_save(self):
        self.save_model_flag=1
        self.thread_flag=0

    def disablebatchframe(self):
        if self.ui.mbatch.isChecked():
            self.ui.comp_start.setText(self.train_config.get('batch_start', 4))
            self.ui.comp_stop.setText(self.train_config.get('batch_stop', 32))
            self.ui.comp_fac.setText(self.train_config.get('batch_factor', 1.001))

    def trynum(self, numstring, defaul, name, integ=True, limits=None):

        try:
            if integ:
                val=int(numstring)
            else: val=float(numstring)
            if limits is not None:
                if not limits[0]<=val<=limits[1] : 
                    val = defaul
                    self.log(f"{name} = {str(val)} Default value assigned", self.red)
                    return val
            print(name, " = ", val)
        except Exception as e:
            val = defaul
            self.log(e, QColor(255,0,0))
            self.log(f'{name} = {str(val)} Default value assigned', self.red)
        return val

    def reinit_entries(self):
        self.train_config['New model name']=self.ui.modname.text()
        self.train_config['Output folder']=self.ui.outputpath.text()
        self.train_config['traindatapath']=self.ui.traindatapath.text()
        self.train_config['Model path folder']=self.ui.existmodelpath.text()
    def closeEvent(self, event):    
        self.disrupt_thread()
        self.reinit_entries()
        save_trainconfig()
        event.accept()

    def browsefunc(self, variab, conf_var):
        #dirname = askdirectory()#(parent=root, initialdir='/home/', title='dirtext') 
        dirname =QFileDialog.getExistingDirectory(self, conf_var, f'{variab.text()}', QFileDialog.ShowDirsOnly)
        if dirname:
            variab.setText(dirname+'/')
            self.train_config[conf_var]=dirname+'/'
    def browsefilefunc(self, variab, conf_var):
        #data_filename = askopenfilename(filetypes=(('Text files', '*.txt'), ('All files', '*.*')), defaultextension='.txt')
        data_filename =QFileDialog.getOpenFileName(self,"Select training data file:", "","All Files (*)")#"All Files (*);;Text Files (*.txt)"
        if data_filename[0]:
            variab.setText(data_filename[0])
            self.train_config[conf_var]=data_filename[0]
    #----------------------training-------------
    @try_except
    def start_training(self, pipeline, nlp, TRAIN_DATA):
      
        self.log_.emit("Training the model...", self.blue)

        
        optimizer = nlp.begin_training(learn_rate=self.learn_rate)
        other_pipes=set(nlp.pipe_names)-set(pipeline)

        with nlp.disable_pipes(*other_pipes):
            print('pipeline:', nlp.pipe_names)
            print('iter / LOSS')#, 'P', 'R', 'F'))
            persent=100/self.n_iter 
            start_time = time.time()
            for i in range(1,self.n_iter+1):

                    random.shuffle(TRAIN_DATA)
                    losses = {}
                    for text, annotations in TRAIN_DATA:
                        nlp.update([text], [annotations], sgd=optimizer, drop=self.drop_ind, losses=losses)

                    self.progressChanged.emit(i*persent)
                    print(f"{i} Epochs ---{(time.time()-start_time):.3f}seconds ---\n", losses)
                    self.loss.append(losses)
                    if self.ui.eval_ck.isChecked():
                        with nlp.use_params(optimizer.averages):
                            results = evaluate(nlp, TRAIN_DATA)
                            print(results)

                    if not self.thread_flag: 
                        self.log_.emit("-----TRAINING DISRUPTED-----", self.red)
                        break
            else:    
                self.log_.emit("********TRAINING COMPLETED*********", self.blue)
        self.save_model(nlp, optimizer)        
        self.thread_flag=0

    @try_except
    def training_with_minibatch(self, pipeline, nlp, TRAIN_DATA):

        self.log_.emit("Training the model...", self.blue)
        #self.prin('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        optimizer = nlp.begin_training(learn_rate=self.learn_rate)
        other_pipes=set(nlp.pipe_names)-set(pipeline)
        with nlp.disable_pipes(*other_pipes):
            print('pipeline:', nlp.pipe_names)
            print('Epochs / LOSS')#, 'P', 'R', 'F'))
            persent=100/self.n_iter
            start_time = time.time()
            for i in range(1,self.n_iter+1):
                    random.shuffle(TRAIN_DATA)
                    losses = {}
                    batches = minibatch(TRAIN_DATA, size=compounding(self.batch_start, self.batch_stop, self.batch_fac))
                    for batch in batches:
                        texts, annotations = zip(*batch)

                        nlp.update(texts, annotations, sgd=optimizer, drop=self.drop_ind, losses=losses)

                    print(f"{i} Epochs ---{(time.time()-start_time):.3f}seconds ---\n", losses)#, file=buffer)
                    self.loss.append(losses)
                    self.progressChanged.emit(i*persent)
                    if self.ui.eval_ck.isChecked():
                        with nlp.use_params(optimizer.averages):
                            results = evaluate(nlp, TRAIN_DATA)
                            print(results)

                    if not self.thread_flag: 
                        self.log_.emit("-----TRAINING DISRUPTED-----", self.red)
                        #print("-----TRAINING DISRUPTED-----");  
                        break
            else:
                self.log_.emit("********TRAINING COMPLETED*********", self.blue)
        self.save_model(nlp, optimizer)        
        self.thread_flag=0

    def save_model(self, nlp, optimizer):
        if self.save_model_flag:
            nlp.meta['name']=self.train_config['new model name']
            if self.ui.averaging.isChecked():
                with nlp.use_params(optimizer.averages):
                        nlp.to_disk(self.output_dir)
                warning=f'Model saved (with parameter averaging) to: {self.output_dir}'
                self.log_.emit(warning, self.green)

            else: 
                nlp.to_disk(Path(self.output_dir))
                warning=f'Model saved to: {self.output_dir}'
                self.log_.emit(warning, self.green)

if __name__=="__main__":  

    config = configparser.ConfigParser()
    config_path = Path(os.getcwd()+"/train.ini")
    if config_path.is_file():
        config.read(config_path)
    else:
        config['DEFAULT'] = {'New model name': 'NewModel',
                            'Output folder': os.getenv("SystemDrive")+'/',
                            'traindatapath': 'Data File path...',
                            'eval_ind': 0.2,
                            'Model path folder': 'Model path...',
                            'n_iter': 100,
                            'learn_rate': 0.001,
                            'drop_ind': 0.2,
                            'batch_start': 4,
                            'batch_stop': 32,
                            'batch_factor': 1.001
                              }
        config['opt']={}
        save_trainconfig()


    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
