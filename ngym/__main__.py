from .ng_utils import *


def main():
    config_path = Path(os.getcwd()+"/config.cfg")
    try:
        CONFIG = load_config(config_path)
        if not CONFIG: raise Exception('empty config.cfg')
    except Exception as r:
        CONFIG=load_config_from_str(default_spacy_config)
        remove_textcat(CONFIG)
        save_trainconfig(CONFIG)

    #https://stackoverflow.com/questions/32672596/pyinstaller-loads-script-multiple-times/32677108#32677108
    #https://github.com/pyinstaller/pyinstaller/issues/1921    
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    w = MyForm(CONFIG)
    w.show()
    sys.exit(app.exec_())
    
if __name__=="__main__":  
    main()

