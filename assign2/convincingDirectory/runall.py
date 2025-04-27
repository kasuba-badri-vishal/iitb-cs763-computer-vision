import subprocess
import os

files_dir = './../data/captured/'
results_dir = './../results/faceRecognition/captured/'


for file in os.listdir(files_dir):
    try:
        if('masked') in file:
            subprocess.call(['python','face-recognition.py','-i',files_dir+file,'-o',results_dir+file,'--type','2'])
        else:
            val = int(file[:-4])
            subprocess.call(['python','face-recognition.py','-i',files_dir+file,'-o',results_dir+file,'--type','2'])
    except:
        continue