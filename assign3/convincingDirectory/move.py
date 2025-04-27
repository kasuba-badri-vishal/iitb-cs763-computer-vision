import os
import shutil

folders = './../data/masked/give-to-student-anon/'
count=0
for folder in os.listdir(folders):
    files = folders+folder+'/'
    for file in os.listdir(files):
        count+=1
        if(count<10):
            shutil.copy2(files+file, './../data/masked/images/0'+str(count)+'.jpg')
        else:
            shutil.copy2(files+file, './../data/masked/images/'+str(count)+'.jpg')