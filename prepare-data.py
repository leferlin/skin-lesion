import os
import csv
import shutil

n_samples = 900
n_class = 2
validation_ratio = 0.2
path_project = "/content"
path_training_samples = path_project+"/ISBI2016_ISIC_Part3_Training_Data"
#path_validation_samples = path_project+"/ISIC-2017_Validation_Data"
path_data = path_project+"/data"
groundTruth = 'ISBI2016_ISIC_Part3_Training_GroundTruth.csv'


os.makedirs("data")
os.chdir(path_data)
os.makedirs("train")
os.makedirs("validation")
os.makedirs("test")

os.chdir("train")
os.makedirs("benign")
os.makedirs("malignant")

os.chdir("..")

os.chdir("validation")
os.makedirs("benign")
os.makedirs("malignant")

os.chdir(path_project)

# open description file
with open(groundTruth, 'rt', encoding="utf8") as f:
    reader = csv.reader(f)
    imageInfo = list(reader)

filesList = list(os.listdir(path_training_samples))

i = 0
for fileName in filesList:
    if fileName.split('.')[1] == "png":
        continue
    for info in imageInfo:
        infoName = info[0]
        print (fileName)
        if fileName == infoName+".jpg":
            imgClass = info[1]
    if i < validation_ratio*n_samples:
        source = path_training_samples+"/"+fileName
        destiny = path_data+"/validation/"+imgClass+"/"+fileName
        shutil.copy(source, destiny)
    else:
        source = path_training_samples+"/"+fileName
        destiny = path_data+"/train/"+imgClass+"/"+fileName
        shutil.copy(source, destiny)
    i += 1


# end
