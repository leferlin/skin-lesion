from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import csv
from sklearn.metrics import average_precision_score

# dimensions of our images.
img_width, img_height = 150, 150

test_data_dir = 'test'

results = 'results.csv'
correctPredicts = 0
malignant = 0
truePositive = 0
benign = 0
trueNegative = 0

# load model
model = load_model('bottleneck_model.h5')

# data generator
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='binary',
    shuffle=False)

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator,steps = nb_samples)

with open(results, 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in range(0, nb_samples):
        if predict[i] > 0.5:
            wr.writerow([filenames[i], 'malignant', predict[i]])
        else:
            wr.writerow([filenames[i], 'benign', predict[i]])


# view results
with open('results.csv', 'r') as f:
    readerResults = csv.reader(f)

    with open('ISBI2016_ISIC_Part3_Test_GroundTruth.csv', 'r') as f2:
        readerGroundTruth = csv.reader(f2)

        # read file row by row
        for rowResults in readerResults:
            nameResults = rowResults[0].split('/')[1].split('.')[0]
            classResults = rowResults[1]
            predictResults = rowResults[2]
            if classResults == 'benign':
                classResults = 0
            else:
                classResults = 1

            for rowGroundTruth in readerGroundTruth:
                nameGroundTruth = rowGroundTruth[0]
                classGroundTruth = int(rowGroundTruth[1].split('.')[0])

                #print (nameResults+' '+nameGroundTruth)
                if nameResults == nameGroundTruth:
                    if classResults == classGroundTruth:
                        correctPredicts += 1
                    if classGroundTruth == 0:
                        benign += 1
                        if classResults == 0:
                            trueNegative += 1
                    if classGroundTruth == 1:
                        malignant += 1
                        if classResults == 1:
                            truePositive += 1

                    print(nameResults+' '+str(classResults)+' '+str(classGroundTruth)+' '+str(predictResults))
                    break

predictions_scores = []
for [cell] in predict:
    predictions_scores.append(cell)

predictions_true = []
with open('ISBI2016_ISIC_Part3_Test_GroundTruth.csv', 'r') as f2:
    readerGroundTruth = csv.reader(f2)
    for row in readerGroundTruth:
        predictions_true.append(float(row[1].split('.')[0]))

print (predictions_scores)
print (predictions_true)

test_acc = correctPredicts/nb_samples
test_sens = truePositive/malignant
test_spec = trueNegative/benign
average_precision = average_precision_score(predictions_true, predictions_scores)
print ("Test accuracy: "+str(test_acc))
print ("Test sensitivity: "+str(test_sens))
print ("Test specificity: "+str(test_spec))
print ("Score: "+str(average_precision))
