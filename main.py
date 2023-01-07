import argparse
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from sklearn.metrics import confusion_matrix , accuracy_score, classification_report
from utils.generate_csv import generate_csv
from utils.k_fold_splits import k_fold_splits
from utils.augmentation import fill, horizontal_shift, vertical_shift, brightness, zoom, channel_shift, horizontal_flip, vertical_flip, rotation, HSV, YUV 
from utils.data_prep import encode_y, process_x
from utils.create_model import create_model
from utils.plot_loss_curves import plot_loss_curves
from utils.beta_norm_ensemble import beta_norm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default = 30,
                    help='Number of Epochs for training')
parser.add_argument('--path1', type=str, default = 'Original Images',
                    help='Path where the images are stored')
parser.add_argument('--path2', type=str, default = 'csv_files/originalset.csv',
                    help='where the csv containing all the images is stored')
parser.add_argument('--batch_size', type=int, default = 16,
                    help='Batch Size for Mini Batch Training')
parser.add_argument('--lr1', type=float, default = 1e-4,
                    help='Learning rate for training for first 20 epochs')
parser.add_argument('--lr2', type=float, default = 1e-5,
                    help='Learning rate for training after 20 epochs')

args = parser.parse_args()

df = generate_csv(args.path1)
df= pd.read_csv(args.path2)

y = np.array(list(df["class"]))
x = np.array(list( df["path"]))
files_for_train_x = []
files_for_validation_x = []
files_for_train_y = []
files_for_validation_y = []

k_fold_splits(x,y, files_for_train_x ,  files_for_validation_x , files_for_train_y , files_for_validation_y,  n_splits = 5 )

y = [[], [], [], [], []]
x = [[], [], [], [], []]

for i in range(5):
  y[i] = encode_y(files_for_train_y[i])
  x[i] = process_x(files_for_train_x[i])  

r = [[], [], [], [], []]
for i in range(5):
  for j in range(len(y[i])):
    r[i].append((x[i][j], y[i][j]))
list1 = [[], [], [], [], []]
for i in range(5):
  for j in range(len(r[i])):
    h_s_aug = horizontal_shift(r[i][j][0])
    v_s_aug = vertical_shift(r[i][j][0])
    b_aug = brightness(r[i][j][0])
    z_aug = zoom(r[i][j][0])
    c_s_aug = channel_shift(r[i][j][0])
    h_f_aug = horizontal_flip(r[i][j][0])
    v_f_aug = vertical_flip(r[i][j][0])
    r_aug = rotation(r[i][j][0])
    hsv_aug = HSV(r[i][j][0])
    yuv_aug = YUV(r[i][j][0])
    list2 = [h_s_aug, v_s_aug, b_aug, z_aug, c_s_aug, h_f_aug, v_f_aug, r_aug, hsv_aug, yuv_aug]
    for items in list2:
      list1[i].append((items, r[i][j][1]))

for i in range(len(list1)):
  for j in range(len(list1[i])):
    r[i].append(list1[i][j])


X_train_fold0 = []
X_train_fold1 = []
X_train_fold2 = []
X_train_fold3 = []
X_train_fold4 = []

for i in range(len(r[0])):
  X_train_fold0.append(r[0][i][0])

for i in range(len(r[1])):
  X_train_fold1.append(r[1][i][0])

for i in range(len(r[2])):
  X_train_fold2.append(r[2][i][0])

for i in range(len(r[3])):
  X_train_fold3.append(r[3][i][0])

for i in range(len(r[4])):
  X_train_fold4.append(r[4][i][0])

X_train_fold0 = np.array(X_train_fold0)
X_train_fold1 = np.array(X_train_fold1)
X_train_fold2 = np.array(X_train_fold2)
X_train_fold3 = np.array(X_train_fold3)
X_train_fold4 = np.array(X_train_fold4)



y_train_fold0 = []
y_train_fold1 = []
y_train_fold2 = []
y_train_fold3 = []
y_train_fold4 = []

for i in range(len(r[0])):
  y_train_fold0.append(r[0][i][1])

for i in range(len(r[1])):
  y_train_fold1.append(r[1][i][1])

for i in range(len(r[2])):
  y_train_fold2.append(r[2][i][1])

for i in range(len(r[3])):
  y_train_fold3.append(r[3][i][1])

for i in range(len(r[4])):
  y_train_fold4.append(r[4][i][1])

y_train_fold0 = np.array(y_train_fold0)
y_train_fold1 = np.array(y_train_fold1)
y_train_fold2 = np.array(y_train_fold2)
y_train_fold3 = np.array(y_train_fold3)
y_train_fold4 = np.array(y_train_fold4)

X_val_0 = process_x(files_for_validation_x[0])
X_val_1 = process_x(files_for_validation_x[1])
X_val_2 = process_x(files_for_validation_x[2])
X_val_3 = process_x(files_for_validation_x[3])
X_val_4 = process_x(files_for_validation_x[4])

y_val_0 = encode_y(files_for_validation_y[0])
y_val_1 = encode_y(files_for_validation_y[1])
y_val_2 = encode_y(files_for_validation_y[2])
y_val_3 = encode_y(files_for_validation_y[3])
y_val_4 = encode_y(files_for_validation_y[4])

X_TRAIN= [X_train_fold0, X_train_fold1, X_train_fold2, X_train_fold3, X_train_fold4]
Y_TRAIN= [y_train_fold0, y_train_fold1, y_train_fold2, y_train_fold3, y_train_fold4]

X_VAL= [X_val_0, X_val_1, X_val_2, X_val_3, X_val_4]
Y_VAL= [y_val_0, y_val_1, y_val_2, y_val_3, y_val_4]

for i in len(X_TRAIN):
    X_train = X_TRAIN[i]
    y_train = Y_TRAIN[i]
    X_val= X_VAL[i]
    y_val= Y_VAL[i]
    fold_no= i+1

     
    train_datagen = ImageDataGenerator(rescale=1./255)

    val_datagen = ImageDataGenerator(rescale=1./255)

    train =  train_datagen.flow(X_train, y_train, 
                                batch_size=args.batch_size,
                                shuffle = True)
    validation = val_datagen.flow(X_val, y_val,
                            batch_size=args.batch_size,
                            shuffle = False)
    
    model1= create_model(fold_no,"Xception", IMG_SIZE = 224, output = 2)
    
    # Using a picewise constant decay scheduler:-
    step = tf.Variable(0, trainable=False)
    boundaries = [20, ]
    values = [args.lr1, args.lr2]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)

    # Compile the model
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Saving the best model:-
    checkpoint = ModelCheckpoint(filepath=f"saved_models/Xception/Xceptionof{fold_no}",
                                monitor="val_accuracy", verbose=2, save_best_only=True)
    callbacks = [checkpoint]
    # Fit data to model
    start = datetime.now()
    history = model1.fit(train, epochs=args.epochs, validation_data=validation, callbacks = [callbacks])
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    plot_loss_curves(history)

    hist_df = pd.DataFrame(history.history) 
    filepath = f"saved_models/Xception/History/{fold_no}" 
    with open(filepath, mode='w') as f:
        hist_df.to_csv(f)

    '''------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    model2= create_model(fold_no,"InceptionV3", IMG_SIZE = 224, output = 2)
    # Compile the model
    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Saving the best model:-
    checkpoint = ModelCheckpoint(filepath=f"saved_models/InceptionV3/InceptionV3of{fold_no}",
                                monitor="val_accuracy", verbose=2, save_best_only=True)

    callbacks = [checkpoint]
    # Fit data to model
    start = datetime.now()
    history = model1.fit(train, epochs=args.epochs, validation_data=validation, callbacks = [callbacks])
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    plot_loss_curves(history)

    hist_df = pd.DataFrame(history.history) 
    filepath = f"saved_models/InceptionV3/History/{fold_no}" 
    with open(filepath, mode='w') as f:
        hist_df.to_csv(f)

    '''------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    model3= create_model(fold_no, "DenseNet169", IMG_SIZE = 224, output = 2)
    # Compile the model
    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Saving the best model:-
    checkpoint = ModelCheckpoint(filepath=f"saved_models/DenseNet169/DenseNet169of{fold_no}",
                                monitor="val_accuracy", verbose=2, save_best_only=True)

    callbacks = [checkpoint]
    # Fit data to model
    start = datetime.now()
    history = model1.fit(train, epochs=args.epochs, validation_data=validation, callbacks = [callbacks])
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    plot_loss_curves(history)

    hist_df = pd.DataFrame(history.history) 
    filepath = f"saved_models/DenseNet169/History/{fold_no}" 
    with open(filepath, mode='w') as f:
        hist_df.to_csv(f)

    '''------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    # Performance Matrices of the base classifiers:-
    ## Xception model:-
    pred1= model1.predict(validation)
    pred1_1d= np.argmax(pred1, axis= 1)
    pred1_1d = pred1_1d.astype("float32")
    accuracy = accuracy_score(pred1_1d, y_val)
    print(f"The accuracy of Xception model on {fold_no}th validation dataset: ", accuracy)
    cm1 = confusion_matrix(y_val, pred1_1d)
    print(f"The Confusion Matrix of Xception model on {fold_no}th validation dataset\n", cm1)
    print(f"The classification report of Xception model on {fold_no}th validation dataset\n", classification_report(y_val, pred1_1d, target_names=["Monkeypox", "Others"]))
    
    ## InceptionV3 model:-
    pred2= model2.predict(validation)
    pred2_1d= np.argmax(pred2, axis= 1)
    pred2_1d = pred2_1d.astype("float32")
    accuracy = accuracy_score(pred2_1d, y_val)
    print(f"The accuracy of InceptionV3 model on {fold_no}th validation dataset: ", accuracy)
    cm1 = confusion_matrix(y_val, pred2_1d)
    print(f"The Confusion Matrix of InceptionV3 model on {fold_no}th validation dataset\n", cm1)
    print(f"The classification report of InceptionV3 model on {fold_no}th validation dataset\n", classification_report(y_val, pred2_1d, target_names=["Monkeypox", "Others"]))
    
    ## DenseNet169 model:-
    pred3= model3.predict(validation)
    pred3_1d= np.argmax(pred3, axis= 1)
    pred3_1d = pred3_1d.astype("float32")
    accuracy = accuracy_score(pred3_1d, y_val)
    print(f"The accuracy of DenseNet169 model on {fold_no}th validation dataset: ", accuracy)
    cm1 = confusion_matrix(y_val, pred3_1d)
    print(f"The Confusion Matrix of DenseNet169 model on {fold_no}th validation dataset\n", cm1)
    print(f"The classification report of DenseNet169 model on {fold_no}th validation dataset\n", classification_report(y_val, pred3_1d, target_names=["Monkeypox", "Others"]))
    
    # Ensembling the predictions of these three models on validation dataset 
    # by sum rule after normalizing them using beta function

    s1n = beta_norm(pred1)
    s2n = beta_norm(pred2)
    s3n = beta_norm(pred3)
    
    #Using Sum Rule:-
    preds = [s1n, s2n, s3n]
    preds = np.array(preds)
    summed = np.sum(preds, axis=0)
    ensemble_prediction = np.argmax(summed, axis=1)
    ensembled_accuracy = accuracy_score(ensemble_prediction, y_val)
    print(f"The accuracy on the {fold_no}th validation dataset after using sum rule following beta_norm: ", ensembled_accuracy)

    cm1 = confusion_matrix(y_val, ensemble_prediction)
    print(f"The Confusion Matrix on the {fold_no}th validation dataset after using sum rule following beta_norm\n", cm1)

    print(f"The classification report on the {fold_no}th validation dataset after using sum rule following beta_norm\n", classification_report(y_val, ensemble_prediction, target_names=["Monkeypox", "Others"]))



