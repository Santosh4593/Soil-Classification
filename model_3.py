import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk,ImageFilter
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models,layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from tensorflow.keras.regularizers import l2
w = tk.Tk()
w.geometry("1200x725")
w.title("Main Window")
w.configure(bg='#1C2541')
sign_image = Label(w,bg='#1C2541')
sign_image2 = Label(w,bg='#1C2541')
grayscale=Label(w,bg='#1C2541')
file_path=""
file_path2=""
acc=0
acc2=0
EPOCHS=50
history=''
history2=''

def upload_image():
    global resize_image, file_path

    try:

        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        resize_image = uploaded.resize((175, 175))

        im = ImageTk.PhotoImage(resize_image)
        sign_image.configure(image=im)
        sign_image.image = im
    except:
        pass

    Label(w, text='Original Image', bg='#EF626C').place(x=40, y=55)


def grayscale_image():
    uploaded = Image.open(file_path)
    # print(type(uploaded))
    resize_image = uploaded.resize((175, 175))
    # b=Image.fromarray(resize_image)
    image = resize_image.convert("L")
    image = image.filter(ImageFilter.FIND_EDGES)

    # a=cv2.imwrite('Canny.jpg',b)
    # readimg=cv2.imread(a)
    #
    # Canny = cv2.Canny(readimg, 100, 200)
    #
    # cv2.imshow("canny",Canny)
    #
    im = ImageTk.PhotoImage(image)
    grayscale.configure(image=im)
    grayscale.image = im

    Label(w, text='Grayscale Image', bg='#EF626C').place(x=40, y=290)



def detect_soil():
    global EPOCHS, history, output,acc,acc2
    messagebox.showinfo("Process Starting", "Please Wait until Soil type is Predicted")


    BATCH_SIZE = 30
    IMAGE_SIZE = 256
    EPOCHS = 50
    CHANNELS = 3
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "Soil-Dataset", seed=123, shuffle=True, image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = dataset.class_names
    print(class_names)
    print(len(dataset))

    for image_batch, label_batch in dataset.take(1):
        print(image_batch.shape)
        print(image_batch[1])
        print(label_batch.numpy())

    plt.figure(figsize=(15, 15))
    for image_batch, labels_batch in dataset.take(1):
        for i in range(BATCH_SIZE):
            ax = plt.subplot(8, 8, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[labels_batch[i]])
            plt.axis("off")

    def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        assert (train_split + test_split + val_split) == 1
        ds_size = len(ds)
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=12)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
        # Autotune all the 3 datasets
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_ds, val_ds, test_ds

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1. / 255),
    ])

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    n_classes = 9

    model = models.Sequential([
        resize_and_rescale,
        # data_augmentation,
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.build(input_shape=input_shape)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        validation_data=val_ds,
        verbose=1,
        epochs=EPOCHS,
    )


    model.evaluate(test_ds)

    acc = history.history['accuracy']
    loss = history.history['loss']
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(EPOCHS), acc, label=' Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(EPOCHS), loss, label=' Loss')
    # plt.legend(loc='upper right')
    # plt.title('Loss')
    # plt.show()

    # image_path = "Soil-Dataset/Black Soil/6.jpg"

    image = preprocessing.image.load_img(file_path)
    image_array = preprocessing.image.img_to_array(image)
    scaled_img = np.expand_dims(image_array, axis=0)
    print(resize_image)

    pred = model.predict(scaled_img)
    output = class_names[np.argmax(pred)]
    print(output)
    print(acc)
    Label(w, text=output, width=12, height=2, font=('Arial',12,'bold'),bg='#A9BCD0').place(x=335, y=490)

    number_of_classes = 6
    model2 = Sequential()
    model2.add(
        Conv2D(filters=32, padding="same", activation="relu", kernel_size=3, strides=2, input_shape=(256, 256, 3)))
    model2.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model2.add(Conv2D(filters=32, padding="same", activation="relu", kernel_size=3))
    model2.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model2.add(Flatten())
    model2.add(Dense(128, activation="relu"))
    model2.summary()
    model2.add(Dense(1, kernel_regularizer=l2(0.01), activation="linear"))
    model2.compile(optimizer='adam', loss="hinge", metrics=['accuracy'])
    history2 = model2.fit(x=train_ds, validation_data=val_ds, epochs=2)
    model2.evaluate(test_ds)

    acc2 = history2.history['accuracy']
    loss2 = history2.history['loss']
    print(acc2)

def load_weed_image():
    global resize_image2, file_path2

    try:

        file_path2 = filedialog.askopenfilename()
        uploaded = Image.open(file_path2)
        resize_image2 = uploaded.resize((175, 175))

        im2 = ImageTk.PhotoImage(resize_image2)
        sign_image2.configure(image=im2)
        sign_image2.image = im2
    except:
        pass

    Label(w, text='Weed Image', bg='#EF626C').place(x=40, y=515)

def detect_weed():
    global EPOCHS, history2, output
    messagebox.showinfo("Process Starting", "Please Wait until Weed type is Predicted")


    BATCH_SIZE = 30
    IMAGE_SIZE = 256
    EPOCHS = 50
    CHANNELS = 3
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "Weed dataset", seed=123, shuffle=True, image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = dataset.class_names
    print(class_names)
    print(len(dataset))

    for image_batch, label_batch in dataset.take(1):
        print(image_batch.shape)
        print(image_batch[1])
        print(label_batch.numpy())

    plt.figure(figsize=(15, 15))
    for image_batch, labels_batch in dataset.take(1):
        for i in range(BATCH_SIZE):
            ax = plt.subplot(8, 8, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[labels_batch[i]])
            plt.axis("off")

    def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        assert (train_split + test_split + val_split) == 1
        ds_size = len(ds)
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=12)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
        # Autotune all the 3 datasets
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_ds, val_ds, test_ds

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1. / 255),
    ])

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    n_classes = 9

    model = models.Sequential([
        resize_and_rescale,
        # data_augmentation,
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.build(input_shape=input_shape)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.summary()

    history2 = model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        validation_data=val_ds,
        verbose=1,
        epochs=EPOCHS,
    )

    model.evaluate(test_ds)

    image = preprocessing.image.load_img(file_path2)
    image_array = preprocessing.image.img_to_array(image)
    scaled_img = np.expand_dims(image_array, axis=0)
    print(image)

    pred = model.predict(scaled_img)
    output2= class_names[np.argmax(pred)]
    print(output2)

    Label(w, text=output2, width=12, height=2, font=('Arial',12,'bold'),bg='#A9BCD0').place(x=615, y=490)

def SVM_acc():
    accuracy2 = acc2[-1] * 100
    accu2 = round(accuracy2, 2)
    Label(w, text=str(accu2)+'%', font=('Arial', 12, 'bold'),bg='#A9BCD0', width=10, height=2).place(x=700, y=590)

def summary():
    if output == 'Alluvial Soil':
        Label(w, text="Nothern Plains, Assam, Bihar and West Bengal",font=('Arial',12),bg='#A9BCD0').place(x=380, y=80)
        Label(w, text="Rich in Humus and organic matter.",font=('Arial',12),bg='#A9BCD0').place(x=500, y=140)
        Label(w, text="1.Manure  2.Compost  3.Fish Extract",font=('Arial',12),bg='#A9BCD0').place(x=420,y=260)
        Label(w, text="75cm to 100cm",font=('Arial',12),bg='#A9BCD0').place(x=470, y=320)
        Label(w, text="1`C to 28`C",font=('Arial',12),bg='#A9BCD0').place(x=470, y=380)

    elif output == "Black Soil":
        Label(w, text="Gujarat, Madhya Pradesh, Maharashtra, Andhra Pradesh",font=('Arial',12),bg='#A9BCD0').place(x=380,y=80)
        Label(w, text="Rich in magnesium, iron, aluminum, and lime.",font=('Arial',12),bg='#A9BCD0').place(x=500, y=140)
        Label(w, text="1.Cocpeat  2.Vermicompost",font=('Arial',12),bg='#A9BCD0').place(x=420, y=260)
        Label(w, text="60cm to 80cm",font=('Arial',12),bg='#A9BCD0').place(x=470, y=320)
        Label(w, text="27`C to 32`C",font=('Arial',12),bg='#A9BCD0').place(x=470, y=380)

    elif output == 'Red Soil':
        Label(w, text="Deccan Plateau",font=('Arial',12),bg='#A9BCD0').place(x=380, y=80)
        Label(w, text="Rich in Potash and is somewhat Acidic in nature.",font=('Arial',12),bg='#A9BCD0').place(x=500, y=140)
        Label(w, text="1.Ammonium Sulphate",font=('Arial',12),bg='#A9BCD0').place(x=420, y=260)
        Label(w, text="140cm to 200cm",font=('Arial',12),bg='#A9BCD0').place(x=470, y=320)
        Label(w, text="18`C to 28`C",font=('Arial',12),bg='#A9BCD0').place(x=470, y=380)


    elif output == 'Yellow Soil':
        Label(w, text="Middle Ganga plain and Piedmont zone of Western Ghats",font=('Arial',12),bg='#A9BCD0').place(x=380, y=80)
        Label(w, text="Rich in Iron Oxides.",font=('Arial',12),bg='#A9BCD0').place(x=500, y=140)
        Label(w, text="1.Triple Super Phosphate",font=('Arial',12),bg='#A9BCD0').place(x=420, y=260)
        Label(w, text="25cm to 60cm",font=('Arial',12),bg='#A9BCD0').place(x=470, y=320)
        Label(w, text="20`C to 25`C",font=('Arial',12),bg='#A9BCD0').place(x=470, y=380)


    elif output == 'Laterite Soil':
        Label(w, text="Central India and Western Peninsula.",font=('Arial',12),bg='#A9BCD0').place(x=380, y=80)
        Label(w, text="It is Acidic in nature and is not very fertile.",font=('Arial',12),bg='#A9BCD0').place(x=500, y=140)
        Label(w, text="1.Sodium Silicate",font=('Arial',12),bg='#A9BCD0').place(x=420, y=260)
        Label(w, text="125cm to 200cm",font=('Arial',12),bg='#A9BCD0').place(x=470, y=320)
        Label(w, text="18`C to 20`C",font=('Arial',12),bg='#A9BCD0').place(x=470, y=380)

    elif output == 'Arid Soil':
        Label(w, text="Haryana, Western Rajasthan and the Rann of Kutch",font=('Arial',12),bg='#A9BCD0').place(x=380, y=80)
        Label(w, text="Sandy texture and quick draining in nature.",font=('Arial',12),bg='#A9BCD0').place(x=500, y=140)
        Label(w, text="1.Ammonium Nitrate  2.Ammonium Phosphate",font=('Arial',12),bg='#A9BCD0').place(x=420,y=260)
        Label(w, text="50cm to 75cm",font=('Arial',12),bg='#A9BCD0').place(x=470, y=320)
        Label(w, text="20`C to 30`C",font=('Arial',12),bg='#A9BCD0').place(x=470, y=380)

    elif output == 'Mountain Soil':
        Label(w, text="Western/Eastern Ghats and regions of the Peninsular Plateau.",font=('Arial',12),bg='#A9BCD0').place(x=380, y=80)
        Label(w, text="Rich in Humus and organic Matter.",font=('Arial',12),bg='#A9BCD0').place(x=500, y=140)
        Label(w, text="1.Ammonium Nitrate",font=('Arial',12),bg='#A9BCD0').place(x=420, y=260)
        Label(w, text="50cm to 75cm",font=('Arial',12),bg='#A9BCD0').place(x=470, y=320)
        Label(w, text="20`C to 30`C",font=('Arial',12),bg='#A9BCD0').place(x=470, y=380)

def crops():
    if output=="Alluvial Soil":
        Label(w, text="Cotton, Wheat, Sorghum, Bajra, Maize",font=('Arial',12),bg='#A9BCD0').place(x=485,y=200)
    elif output=="Black Soil":
        Label(w, text="Cotton,Wheat,Linseed,Oilseeds",font=('Arial',12),bg='#A9BCD0').place(x=485, y=200)
    elif output=="Red Soil":
        Label(w, text="Groundnut, Potato, Maize(Corn), Rice, Wheat",font=('Arial',12),bg='#A9BCD0').place(x=485, y=200)
    elif output=="Yellow Soil":
        Label(w, text="Groundnut, Potato, Cofee, Coconut,Rice etc.",font=('Arial',12),bg='#A9BCD0').place(x=485, y=200)
    elif output=="Laterite Soil":
        Label(w, text="Cotton, Wheat, Rice, Pulses, Tea, Coffee, Coconut",font=('Arial',12),bg='#A9BCD0').place(x=485, y=200)
    elif output=="Arid Soil":
        Label(w, text="Corn, Sorghum, Pearl Millets, Seasame.",font=('Arial',12),bg='#A9BCD0').place(x=485,y=200)
    elif output=="Mountain Soil":
        Label(w, text="Maize, Tea, Coffee, Spices, Tropical Fruits.",font=('Arial',12),bg='#A9BCD0').place(x=485, y=200)








def accuracy():
    accuracy=acc[0]*100
    accu=round(accuracy,2)
    Label(w,text=str(accu)+'%',font=('Arial',12,'bold'),bg='#A9BCD0',width=10,height=2).place(x=460,y=590)

def accuracy_graph():
    # Label(w,text="Accuracy and Loss Graph").place(x=775,y=75)
    acc = history.history['accuracy']
    loss = history.history['loss']
    # plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), acc, label=' Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), loss, label=' Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    # a=plt.savefig("Graph.jpg")
    plt.show()
    # uploaded1 = Image.open('Graph.jpg')
    # resize_image1 = uploaded1.resize((300,225))
    #
    # im = ImageTk.PhotoImage(resize_image1)
    # graph_image.configure(image=im)
    # graph_image.image = im



Label(w,text='SOIL CLASSIFICATION USING DEEP LEARNING',font=('Arial',16,('bold','underline')),bg='#1C2541',fg='white').pack()
Label(w,text='',bg='#A9BCD0',width=30,height=39).place(x=930,y=50)
Label(w,text='Menu',bg='#EF626C').place(x=960,y=40)
Button(w,text='Load Soil Image',width=15,font=('Arial',14),bg='#EF626C',command=upload_image).place(x=950,y=80)
Button(w,text='Detect Soil',width=15,font=('Arial',14),bg='#EF626C',command=detect_soil).place(x=950,y=135)
Button(w,text='CNN Accuracy',width=15,font=('Arial',14),bg='#EF626C',command=accuracy).place(x=950,y=190)
Button(w,text='SVM Accuracy',width=15,font=('Arial',14),bg='#EF626C',command=SVM_acc).place(x=950,y=245)
Button(w,text='Suitable Crops',width=15,font=('Arial',14),bg='#EF626C',command=crops).place(x=950,y=300)
Button(w,text='Grayscale image',width=15,font=('Arial',14),bg='#EF626C',command=grayscale_image).place(x=950,y=355)
Button(w,text='Graph',width=15,font=('Arial',14),bg='#EF626C',command=accuracy_graph).place(x=950,y=410)
Button(w,text='Soil Summary',font=('Arial',14),bg='#EF626C',height=1,command=summary, width=15).place(x=950,y=465)
Button(w,text='Load Weed image',width=15,font=('Arial',14),bg='#EF626C',command=load_weed_image).place(x=950,y=520)
Button(w,text='Detect Weed type',width=15,font=('Arial',14),bg='#EF626C',command=detect_weed).place(x=950,y=575)

Label(w,text='',bg='#A9BCD0',width=90,height=25).place(x=260,y=50)
Label(w,text="Summary",bg='#EF626C').place(x=270,y=40)
Label(w,text="REGION: ",font=2,bg='#A9BCD0').place(x=280,y=80)
Label(w,text="CHARACTERISTICS: ",font=2,bg='#A9BCD0').place(x=280,y=140)
Label(w,text='SUITABLE CROPS: ',font=2,bg='#A9BCD0').place(x=280,y=200)
Label(w,text="FERTILIZER: ",font=2,bg='#A9BCD0').place(x=280,y=260)
Label(w,text="WATER SUPPLY: ",font=2,bg='#A9BCD0').place(x=280,y=320)
Label(w,text="TEMPERATURE: ",font=2,bg='#A9BCD0').place(x=280,y=380)

Label(w,text='',bg='#A9BCD0',width=25,height=4).place(x=320,y=470)
Label(w,text='Type of Soil',bg='#EF626C').place(x=330,y=460)

Label(w,text='',bg='#A9BCD0',width=25,height=4).place(x=600,y=470)
Label(w,text='Type of Weed',bg='#EF626C').place(x=610,y=460)

Label(w,text='',bg='#A9BCD0',width=90,height=4).place(x=260,y=580)
Label(w,text='Result',bg='#EF626C').place(x=310,y=570)
Label(w,text='CNN Accuracy: ',font=('Arial',12,'bold'),width=18,height=2,bg='#A9BCD0').place(x=290,y=590)
Label(w,text='SVM Accuracy: ',font=('Arial',12,'bold'),width=18,height=2,bg='#A9BCD0').place(x=550,y=590)
# Label(w,text='Best Precision',font=('Arial',8,'bold'),width=18,height=2).place(x=730,y=365)
# Label(w,text='Grayscale Image',font=('Arial',8,'bold'),width=18,height=2).place(x=860,y=365)

# upload = Button(w, text="Upload an image", command=upload_image, padx=10, pady=5)
# upload.place(x=20,y=20)
# Label(w,text='',bg='#A9BCD0',width=80,height=4).place(x=20,y=65)
sign_image.place(x=30,y=65)
grayscale.place(x=30,y=300)
sign_image2.place(x=30,y=525)
w.mainloop()
