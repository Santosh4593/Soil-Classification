from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models,layers
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import cv2
BATCH_SIZE=30
IMAGE_SIZE=256
EPOCHS=1
CHANNELS=3
dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "Soil-Dataset",seed=123,shuffle=True,image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names=dataset.class_names
print(class_names)
print(len(dataset))

for image_batch,label_batch in dataset.take(1):
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
  layers.experimental.preprocessing.Rescaling(1./255),
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
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
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

"""plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label=' Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label=' Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()"""

global img, filename

w = Tk()
w.geometry("500x500")
w.title("Main Window")

def openNewWindow():
    global img

    try:
        f_types = [('Jpg Files', '.jpg'), ('Png Files', '.png'), ('Jpeg Files', '*.jpeg')]
        print(f_types)
        filename = filedialog.askopenfilename(filetypes=f_types)
        ab = preprocessing.image.load_img(filename)
        image_array = preprocessing.image.img_to_array(ab)
        scaled_img = np.expand_dims(image_array, axis=0)
        pred = model.predict(scaled_img)
        output = class_names[np.argmax(pred)]
        print(output)
        l = Label(w, text=output)
        l.pack(pady=50)
        # sign_image = Label(w)
        # uploaded = Image.open(filename)
        # uploaded.thumbnail(((w.winfo_width() / 2.25), (w.winfo_height() / 2.25)))
        # im = ImageTk.PhotoImage(uploaded)
        # sign_image.configure(image=im)
        # sign_image.image = im
        # sign_image.pack(pady=70)
        # print(type(sign_image))

        # print(im)
        print(filename)
        #show_classify_button(file_path)
    except:
        pass


    def info():


        image = Image.open(filename)
        resize_image = image.resize((300, 205))
        img = ImageTk.PhotoImage(resize_image)
        bg = ImageTk.PhotoImage(file=filename)



        window = Toplevel(w)
        window.geometry("1000x900")
        window.title("New Window")
        # b2 = tk.Label(window, image=bg)
        # b2.pack()
        if output == 'Alluvial Soil':
            Label(window, text="FOUND IN: Most of the delta areas of North India.", font=20).pack()
            Label(window, text="It covers over 35% of total Indian land.", font=20).pack()
            Label(window, text="CHARACTERISTICS: Mix of sandy loam and clay soil. Quick draining in nature.",
                  font=20).pack()
            Label(window, text="Rich in humus (organic matter) and phosphoric acid. Poor in potash and nitrogen.",
                  font=20).pack()
            Label(window,
                  text="SUITABLE CROPS: Cotton, wheat, sorghum, bajra, maize, barley, jute, tobacco, green and black gram, ",
                  font=20).pack()
            Label(window,
                  text="chickpea, pigeon pea, soybean, sesame, groundnut, linseed + any type of oilseed, fruit, and vegetable.",
                  font=20).pack()
        elif output == "Black Soil":
            Label(window,
                  text="FOUND IN:Deccan lava tract. This includes states of Gujarat, Madhya Pradesh, Maharashtra, Andhra Pradesh,",
                  font=20).pack()
            Label(window,
                  text=" Tamil Nadu, and Telangana. This soil type is prominent in river valley of rivers Narmada, Godavari, Tapi, and Krishna.",
                  font=20).pack()
            Label(window,
                  text="CHARACTERISTICS: These soils are formed when lava rocks weather away. Rich in magnesium, iron, aluminum, and lime.",
                  font=20).pack()
            Label(window,
                  text="Poor in nitrogen, phosphorus, and organic matter. Black soils get sticky when fully wet. They develop cracks when fully dry.",
                  font=20).pack()
            Label(window,
                  text="SUITABLE CROPS: Cotton is the major crop. So, black soil is also called black cotton soil. Other crops include wheat, cereals, rice, ",
                  font=20).pack()
            Label(window,
                  text=" jowar, sugarcane, linseed, sunflower, groundnut, tobacco, millets, citrus fruits, oilseed crops of all kinds, and vegetables.",
                  font=20).pack()
        elif output == 'Red Soil':
            Label(window, text="FOUND IN: Deccan plateau, Western Ghat, Orissa, and Chhattisgarh.", font=20).pack()
            Label(window,
                  text="CHARACTERISTICS: The soils are red due to iron oxide in them. The soils form when metamorphic rocks weather away",
                  font=20).pack()
            Label(window,
                  text="Rich in potash. Somewhat acidic. Poor in nitrogen, magnesium, lime, phosphorus, and organic matter.",
                  font=20).pack()
            Label(window, text="SUITABLE CROPS: Groundnut, potato, maize/corn, rice, ragi, wheat, millets, pulses, ",
                  font=20).pack()
            Label(window, text="sugarcane, oilseeds, and fruits like citrus, orange, mango, and vegetables..",
                  font=20).pack()
        elif output == 'Yellow Soil':
            Label(window, text="FOUND IN: Deccan plateau, Western Ghat, Orissa, and Chhattisgarh.", font=20).pack()
            Label(window,
                  text="CHARACTERISTICS: The soils are red due to iron oxide in them. The soils form when metamorphic rocks weather away",
                  font=20).pack()
            Label(window,
                  text="Rich in potash. Somewhat acidic. Poor in nitrogen, magnesium, lime, phosphorus, and organic matter.",
                  font=20).pack()
            Label(window, text="SUITABLE CROPS: Groundnut, potato, maize/corn, rice, ragi, wheat, millets, pulses, ",
                  font=20).pack()
            Label(window, text="sugarcane, oilseeds, and fruits like citrus, orange, mango, and vegetables..",
                  font=20).pack()
        elif output == 'Laterite Soil':
            Label(window, text="FOUND IN: Madhya Pradesh, Kerala, Karnataka, Tamil Nadu, Assam, and Orissa.",
                  font=20).pack()
            Label(window,
                  text="CHARACTERISTICS: Acidic soils, rich in iron. They are used in the making of bricks due to high",
                  font=20).pack()
            Label(window,
                  text="iron content. Poor in organic matter, calcium, nitrogen, and phosphate. Not very fertile.",
                  font=20).pack()
            Label(window,
                  text="SUITABLE CROPS: Cotton, wheat, rice, pulses, rubber, tea, coffee, coconut, and cashews. ",
                  font=20).pack()
        elif output == 'Arid Soil':
            Label(window, text="FOUND IN: Aravalli west.", font=20).pack()
            Label(window,
                  text="CHARACTERISTICS: Sandy soils with low clay content. Poor in organic matter and moisture .",
                  font=20).pack()
            Label(window,
                  text="because arid regions are usually dry. Saline in nature with low nitrogen and high salt. Rich in plant food",
                  font=20).pack()
            Label(window, text="SUITABLE CROPS:Saline resistant and drought tolerant crops are suitable. ",
                  font=20).pack()
            Label(window, text=" Barley, maize, wheat, millets, cotton, and pulses.", font=20).pack()
        elif output == 'Mountain_Forest Soil':
            Label(window,
                  text="FOUND IN: Himalayan area, Western and Eastern Ghats, and a few regions of the Peninsular Plateau.",
                  font=20).pack()
            Label(window,
                  text="CHARACTERISTICS: Acidic soil, rich in organic matter. Poor in lime, phosphorus, and potash.",
                  font=20).pack()
            Label(window, text="Good fertilization is required in these soils.", font=20).pack()
            Label(window,
                  text="SUITABLE CROPS: Wheat, barley, maize, tea, coffee, spices, tropical and temperate fruits. ",
                  font=20).pack()

        def show_image():

            # f_types = [('Jpg Files', '*.jpg')]
            # print(f_types)
            # filename = filedialog.askopenfilename(filetypes=f_types)
            # image = Image.open(filename)
            # resize_image = image.resize((300,205))
            # img = ImageTk.PhotoImage(resize_image)
            # print(type(img))
            #
            # b2 = tk.Button(w, image=img)
            # b2.pack()
            x = cv2.imread(filename)
            resize = cv2.resize(x, (500, 500))
            # y=cv2.imwrite('op.jpg',resize)
            cv2.imshow('IMAGE', resize)
            cv2.waitKey(0)
        b = Button(window, text="Show image", command=show_image)
        b.pack(pady=40)

    b = Button(w, text="Show info",command=info)
    b.pack(pady=60)
l =Label(w,text="puT iMAGE HERE")
l.pack(pady=30)

c=Button(w,text='Upload',command=openNewWindow)

c.pack(pady=35)

w.mainloop()