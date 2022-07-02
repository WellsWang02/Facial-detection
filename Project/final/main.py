from utils import *

# Load training set
X_train, y_train = load_data()
print("X_train.shape == {}".format(X_train.shape))
print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

# Load testing set
X_test, _ = load_data(test=True)
print("X_test.shape == {}".format(X_test.shape))


# Import deep learning resources from Keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense


# # Build a CNN architecture
#
# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(96, 96, 1)))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Flatten())
#
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
#
#
# model.add(Dense(30))
#
#
# # Summarize the model
# model.summary()



## TODO: Specify a CNN architecture
# Your model should accept 96x96 pixel graysale images in
# It should have a fully-connected output layer with 30 values (2 for each facial keypoint)

model = Sequential()
model.add(Convolution2D(8, (3,3), input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))

model.add(Convolution2D(16, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))

model.add(Convolution2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(500))
model.add(Dropout(0.3))

model.add(Dense(30))

# Summarize the model
model.summary()


#step6: Compile and Train the Model
# from keras.callbacks import ModelCheckpoint, History
# from keras.optimizers import Adam
#
# hist = History()
# epochs = 50
# batch_size = 64
#
# checkpointer = ModelCheckpoint(filepath='weights.final_2.hdf5',
#                                verbose=1, save_best_only=True)
#
# ## TODO: Compile the model
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#
# hist_final = model.fit(X_train, y_train, validation_split=0.2,
#           epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, hist], verbose=1)
#
#
# model.save('my_model.h5')


#step6: Compile and Train the Model
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

## TODO: Compile the model
model.compile(optimizer='Adagrad', loss='mean_squared_error', metrics=['accuracy'])

epochs = 15
## TODO: Train the model
history = model.fit(X_train, y_train,
          validation_split=0.2,
          epochs=epochs, batch_size=10, verbose=1)

## TODO: Save the model as model.h5
model.save('my_model.h5')


# # Visualize the training and validation loss of the neural network
# plt.plot(range(epochs), hist_final.history[
#          'val_loss'], 'g-', label='Val Loss')
# plt.plot(range(epochs), hist_final.history[
#          'loss'], 'g--', label='Train Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
#
#
#
# # Visualize a subset of the test predictions
# y_test = model.predict(X_test)
# fig = plt.figure(figsize=(20,20))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
#     plot_data(X_test[i], y_test[i], ax)
#
#
#
# #facial detection``````
# # Load in color image for face detection
# image = cv2.imread('images/obamas4.jpg')
#
#
# # Convert the image to RGB colorspace
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#
# # plot our image
# fig = plt.figure(figsize = (9,9))
# ax1 = fig.add_subplot(111)
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.set_title('image')
# ax1.imshow(image)
#
#
#
# # Use the face detection code with our trained conv-net
# def plot_keypoints(img_path, face_cascade_path, model_path, scale=1.2, neighbors=5, key_size=10):
#
#     face_cascade=cv2.CascadeClassifier(face_cascade_path)
#     img = cv2.imread(img_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scale, neighbors)
#     fig = plt.figure(figsize=(40, 40))
#     ax = fig.add_subplot(121, xticks=[], yticks=[])
#     ax.set_title('Image with Facial Keypoints')
#
#     # Print the number of faces detected in the image
#     print('Number of faces detected:', len(faces))
#
#     # Make a copy of the orginal image to draw face detections on
#     image_with_detections = np.copy(img)
#
#     # Get the bounding box for each detected face
#     for (x,y,w,h) in faces:
#         # Add a red bounding box to the detections image
#         cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
#         bgr_crop = image_with_detections[y:y+h, x:x+w]
#         orig_shape_crop = bgr_crop.shape
#         gray_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
#         resize_gray_crop = cv2.resize(gray_crop, (96, 96)) / 255
#         model = load_model(model_path)
#         landmarks = np.squeeze(model.predict(
#             np.expand_dims(np.expand_dims(resize_gray_crop, axis=-1), axis=0)))
#         ax.scatter(((landmarks[0::2] * 48 + 48)*orig_shape_crop[0]/96)+x,
#                    ((landmarks[1::2] * 48 + 48)*orig_shape_crop[1]/96)+y,
#                    marker='o', c='c', s=key_size)
#
#     ax.imshow(cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB))
#
#
# # Paint the predicted keypoints on the test image
# obamas = plot_keypoints('images/obamas4.jpg',
#                         'detector_architectures/haarcascade_frontalface_default.xml',
#                         'my_model_final.h5')
