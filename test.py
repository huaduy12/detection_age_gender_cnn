from tkinter import *

# a = Tk()
# a.title("Cua so quan ly")
#
# # top['bg'] = 'red'
# a.attributes('-topmost',True)   # làm cho cửa số luôn luôn hiện trên các uứng dụng khác
# a.geometry("600x600")
# # name = Label(a,text='Hua Nhu Duy',font=('Arial',14),bg = 'red',fg='white')
# # name.place(x=100,y=100)
# #
# # def anvao():
# #     name = Label(a, text='Da click vao', font=('Arial', 14), bg='red', fg='white')
# #     name.place(x=100, y=200)
# #     name = Label(a, text='Hua Nhu Duy 2', font=('Arial', 14), bg='red', fg='white')
# #     name.place(x=100, y=300)
# #
# #
# # entry = Entry(a,width=20,font=('Arial',14))
# # entry.place(x=100,y=60)
# # entry.focus()
# # but = Button(a,text='Click vap day',width=12,height=5,bg='grey',font=('Arial',16),fg='white',command=anvao)
# # but.place(x=250,y=100)
#
# labelName = Label(a,text="Nhập tên của bạn: ",font=('Arial',14))
# labelName.place(x=50,y=50)
#
# inputName = Entry(a,width=20,font=('Arial',14))
# inputName.place(x=210,y=50)
#
# soALabel = Label(a,text="Nhập số a: ",font=('Arial',14))
# soALabel.place(x=50,y=100)
#
# soAInput = Entry(a,width=20,font=('Arial',14))
# soAInput.insert(END,'20')
# soAInput.place(x=210,y=100)


# soBLabel = Label(a,text="Nhập sô b: ",font=('Arial',14))
# soBLabel.place(x=50,y=150)
#
# soBInput = Entry(a,width=20,font=('Arial',14))
# soBInput.insert(END,'20')
# soBInput.place(x=210,y=150)
#
# def displayName():
#     result = float(soAInput.get()) + float(soBInput.get())
#     labelDisplay = Label(a,text='Tên của bạn là: '+ inputName.get() +" "+ str(result),font=('Arial',14))
#     labelDisplay.place(x=210,y=250)
#
# buttonName = Button(a,text='Hiển thị tên',width=10,font=('Arial',14),command=displayName)
# buttonName.place(x=210,y=200)
# a.mainloop()

import joblib
# Imports
from keras.applications.vgg16 import preprocess_input
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cv2
import os

import tensorflow as tf

import matplotlib.pyplot as plt

from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPooling2D,AvgPool2D,GlobalAveragePooling2D,Dense,Dropout,BatchNormalization,Flatten,Input, AveragePooling2D
from sklearn.model_selection import train_test_split


# model_age = load_model('file_model/output_model_age.h5')
# model_gender = load_model('file_model/output_model_gender.h5')
#
# X_test = plt.imread("C:/Users/hua nhu duy/Downloads/UTKFace/115_1_0_20170120134725991.jpg.chip.jpg")
# print(X_test)
#
# img = cv2.cvtColor(X_test,cv2.COLOR_BGR2GRAY)
# img=cv2.resize(img,(128,128))
#
# y_predict_gender=model_gender.predict(img.reshape(1,128,128,1))
# y_predict_age=model_age.predict(img.reshape(1,128,128,1))
#
# print('Du doan gioi tinh:',np.argmax(y_predict_gender))
# print('Du doan tuoi:',np.argmax(y_predict_age))
# plt.imshow(X_test)
# plt.show()


# A function to save video from the video source
# def save_video(video_path):
#     # Open the video file with cv2
#     video = cv2.VideoCapture(video_path)
#     if not video.isOpened():
#         print("Không thể mở file video")
#         return
#
#     # Get the video properties
#     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = video.get(cv2.CAP_PROP_FPS)
#     codec = video.get(cv2.CAP_PROP_FOURCC)
#
#     # Generate a file name with the current date and time
#     file_name = "video_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
#
#     # Create a video writer object
#     video_writer = cv2.VideoWriter(file_name, int(codec), fps, (width, height))
#
#     # Loop through the video frames
#     while True:
#         # Get a frame from the video source
#         ret, frame = video.read()
#         if ret:
#             frame = run_cam(video, frame)  # add this line
#             # Write the frame to the video writer
#             video_writer.write(frame)
#         else:
#             # Break the loop when the video is over
#             break
#
#     # Release the video source and the video writer
#     close_windowVideo()
#     video.release()
#     video_writer.release()
#
#     # Print a message to confirm
#     print("Video đã được lưu với tên", file_name)



X_test = plt.imread("C:/Users/hua nhu duy/Downloads/UTKFace/35_1_0_20170120140127200.jpg.chip.jpg")
X_test =cv2.resize(X_test,(128,128))
model = load_model("./file_model/cnn_2_output_new.h5")
predict = model.predict(X_test.reshape(1, 128, 128, 3))
print(predict)
print(predict[0][0])
gender = "Female" if predict[0][0] > 0.5 else "Male"
age = np.argmax(predict[1][0:])
print(gender)
print(age)