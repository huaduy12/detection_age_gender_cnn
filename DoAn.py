from tkinter import *

import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog as fd
import cv2
import datetime
from keras.models import load_model
import matplotlib.pyplot as plt


model = load_model('./file_model/cnn_2_output_new.h5')

face_cascade = cv2.CascadeClassifier('./file_model/haarcascade_frontalface_default.xml')

# Define the font and text properties for cv2.putText
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 0.4
fontColor = (0, 255, 0)
thickness = 1
lineType = 2


age_group = {0: '0 - 15',
             1: '16 - 30',
             2: '31 - 45',
             3: '46 - 60',
             4: 'Over 60'
             }

window = Tk()
window.title("Dự đoán ảnh")
window.geometry("600x600")

iconSave = ImageTk.PhotoImage(Image.open("image/icons8-save-50 (1).png").resize((40, 40)))
iconCancel = ImageTk.PhotoImage(Image.open("image/icons8-cancel-16.png").resize((40, 40)))

# start camera
vid = cv2.VideoCapture(0)
width, height = 800, 600
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

fromCamera = Toplevel()
fromCamera.title("Man hinh Camera")

labelCamera = Label(fromCamera, text='Camera')
labelCamera.grid(row=0, column=0, columnspan=2)
fromCamera.withdraw()
# end


# start open video
fromVideo = Toplevel()
fromVideo.title("Man hình Video")

labelVideo = Label(fromVideo, text='Video')
labelVideo.grid(row=0, column=0, columnspan=2)
fromVideo.withdraw()
# end


# Create a file dialog to choose a video file
def choose_file(label):
    global video_path
    video_path = fd.askopenfilename(initialdir='/', title="Chọn video",
                                    filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
    # Open the video file with cv2

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Không thể mở file video")
    else:
        # Display the video on the label
        display_video(label, video)


after_idVideo = None
def display_video(label, video):
    global after_idVideo
    # Get a frame from the video source
    fromVideo.deiconify()
    window.withdraw()

    ret, frame = video.read()
    if ret:
        # Convert image from one color space to other
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        opencv_image = run_cam(video,opencv_image)
        # Capture the latest frame and transform to image
        captured_image = Image.fromarray(opencv_image)

        # Convert captured image to photo image
        photo_image = ImageTk.PhotoImage(image=captured_image)

        # Display the image on the label
        label.configure(image=photo_image)
        label.image = photo_image

        # Repeat every 10 milliseconds
        after_idVideo = label.after(4, display_video, label, video)
    else:
        # Release the video source when the video is over
        video.release()


def openFile():
    # window.withdraw()
    filetypes = (('jpg files', '*.jpg'), ('png files', '*.png'), ('All files', '*.*'))
    filename = fd.askopenfilename(initialdir='/', title="Chọn ảnh", filetypes=filetypes)
    if filename != '':

        screenFile = Toplevel()  # định nghĩa 1 màn hình con
        screenFile.title("Hiển thị ảnh")
        screenFile.geometry("500x500")

        labelDirector = Label(screenFile, text=filename)
        labelDirector.grid(row=0, column=0, columnspan=2)  # đặt label ở hàng 0, cột 0, và kéo dài 2 cột
        global my_image
        X_predict = plt.imread(filename)
        X_predict = run_cam_openFile(X_predict)
        my_image = ImageTk.PhotoImage(image=Image.fromarray(X_predict).resize((500,300)))
        my_image_label = Label(screenFile, image=my_image)
        my_image_label.grid(row=1, column=0, columnspan=2)

        # predict

        # img = cv2.cvtColor(X_predict, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(X_predict, (128, 128))
        predict = model.predict(img.reshape(1, 128, 128, 3), verbose=False)
        gender = "Nữ" if predict[0][0] > 0.5 else "Nam"
        age = np.argmax(predict[1][0:])
        labelPredict = Label(screenFile, font=('Arial', 15))
        labelPredict.grid(row=2, column=0, columnspan=2)

        buttonSave = Button(screenFile, text='Save', width=130, font=('Arial', 12), image=iconSave, compound=LEFT,
                            bg="#51aded", fg="white",command=lambda : capture_image_file(X_predict))
        buttonSave.grid(row=3, column=0, sticky=E, padx=10,
                        pady=10)

        def close_window():
            screenFile.destroy()
            window.deiconify()

        buttonCancel = Button(screenFile, text='Hủy', width=130, font=('Arial', 12), image=iconCancel, compound=LEFT,
                              bg="#ed5153", fg="white", command=close_window)
        buttonCancel.grid(row=3, column=1, sticky=W, padx=10,
                          pady=10)
#end

def run_cam_openFile(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 6,minSize=(30,30),maxSize=(400,400))
    # Display the resulting frame

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # print(frame.shape)
        temp = frame[y: y + h, x: x + w]
        temp = cv2.resize(temp, (128, 128))

        predict = model.predict(temp.reshape(1, 128, 128, 3), verbose=False)

        gender = "Nu" if predict[0][0] > 0.5 else "Nam"
        age = np.argmax(predict[1][0:])

        cv2.putText(frame, f"{gender}, {age_group[age]} tuoi",
                    (x, y - 10),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
    # Return the frame
    return frame


after_id = None


# chụp ảnh tử file
def capture_image_file(X_predict):
    frame = cv2.cvtColor(X_predict, cv2.COLOR_BGR2RGBA)
    # Generate a file name with the current date and time
    file_name = "image_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"

    # Save the image to the file
    cv2.imwrite(file_name, frame)

    # Print a message to confirm
    print("Ảnh đã được lưu với tên", file_name)


def open_camera():
    global after_id
    fromCamera.deiconify()
    window.withdraw()
    # Capture the video frame by frame
    _, frame = vid.read()

    # Run the face detection and prediction function
    frame = run_cam(vid, labelCamera)

    print("Frme: ",frame.shape)
    # Capture the latest frame and transform to image
    captured_image = Image.fromarray(frame)

    # Convert captured image to photoimage
    photo_image = ImageTk.PhotoImage(image=captured_image)

    # Displaying photoimage in the label

    labelCamera.photo_image = photo_image

    # Configure image in the label
    labelCamera.configure(image=photo_image)

    # Repeat the same process after every 10 seconds
    after_id = labelCamera.after(10, open_camera)


def run_cam(cap, frame):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.3, 10,minSize=(60,60))
    # Display the resulting frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        temp = frame[y: y + h, x: x + w]
        temp = cv2.resize(temp, (128, 128))

        # temp = cv2.resize(temp, (128, 128))
        # temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
        predict = model.predict(temp.reshape(1, 128, 128, 3), verbose=False)
        gender = "Nu" if predict[0][0] > 0.5 else "Nam"
        age = np.argmax(predict[1][0:])

        cv2.putText(frame, f" {gender}, {age_group[age]} tuoi",
                    (x, y - 10),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
    return frame


def capture_image(vid):
    # Get a frame from the video source
    _, frame = vid.read()

    # Run the face detection and prediction function
    frame = run_cam(vid, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    # Generate a file name with the current date and time
    file_name = "image_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"

    # Save the image to the file
    cv2.imwrite(file_name, frame)

    # Print a message to confirm
    print("Ảnh đã được lưu với tên", file_name)


labelName = Label(window, text="Click vào nút để lựa chọn theo dự đoán: ", font=('Arial', 14))
labelName.place(relx=0.5, rely=0.1, anchor=CENTER)

img = ImageTk.PhotoImage(Image.open("image/icons8-exit-16.png").resize((40, 40)))
iconImage = ImageTk.PhotoImage(Image.open("image/icons8-image-100.png").resize((40, 40)))
buttonImage = Button(window, text='Chọn ảnh', width=200, font=('Arial', 14), image=iconImage, compound=LEFT,
                     bg="#51aded", fg="white", command=openFile)
buttonImage.place(relx=0.5, rely=0.3, anchor=CENTER)

iconVideo = ImageTk.PhotoImage(Image.open("image/icons8-video-64.png").resize((40, 40)))
button2 = Button(window, text='Chọn video', width=200, font=('Arial', 14), image=iconVideo, compound=LEFT, bg="#51aded",
                 fg="white", command=lambda: choose_file(labelVideo))
button2.place(relx=0.5, rely=0.5, anchor=CENTER)

iconCamera = ImageTk.PhotoImage(Image.open("image/icons8-camera-64.png").resize((40, 40)))
button3 = Button(window, text='Chọn từ camera', width=200, font=('Arial', 14), image=iconCamera, compound=LEFT,
                 bg="#51aded", fg="white", command=open_camera)
button3.place(relx=0.5, rely=0.7, anchor=CENTER)

button4 = Button(window, text='Thoát', width=200, font=('Arial', 14), command=window.quit, image=img, compound=LEFT,
                 bg="red", fg="white")
button4.place(relx=0.5, rely=0.9, anchor=CENTER)

# start button close, save in screen camera
buttonSave = Button(fromCamera, text='Lưu ảnh', width=130, font=('Arial', 12), image=iconSave, compound=LEFT,
                    bg="#51aded", fg="white", command=lambda: capture_image(vid))
buttonSave.grid(row=1, column=0, sticky=E, padx=10, pady=10)


def close_window():
    global after_id
    fromCamera.withdraw()
    window.deiconify()
    labelCamera.after_cancel(after_id)


buttonCancel = Button(fromCamera, text='Hủy', width=130, font=('Arial', 12), image=iconCancel, compound=LEFT,
                      bg="#ed5153", fg="white", command=close_window)
buttonCancel.grid(row=1, column=1, sticky=W, padx=10, pady=10)

# end camera


def close_windowVideo():
    global after_idVideo
    fromVideo.withdraw()
    window.deiconify()
    labelVideo.after_cancel(after_idVideo)


buttonCancel = Button(fromVideo, text='Hủy', width=130, font=('Arial', 12), image=iconCancel, compound=LEFT,
                      bg="#ed5153", fg="white", command=close_windowVideo)
buttonCancel.grid(row=1, column=0, columnspan=2)


window.mainloop()
