from face_detection import create_video_output
from face_detection import create_webcam_output
from face_detection import create_image_output
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import shutil

root = Tk()
root.title('Sentiment Analysis')
root.geometry('500x400+500+500')
root.resizable(0, 0)

def open_file(label=0):
    if label == 1:
        create_webcam_output()
    else:
        file_paths = filedialog.askopenfilenames(parent=root, initialdir=os.getcwd(), title='Please Select a file', filetypes=[('All files', '.*')])
        print(file_paths)
        if label == 2:
            if len(file_paths) == 0:
                create_image_output()
            else:
                create_image_output(file_paths)
        elif label == 3:
            dirpath = './video_data/frames/'
            if len(file_paths) == 0:
                create_video_output()
            else:
                for filename in os.listdir(dirpath):
                    frame_path = os.path.join(dirpath, filename)
                    try:
                        shutil.rmtree(frame_path)
                    except OSError:
                        os.remove(frame_path)
                create_video_output(file_paths)



background_img = Image.open('Sentiment_background.jpg')
background_tk = ImageTk.PhotoImage(background_img)
label_back = Label(root, image=background_tk)
label_back.pack()

photo = PhotoImage(file='Danger_Noodle.jpg')
root.iconphoto(False, photo)

btnImage = Button(root, text="Image", width=10, command=lambda: open_file(2))
btnImage.place(x=100, y=300)
btnVideo = Button(root, text="Video", width=10, command=lambda: open_file(3))
btnVideo.place(x=200, y=300)
btnWebcam = Button(root, text="Webcam", width=10, command=lambda: open_file(1))
btnWebcam.place(x=300, y=300)
btnExit = Button(root, text="Exit", width=10, command=root.destroy)
btnExit.place(x=200, y=350)

team_name = "Danger Noodles"
label1 = Label(root, text=team_name)
label1.config(width=200, font=('Courier', 20))
label1.pack()



root.mainloop()


