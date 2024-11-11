import cv2 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import cvlib as cv
from PIL import Image
from sklearn.cluster import KMeans
import webcolors
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image,ImageTk

def read_image(path):
    image = cv2.imread(path)  # Replace 'image.jpg' with your image path
    # Convert the image from BGR to RGB (OpenCV loads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2. resize(image,(300,300))
    face ,con = cv.detect_face(image)
    try:
        x,y,x_end,y_end = face[0]
    except:
        tk.messagebox.showinfo('Error' ,"No face detected.")
        return
    
    clr_image = image[y:y_end,x:x_end]
    try:
        clr_image = cv2.resize(clr_image,(110,110))
    except:
         tk.messagebox.showinfo('Error' ,"No dress detected.")
         return
    region,age = nation(clr_image)

    gray_img = image[y:y_end,x:x_end]
    gray_img = cv2.cvtColor(gray_img,cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img,(48,48))
    expression = emotion(gray_img)


    shape = image.shape
    y2 =y_end
    y2  += round((shape[0]-y_end)*50)
    x2 = x_end
    x2 += round((shape[1]-x_end)*0.50)
    image1 = image[y_end:y2,x-50:x2]
    image1 = cv2.resize(image1,(48,48))

    pixels = image1.reshape((-1, 3))
    k = 3  # Number of clusters (You can adjust this)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color_index = labels[np.argmax(counts)]
    dominant_color = dominant_colors[dominant_color_index]

    color_name = get_closest_color_name(dominant_color)
    text = decision(region,age,expression,color_name)
    
    return text



def nation(img):
    pred = np.array(img).astype(np.float32)/255.0
    pred = np.expand_dims(pred,axis=0)
    model = load_model('Nationality.keras')
    pred = model.predict(pred)
    cls = {0:'American',1:'African',2:'Asian',3:'Indian',4:'others'}
    region = cls[np.argmax(pred[0])]
    age = round(pred[1][0][0])
    return region,age



def emotion(img):
    pred = np.expand_dims(np.array(img).astype(np.float32)/255.0,axis=0)
    model = load_model('Emotion.keras')
    pred = model.predict(pred)
    cls = {0:'angry', 1:' fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}
    exp = cls[np.argmax(pred[0])]
    return exp


def get_closest_color_name(rgb_tuple):
    try:
        # Try to get the exact color name
        color_name = webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        # If the exact name is not found, get the closest color name
        color_name = get_closest_web_color_name(rgb_tuple)
    return color_name

def get_closest_web_color_name(rgb_tuple):
    min_diff = float('inf')
    closest_color = None
    
    for hex_color, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r, g, b = webcolors.hex_to_rgb(hex_color)
        diff = ((r - rgb_tuple[0]) ** 2 + (g - rgb_tuple[1]) ** 2 + (b - rgb_tuple[2]) ** 2) ** 0.5
        if diff < min_diff:
            min_diff = diff
            closest_color = name
    
    return closest_color


def decision(region,age,expression,color_name):
    if region == 'American':
        return f'Nationality: {region}\n Age: {age}\n Emotion: {expression}'
    elif region =='African':
        return f'Nationality: {region}\n Emotion: {expression}\n Dress_color: {color_name} '
    elif region =='Indian':
        return f'Nationality: {region}\n Age: {age}\n Emotion: {expression}\n Dress_color: {color_name}'
    else:
        return f'Nationality: {region}\n Emotion: {expression}'

def image_gui():
      
    win = tk.Tk()
    win.geometry('600x600')
    win.config(background='purple')
    win.title('Nationality Detection')
    global label,frame,label1,label2

    label = tk.Label(win,width=30,height=15,background='lightgray')
    label.pack(side='left',padx=10,pady=10)
     
    frame = tk.Frame(win,width=500,height=500,background='lightgray')
    frame.pack_propagate(False)
    frame.pack(padx=10,pady=20)

    label1 = tk.Label(frame,width=300,height=300,bg='lightgray')
    label1.pack_propagate(False)
    label1.pack(side='top',padx=10,pady=10)

    label2 = tk.Label(frame,text='None!',width=50,height=50,bg='lightgray',compound='bottom')
    label2.pack_propagate(False)
    label2.pack(side='bottom',padx=10,pady=10)

    def open_file_dialog():
        # Open the file dialog
        global file_path
        file_path = tk.filedialog.askopenfilename(
            title="Select a File",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]  # File type filters
        )
        # Show a message with the selected file path 
        if file_path:
            images(file_path)

            clicked.set('Custom Images')
        else:
            tk.messagebox.showinfo("File Selection Cancelled", "No file selected.")

    def get_selected_option():
        selected_option = clicked.get()
        if selected_option:
            
            images(selected_option)
        else:
            tk.messagebox.showinfo("File Selection Cancelled", "No file selected.")


    def images(filepath):
           
        t = read_image(filepath)
        img1 = Image.open(filepath)
        img1 = img1.resize((300,300))
        img1 = ImageTk.PhotoImage(img1)

        label1.config(image=img1,bg='white')
        label1.image = img1
        label2.config(text=t,font=('italic',15,'bold'),bg='white',fg='purple')
    
    
    def win_close():
        win.destroy()

    clicked = tk.StringVar()
    clicked.set('Custom Images')
    options = 'image.jpg image1.jpg image2.jpg image3.jpg image4.jpg image5.jpg image6.jpg'.split()

    # Create OptionMenu (Dropdown)
    dropdown = tk.OptionMenu(label, clicked, *options)
    dropdown.config(fg='white',bg='purple',relief='raised',bd=5)

    # Add a button to display the selected option
    button = tk.Button(label, text="Show Selected")
    button.config(fg='white',bg='purple',command=get_selected_option,relief='raised',bd=5)

    button1 = tk.Button(label,
                    text='choose file')
    button1.config(command=open_file_dialog,fg='white',bg='purple',relief='raised',bd=5)

                    
    button1.grid(row=0, column=0, padx=10, pady=10)  # First button in row 0, column 0
    dropdown.grid(row=1, column=0, padx=10, pady=10)  # Second button in row 0, column 1
    button.grid(row=2, column=0, padx=10, pady=10)

    off = tk.Button(win,text='Exit',font=('arial',10),bg = 'white',fg='purple',command=win_close,relief='ridge')
    off.pack()
    
    win.mainloop()
image_gui()
