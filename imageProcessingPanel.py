#!/usr/bin/env python
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk , Image          # We need pillow to visualize image in tkinter in an easy way 
import cv2
import numpy as np

class image_processing():
    panelA = None # Initialize a variable where the original image is displayed
    panelB = None # Initialize a variable where the result image is displayed
    inv = np.array([]) # Initialize binary inverse image
    kernel = np.ones((5,5),np.uint8) # Kernel size initializing

    def __init__(self):
        self.create_root() 
        
        
    def select_image(self):
        self.path = filedialog.askopenfilename()     # open a file chooser dialog and allow the user to select an input image
        self.image = cv2.imread(self.path) # image reading
        self.image = cv2.resize(self.image, (500,500)) # resize the image to fit my frame
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) # change the color space to RGB because OpenCV read image in BGR
        self.img = self.image.copy() # making a copy of the image
        self.result = self.img # initializing the resulting image equal to the original image

        self.image = Image.fromarray(self.image)       # convert the images to PIL format...
        self.image = ImageTk.PhotoImage(self.image)    # ...and then to tkinter format
        
        # if the panel are None, initialize them
        if self.panelA is None:
            self.panelA = Label(image=self.image)         # the first panel will store our original image
            self.panelA.image = self.image
            self.panelA.pack(side=LEFT,padx=5)   # We need pack for localization
        else:     # otherwise, update the image panel
            self.panelA.configure(image=self.image)
            self.panelA.image = self.image
            
            
    # function to apply the resulting image to the panel B
    def Apply(self,res):
        res= Image.fromarray(res) # convert the images to PIL format...
        res = ImageTk.PhotoImage(res)    # ...and then to tkinter format
        # if the panel are None, initialize them
        if self.panelB is None:
            self.panelB = Label(image=res)          # while the second panel will store the resulting image
            self.panelB.image = res
            self.panelB.pack(side=RIGHT,padx=5)
        else:     # otherwise, update the image panel
            self.panelB.configure(image=res)
            self.panelB.image = res    
        
    
    # creating the root
    def create_root(self):
        self.root = Tk() # take an object from tkinter
        self.root.title("Image processing panal") # frame title
        self.root.geometry('1350x1024') # geometry of frame 

        #creating labelFrames
        self.labelframe_right= LabelFrame(self.root)
        self.labelframe_right.pack(side=RIGHT,fill=Y)


        self.labelframe_left= LabelFrame(self.root)
        self.labelframe_left.pack(side=LEFT,fill=Y,ipadx=10)


        #color_traking
        self.labelframe_widget = LabelFrame(self.labelframe_right, text="Color Tracking")
        self.label_widget=Label(self.labelframe_widget)
        self.labelframe_widget.pack(padx="10",pady="10")



        #Thresholding
        self.labelframe_widget1 = LabelFrame(self.labelframe_right, text="Thresholding")
        self.label_widget1=Label(self.labelframe_widget1)
        self.labelframe_widget1.pack(padx="10",pady="10")

        #Blurring
        self.labelframe_widget2 = LabelFrame(self.labelframe_right, text="Blurring")
        self.label_widget2=Label(self.labelframe_widget2)
        self.labelframe_widget2.pack(padx="10",pady="10",ipadx="20",ipady="20")

        #Morphological Transformation
        self.labelframe_widget3 = LabelFrame(self.labelframe_left, text="Transformation")
        self.label_widget3=Label(self.labelframe_widget3)
        self.labelframe_widget3.pack(padx="10",pady="10",ipadx=15)

        #Edge Detector
        self.labelframe_widget4 = LabelFrame(self.labelframe_left, text="Edge Detector")
        self.label_widget4=Label(self.labelframe_widget4)
        self.labelframe_widget4.pack(padx="10",pady="10",ipadx=15)        

        #radio_buttons        
        #radio_buttons for color tracking
        self.v =IntVar()
        self.v.set(1)
        self.radiobutton_widget1 = Radiobutton(self.labelframe_widget, text="None                ", variable=self.v, value=1)
        self.radiobutton_widget2 = Radiobutton(self.labelframe_widget, text="Blue                 ", variable=self.v, value=2)
        self.radiobutton_widget3 = Radiobutton(self.labelframe_widget, text="Green               ", variable=self.v, value=3)
        self.radiobutton_widget4 = Radiobutton(self.labelframe_widget, text="Red                  ", variable=self.v, value=4)
        self.radiobutton_widget1.pack(padx=10, pady=10)
        self.radiobutton_widget2.pack(padx=10, pady=10)
        self.radiobutton_widget3.pack(padx=10, pady=10)
        self.radiobutton_widget4.pack(padx=10, pady=10)




        #radio_buttons for Thresholding Adaptive Mean
        self.v1 =IntVar()
        self.v1.set(5)
        self.radiobutton_widget5 = Radiobutton(self.labelframe_widget1, text="None                  ", variable=self.v1, value=5)
        self.radiobutton_widget6 = Radiobutton(self.labelframe_widget1, text="Adaptive Gaussian", variable=self.v1, value=6)
        self.radiobutton_widget7 = Radiobutton(self.labelframe_widget1, text="Adaptive Mean   ", variable=self.v1, value=7)
        self.radiobutton_widget8 = Radiobutton(self.labelframe_widget1, text="Otsu                  ", variable=self.v1, value=8)
        self.radiobutton_widget9 = Radiobutton(self.labelframe_widget1, text="Binary                ", variable=self.v1, value=9)
        self.radiobutton_widget10 = Radiobutton(self.labelframe_widget1,text="Binary inverse    ", variable=self.v1, value=10)
        self.radiobutton_widget5.pack(padx=10, pady=10)
        self.radiobutton_widget6.pack(padx=10, pady=10)
        self.radiobutton_widget7.pack(padx=10, pady=10)
        self.radiobutton_widget8.pack(padx=10, pady=10)
        self.radiobutton_widget9.pack(padx=10, pady=10)
        self.radiobutton_widget10.pack(padx=10, pady=10)



        #radio_buttons for Blurring
        self.v2 =IntVar()
        self.v2.set(11)   
        self.radiobutton_widget11 = Radiobutton(self.labelframe_widget2, text="None               ",variable=self.v2, value=11)
        self.radiobutton_widget12 = Radiobutton(self.labelframe_widget2, text="Gaussain         ",variable=self.v2, value=12)
        self.radiobutton_widget13 = Radiobutton(self.labelframe_widget2, text="Median           ",variable=self.v2, value=13)
        self.radiobutton_widget14 = Radiobutton(self.labelframe_widget2, text="Averaging       ",variable=self.v2, value=14)
        self.radiobutton_widget11.pack( pady=10)
        self.radiobutton_widget12.pack( pady=10)
        self.radiobutton_widget13.pack( pady=10)
        self.radiobutton_widget14.pack( pady=10)        



        #radio_buttons for Morphological Transformation  
        self.v3 =IntVar()
        self.v3.set(15)
        self.radiobutton_widget15 = Radiobutton(self.labelframe_widget3, text="None      ", variable=self.v3, value=15)
        self.radiobutton_widget16 = Radiobutton(self.labelframe_widget3, text="Dilation   ", variable=self.v3, value=16)
        self.radiobutton_widget17 = Radiobutton(self.labelframe_widget3, text="Erosion   ", variable=self.v3, value=17)
        self.radiobutton_widget18 = Radiobutton(self.labelframe_widget3, text="Closing   ", variable=self.v3, value=18)
        self.radiobutton_widget19 = Radiobutton(self.labelframe_widget3, text="Opening", variable=self.v3, value=19)
        self.radiobutton_widget20 = Radiobutton(self.labelframe_widget3, text="Gradient  ", variable=self.v3, value=20)
        self.radiobutton_widget15.pack(padx=10, pady=10)
        self.radiobutton_widget16.pack(padx=10, pady=10)
        self.radiobutton_widget17.pack(padx=10, pady=10)
        self.radiobutton_widget18.pack(padx=10, pady=10)
        self.radiobutton_widget19.pack(padx=10, pady=10)
        self.radiobutton_widget20.pack(padx=10, pady=10)


        #radio_buttons for Edge Detector
        self.v4 =IntVar()
        self.v4.set(21)
        self.radiobutton_widget21 = Radiobutton(self.labelframe_widget4, text="None     ", variable=self.v4, value=21)
        self.radiobutton_widget22 = Radiobutton(self.labelframe_widget4, text="ScharrX  ", variable=self.v4, value=22)
        self.radiobutton_widget23 = Radiobutton(self.labelframe_widget4, text="ScharrY  ", variable=self.v4, value=23)
        self.radiobutton_widget24 = Radiobutton(self.labelframe_widget4, text="Laplacian", variable=self.v4, value=24)
        self.radiobutton_widget25 = Radiobutton(self.labelframe_widget4, text="Canny    ", variable=self.v4, value=25)
        self.radiobutton_widget26 = Radiobutton(self.labelframe_widget4, text="SobelX   ", variable=self.v4, value=26)
        self.radiobutton_widget27 = Radiobutton(self.labelframe_widget4, text="SobelY   ", variable=self.v4, value=27)
        self.radiobutton_widget21.pack(padx=10, pady=10)
        self.radiobutton_widget22.pack(padx=10, pady=10)
        self.radiobutton_widget23.pack(padx=10, pady=10)
        self.radiobutton_widget24.pack(padx=10, pady=10)
        self.radiobutton_widget25.pack(padx=10, pady=10)
        self.radiobutton_widget26.pack(padx=10, pady=10)
        self.radiobutton_widget27.pack(padx=10, pady=10)

        # scale_widget to canny thresholding value
        self.scale_widget = Scale(self.labelframe_left, label="Canny min thresh", from_=0, to=255,orient=HORIZONTAL)
        self.scale_widget.set(25)
        self.scale_widget.pack(side=BOTTOM, fill = "y", expand="yes", padx="10",ipadx=10)
        self.scale_widget1 = Scale(self.labelframe_left, label="Canny max thresh", from_=0, to=255,orient=HORIZONTAL)
        self.scale_widget1.set(25)
        self.scale_widget1.pack(side=BOTTOM, fill = "y",  expand="yes", padx="10",ipadx=10)
        
        
        #Buttons
        # select an image button 
        self.btn = Button(self.root, text="Select an image",command=self.select_image) # yourapp , text ,  binded function
        self.btn.pack(side="bottom", padx="10", pady="5") #  Localization
        
        # Apply button to apply result image
        self.btn2 = Button(self.root, text="Apply",command = self.get_value) # yourapp , text ,  binded function
        self.btn2.pack(side="bottom", padx="10", pady="5")
        
        # kick off the GUI
        self.root.mainloop()
    # function to get the radio buttons values     
    def get_value(self):
        self.color = self.v.get()
        self.Threhsold = self.v1.get()
        self.blurring = self.v2.get()
        self.Transformation = self.v3.get()
        self.Edge_Detector = self.v4.get()
        self.choice_list = [self.color,self.Threhsold,self.blurring,self.Transformation,self.Edge_Detector]
        self.execution(self.choice_list)
        self.result = self.img # Return the resulting image to the original after pressing the Apply button
        self.inv = np.array([]) # Return the binary inverse image to the empty NumPy array after pressing the Apply button
    
    #color tracking function
    def color_tracking(self,colors):
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)# change the color space to HSV 
        if colors == 2: 
            # define range of red color in HSV
            self.lower = np.array([90,50, 50])
            self.upper = np.array([120, 255,255])
            # Threshold the HSV image using inRange function to get only red colors
            self.mask = cv2.inRange(self.hsv, self.lower, self.upper)
            self.width = int(self.img.shape[1]) # image width
            self.height = int(self.img.shape[0]) # image height
            self.dim = (self.width, self.height) # image dimension
            self.mask = cv2.resize(self.mask,self.dim)# resize the mask as the image
            self.result = cv2.bitwise_and(self.img,self.img, mask= self.mask) # result of adding the original image and mask together
            self.Apply(self.result)# pass the result to apply function
        elif colors == 3:
            # define range of red color in HSV
            self.lower = np.array([36, 25, 25])
            self.upper = np.array([70, 255,255])
            # Threshold the HSV image using inRange function to get only red colors
            self.mask = cv2.inRange(self.hsv, self.lower, self.upper)
            self.width = int(self.img.shape[1])
            self.height = int(self.img.shape[0])
            self.dim = (self.width, self.height)
            self.mask = cv2.resize(self.mask,self.dim)
            self.result = cv2.bitwise_and(self.img,self.img, mask= self.mask)
          #  self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)
            self.Apply(self.result)
        elif colors == 4:  
            # define range of red color in HSV
            self.lower = np.array([160,50,50])
            self.upper = np.array([180,255,255])
            # Threshold the HSV image using inRange function to get only red colors
            self.mask = cv2.inRange(self.hsv, self.lower, self.upper)
            self.width = int(self.img.shape[1])
            self.height = int(self.img.shape[0])
            self.dim = (self.width, self.height)
            self.mask = cv2.resize(self.mask,self.dim)
            self.result = cv2.bitwise_and(self.img,self.img, mask= self.mask)
           # self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)
            self.Apply(self.result)
            
    # thresholding function
    def Thresholding_choice(self,thresh):
        if len(self.result.shape) == 3:
            self.gray = cv2.cvtColor(self.result, cv2.COLOR_BGR2GRAY) # Change the color space to gray if it isn't
        if thresh == 6:   
            self.result = cv2.adaptiveThreshold(self.gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)# adaptice gaussian
            self.Apply(self.result)
        elif thresh == 7:
            # adaptive mean
            self.result = cv2.adaptiveThreshold(self.gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) # image  ,  # max  ,# method  , # Threshold type  , # Block Size , # Constant
            self.Apply(self.result)
        elif thresh == 8:
            self.ret,self.result = cv2.threshold(self.gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)# otsu thresholing
            self.Apply(self.result)
        elif thresh == 9:
            self.ret,self.result = cv2.threshold(self.gray,127,255,cv2.THRESH_BINARY) # binary thresholding
            self.Apply(self.result)
        elif thresh == 10:
            self.ret,self.result = cv2.threshold(self.gray,127,255,cv2.THRESH_BINARY_INV)# binary inverse thresholding
            self.inv = self.result
            self.Apply(self.result)
   
    # blurring function        
    def blurred (self,blur):
        if blur == 12:
            self.result = cv2.GaussianBlur(self.result,(21,21),10)# Gaussian blurring
            self.Apply(self.result)
        elif blur == 13:
            self.result = cv2.medianBlur(self.result,7)# median blurring
            self.Apply(self.result)
        elif blur == 14:
            self.result = cv2.blur(self.result,(15,15))# Averaging blurring
            self.Apply(self.result)            
    
    # transformation function
    def transforming(self,transform):     
        if len(self.result.shape) == 3:
            self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2GRAY)# Change the color space to gray if it isn't
        if transform == 16:
            if self.inv.size == 0: # If the inv variable we initialized has a size of zero, then change the image to a binary inverse and then dilate it 
                self.ret,self.inv = cv2.threshold(self.result,127,255,cv2.THRESH_BINARY_INV)
                self.result = cv2.dilate(self.inv,self.kernel,iterations = 1)# Dilation
                self.Apply(self.result)
            else: # else dilate it directly
                self.result = cv2.dilate(self.result,self.kernel,iterations = 1)
                self.Apply(self.result)
                
        elif transform == 17:
            if self.inv.size == 0:
                self.ret,self.inv = cv2.threshold(self.result,127,255,cv2.THRESH_BINARY_INV)
                self.result = cv2.erode(self.inv,self.kernel,iterations = 1)# Erosion
                #  img , kernel , iterations
                self.Apply(self.result)
            else:
                self.result = cv2.erode(self.result,self.kernel,iterations = 1)#Erosion
                #  img , kernel , iterations
                self.Apply(self.result)
        elif transform == 18:
            if self.inv.size == 0:
                self.ret,self.inv = cv2.threshold(self.result,127,255,cv2.THRESH_BINARY_INV)
                self.result = cv2.morphologyEx(self.inv, cv2.MORPH_CLOSE, self.kernel) # Closing
                self.Apply(self.result)
            else:
                self.result = cv2.morphologyEx(self.result, cv2.MORPH_CLOSE, self.kernel) # Closing
                self.Apply(self.result)                
            
        elif transform == 19:
            if self.inv.size == 0:
                self.ret,self.result = cv2.threshold(self.result,127,255,cv2.THRESH_BINARY_INV)
                self.result = cv2.morphologyEx(self.result, cv2.MORPH_OPEN, self.kernel) # Openning
                self.Apply(self.result)
            else:
                self.result = cv2.morphologyEx(self.result, cv2.MORPH_OPEN, self.kernel) # Openning
                self.Apply(self.result)
            
        elif transform == 20:
            if self.inv.size == 0:
                self.ret,self.result = cv2.threshold(self.result,127,255,cv2.THRESH_BINARY_INV)
                self.result = cv2.morphologyEx(self.result, cv2.MORPH_GRADIENT, self.kernel) #Gradient # Dilation - Erosion
                self.Apply(self.result)
            else:
                self.result = cv2.morphologyEx(self.result, cv2.MORPH_GRADIENT, self.kernel) #Gradient  # Dilation - Erosion
                self.Apply(self.result)
                
    # Edge detection function            
    def edge_detect(self,edge):
        if len(self.result.shape) == 3:
            self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2GRAY)# Change the color space to gray if it isn't
        if edge == 22:
            self.result = cv2.Scharr(self.result,cv2.CV_64F,1,0,5) #Scharr edging  # img ,ddepth, dx , dy, kernel
            self.Apply(self.result)
        elif edge == 23:
            self.result = cv2.Scharr(self.result,cv2.CV_64F,0,1,5) #Scharr edging # img ,ddepth, dx , dy, kernel
            self.Apply(self.result)
        elif edge == 24:
            self.result = cv2.Laplacian(self.result,cv2.CV_64F,ksize=3) #Laplacian edging # img , ddepth , kernel
            self.Apply(self.result)
        elif edge == 25:
            self.thresh_min = self.scale_widget.get() # Canny minimum threshold 
            self.thresh_max = self.scale_widget1.get()# Canny minimum threshold
            self.result = cv2.Canny(self.result,self.thresh_min,self.thresh_max)# Canny edging # image , min threshold , max threshold
            self.Apply(self.result)
        elif edge == 26:
            self.result = cv2.Sobel(self.result,-1,1,0,ksize=5) # Sobel
            self.Apply(self.result)
        elif edge == 27:
            self.result = cv2.Sobel(self.result,-1,0,1,ksize=5)# Sobel
            self.Apply(self.result) 
            
    #   Execution function      
    def execution(self,choice):
        if choice == [1 , 5 , 11 , 15 , 21]: # if all the radio buttons are None ===> the resulting image equal the original one
            self.result = self.img 
            if self.panelB is None: # if the panels are None, initialize them
                self.panelB = Label(image=self.image)          # while the second panel will store the resulting image which is the original image here
                self.panelB.image = self.image
                self.panelB.pack(side=RIGHT,padx=5)
            else:  # otherwise, update the image panel
                self.panelB.configure(image=self.image)
                self.panelB.image = self.image 
        else: # otherwise, execute all the radio buttons
            self.color_tracking(choice[0])
            self.Thresholding_choice(choice[1])
            self.blurred (choice[2])
            self.transforming(choice[3])
            self.edge_detect(choice[4])
                   
exe = image_processing() # object from the image_processing class
exe
