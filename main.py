from sklearn import datasets,linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
import tkinter as tk
from tkinter import ttk
import tkinter
from DropDown_file import *
from tkinter import *
import cv2
import PIL.Image,PIL.ImageTk
import imutils
from functools import partial
from datetime import date


linearRegressions=[]
CarsDetail=[]
CarNames=[]

# answer

cv_img=cv2.cvtColor(cv2.imread('Car_Background.jpeg'),cv2.COLOR_BGR2RGB)

class CarsRegression:
    def __init__(self,name,LinRegreModel):
        self.name=name
        self.LinearRegression=LinRegreModel

class CarsData:
    def __init__(self,name,df):
        self.name=name
        self.DataFrame=df

path=os.getcwd()

path+='/Cars'

files=os.listdir(path)

# For extracting all the file from the Cars folder with xlsx extention
xlsx_Files=[f for f in files if f[-4:]=='xlsx']

# For labeling all the object with file name and linear regression model
for file in xlsx_Files:
    linearRegrName=file[:-5]
    RegresserModel=linear_model.LinearRegression()
    
    model=CarsRegression(linearRegrName,RegresserModel)

    linearRegressions.append(model)

# Making a db for the dataframes for the with it's car name 
for file in xlsx_Files:
    df=pd.read_excel('./Cars/'+file)
    CarsDetail.append(CarsData(file[:-5],df))
    CarNames.append(file[:-5])

for i in range(len(linearRegressions)):
    df=CarsDetail[i].DataFrame
    train_X=df[['KM','Model','Fuel (P:0 , D:1)','Condition(Denting/Screches)']]
    train_Y=pd.DataFrame(df['Price'])
    model=linearRegressions[i].LinearRegression
    model.fit(train_X,train_Y)


def number(km,sCar,score,fuel_input,model,answer):
    try:
        KMs=int(km.get())
        score=int(score.get())
        fuel=int(fuel_input.get())
        modelYear=int(model.get())

        # print(KMs,score,fuel,modelYear)

        if KMs<0:
            answer.config(text='Please enter the +ve KMs')
        elif score<0 or score>10:
            answer.config(text='Please enter the Codition score between 0 and 10!')
        elif fuel<0 or fuel>1:
            answer.config(text='Please enter 0 or 1 for fuel type!')
        elif modelYear>date.today().year:
            answer.config(text='Pleae enter a valid year!')
        else:
            answer.config(text='Calculating the price......')
            answer.config(text=sCar.get())
            Car=sCar.get()
            
            print(Car)

            index=0
            for cars in CarNames:
                index+=1
                if cars==Car:
                    print('Yes i found it!')
                    break

            index=index-1

            print(CarNames[index])

            prediction=[]
            prediction.append(KMs)
            prediction.append(modelYear)
            prediction.append(fuel)
            prediction.append(score)

            print(prediction)

            rModel=linearRegressions[index].LinearRegression

            pPrice=rModel.predict([prediction])[0][0]
            Price=str(round(pPrice,2))
            answer.config(text='Predicted Price: Rs '+Price)
        
    except :
        answer.config(text='Please enter the right details!')
        


def test(test_list):
        """Run a mini application to test the AutocompleteEntry Widget."""
        root = Tk(className='AutocompleteEntry demo')
        root.geometry('800x695')
        canvas=Canvas(root,width=800,height=380)

        # bg=PhotoImage(file='Car_Background.jpeg')

        photo=PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
        rimage=imutils.resize(cv_img,width=800,height=380)
        photo=PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(rimage))
        image_on_canvas=canvas.create_image(0,0,ancho=tkinter.NW,image=photo)
        canvas.pack()

        carName=Label(root,text='Enter the carName:')
        carName.pack(pady=2)

        combo = AutocompleteCombobox(root)
        combo.set_completion_list(test_list)
        combo.pack()
        # combo.default(0)fuel_input
        combo.focus_set()

        KmText=Label(root,text='Enter the KMs:')
        KmText.pack(pady=1)

        km=Entry(root)
        km.pack(pady=1)

        score_txt=Label(root,text='Enter the condition score(dents,\nEngine Condintion...):')
        score_txt.pack(pady=1)

        score=Entry(root)
        score.pack(pady=1)
        
        fuel=Label(root,text='Enter the fuel type 0 for Petrol,1 for Diesel:')
        fuel.pack(pady=1)

        fuel_input=Entry(root,text='Fuel type')
        fuel_input.pack(pady=1)

        model_txt=Label(root,text='Enter the car model Year')
        model_txt.pack(pady=1)

        model=Entry(root)
        model.pack(pady=1)

        answer=Label(root,text='Predicted price:')
        answer.pack(pady=1)
        
        my_button=Button(root,text='Calculate Price',
        command=partial(number,km,combo,score,fuel_input,model,answer))
        my_button.pack(pady=1)

        # entry = AutocompleteEntry(root)
        # entry.set_completion_list(test_list)
        # entry.pack()
        # entry.focus_set()

        # I used a tiling WM with no controls, added a shortcut to quit
        root.bind('<Control-Q>', lambda event=None: root.destroy())
        root.bind('<Control-q>', lambda event=None: root.destroy())
        root.mainloop()

if __name__ == '__main__':
    # test_list = ['apple', 'banana', 'CranBerry', 'dogwood', 'alpha', 'Acorn', 'Anise']
    test(CarNames)