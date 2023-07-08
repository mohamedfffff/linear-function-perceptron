from random import random
import math
import numpy as np
from turtle import *

sessions = 10
learning_rate = 0.01
width = 600
height = 600
input_number = 2
points_number = 100
points = []
answers = []

class Perceptron:
    # Initializing weights and bias for perceptron
    def __init__(self,n):
        self.weights = np.zeros(n)
        self.bias = 0
    #guessing the answer
    def predict(self,input):
        #calcuate perceptron output
        output = np.dot(input, self.weights) + self.bias
        #passing output through activation function
        predict = self.activation(output)
        return predict
    # sigmoid activation function
    def activation(self, z):
        return np.sign(z) #1 / (1 + np.exp(-z))
    #calcuate the accuracy of the perceptrom predictions 
    def accuracy(self,predict,answer):
        correct = 0
        for i in range(len(answer)):
            if(predict[i] == answer[i]):
                correct += 1
        return correct*100/len(answer)
    #training
    def train(self, X, Y):
        #train how many times in sessions
        for iteration in range(sessions):
            #drawing guessed points in this training session
            guess = self.predict(X)
            draw_points(guess)
            update()
            # Traversing through the entire input set
            for i in range(len(X)):
                #calculate differance between prediction and answer
                error = Y[i]-guess[i]
                #Updating weights and bias
                self.weights += np.dot(learning_rate*error , X[i])
                self.bias += learning_rate * error
            #print accuracy percentage
            print(str(self.accuracy(guess,answers))+"%")
            
def create_points(n):
    for i in range(n):
        #generate points with random x and y
        x = np.random.uniform(-1,1)
        y = np.random.uniform(-1,1)
        points.append([x,y])
        #generate answers
        #down the function line
        if(formula(x*width/2)>=y*width/2):
            answers.append(1)
        #on the function line
        elif(formula(x*width/2)==y*width/2):
            answers.append(0)
        #up the function line
        else:answers.append(-1)

def formula(x):
    #weights use * and bias use +
    # y = -(x-100)**2 /100 +100
    # y = 2*x-100
    y = -3*x+100
    # y = 3*x+100
    # y = x+100
    # y = x-100
    # y = x
    # y = -x
    # y = 100 
    # y = 0 
    # y = -100
    return y

def draw_setup():
    #setup 
    setup(width,height)
    tracer(0)
    hideturtle()
    bgcolor("white")
    #central dot
    color("purple")
    dot()
    #draw horizontal line
    color("black")
    penup()
    goto(-width/2,0)
    pendown()
    forward(width)
    #draw vertical line
    penup()
    setheading(90)
    goto(0,-height/2)
    pendown()
    forward(height)
    penup()
    #draw the function line
    draw_function_line()

def draw_function_line():
    #going to graph
    penup()
    color("red")
    goto(-width/2,formula(-width/2))
    pendown()
    #drawing thefunction
    for i in range(int(-width/2),int(width/2)):
        goto(i,formula(i))
    penup()

def draw_points(prediction):
    #draw the points
    for i in range(len(points)):
        x,y = points[i]
        #select color based on position according to function line
        if(prediction[i]>=0):
            #bellow
            color("green")
        else:
            #above
            color("blue")
        goto(x*width/2,y*height/2)
        dot(15)



create_points(points_number)
draw_setup()
p = Perceptron(2)
p.train(points,answers)
mainloop()