"""
Designed and developed by: Nika Abedi - contact email: nikka.abedi@gmail.com
*****************************************************************************
NOT SPARSE IMAGE RECONSTRUCTION (NSIR) Class
*****************************************************************************
The goal of this class is to reconstruct the input image based on a combination of dictionary element and corresponding
coefficients. However, the feature vector is not considered to be sparse. In fact, it represents a good example of
developing a gradient based solution for a minimization problem. The requisite information about variables which has
been used by the class, is provided as follows:

y: n by 1 vector - the main output or image
Dic: n by m dictionary
x: m by 1 vector
alpha : defining the step size in optimization based on x, set to 0.000000009 by default
alphaD: defining the step size in optimization based on Dictionary , set to 0.01 by the default
yDim: dimension of y
epoch: number of iterations

"""
import numpy as np
from skimage import io
from matplotlib import pyplot as plt


class NSIR:

    # -------------------------------------------------------------
    # Class constructor
    # -------------------------------------------------------------
    def __init__(self):
        self.y = np.array([])
        self.Dic = np.array([])
        self.x = np.array([])
        self.alpha = 0.000000009
        self.alphaD = 0.01
        self.yDim = np.array([])
        self.epoch = [200, 1]
        # Read the input image and converted to a vector in grayscale level
        self.y = io.imread('C:\\Users\\NIka\\Desktop\\0.jpg', as_gray=True)
        self.yDim = self.y.shape
        self.y = np.array(self.y).flatten()
        self.y = np.reshape(self.y, [self.y.size, 1])
        self.x = np.random.rand(10, 1)
        self.Dic = np.random.rand(self.y.size, self.x.size)

    # -------------------------------------------------------------
    # Returns y dimensions
    # -------------------------------------------------------------
    def getYsize(self):
        return self.y.shape

    # -------------------------------------------------------------
    # Returns x dimensions
    # -------------------------------------------------------------
    def getXsize(self):
        return self.x.shape

    # -------------------------------------------------------------
    # Returns Dictionary size
    # -------------------------------------------------------------
    def getDicsize(self):
        return self.Dic.shape

    # -------------------------------------------------------------
    # Returns input size
    # -------------------------------------------------------------
    def inputfile(self):
        return self.y

    # -------------------------------------------------------------
    # Returns x size
    # -------------------------------------------------------------
    def creatX(self):
        return self.x

    # -------------------------------------------------------------
    # Returns dictionary size
    # -------------------------------------------------------------
    def createDic(self):
        return self.Dic

    # -------------------------------------------------------------
    # Calculates the gradient of Dictionary
    # -------------------------------------------------------------
    def gradD(self, D):
        dx = (-2 * np.matmul(self.x, (self.y - np.matmul(D, self.x)).T)).T
        return dx

    # -------------------------------------------------------------
    # Calculates the gradient of x
    # -------------------------------------------------------------
    def gradX(self, X):
        dx = -2 * np.matmul(self.Dic.T, (self.y - np.matmul(self.Dic, X)))
        return dx

    # -------------------------------------------------------------
    # Objective function
    # -------------------------------------------------------------
    def objfunc(self, input, flag):
        if flag == 0:
            # calculates the objective function with the gradient of x
            obj = np.sum(np.power(self.y - np.matmul(input, self.x), 2))
        elif flag == 1:
            # calculates the objective function with the gradient of Dic
            obj = np.sum(np.power(self.y - np.matmul(self.Dic, input), 2))
        else:
            # calculates the objective function with the both gradient of x and Dic
            obj = np.sum(np.power(self.y - np.matmul(self.Dic, self.x), 2))

        return obj

    # -------------------------------------------------------------
    # Minimization based on Dictionary
    # -------------------------------------------------------------
    def optDic(self):
        for i in range(self.epoch[0]):
            d_new = self.Dic - (self.alphaD * self.gradD(self.Dic))
            if self.objfunc(d_new, 0) <= self.objfunc(self.Dic, 0):
                self.Dic = d_new

            # print("epoch= ", i, "obj", self.objfuncDchange(self.Dic))

    # -------------------------------------------------------------
    # Minimization based on x
    # -------------------------------------------------------------
    def optX(self):
        for i in range(0, self.epoch[0]):
            x_new = self.x - (self.alpha * self.gradX(self.x))
            if self.objfunc(x_new, 1) <= self.objfunc(self.x, 1):
                self.x = x_new

            # print("epoch= ", i, "obj", self.objfuncXchange(self.x))

    # -------------------------------------------------------------
    # The main minimization method
    # -------------------------------------------------------------
    def minimization(self):
        for i in range(self.epoch[1]):
            self.optDic()
            self.optX()
            print("epoch= ", i, "obj", self.objfunc(None, 3))

    # -------------------------------------------------------------
    # Visualizing the outputs
    # -------------------------------------------------------------
    def visual(self):
        plt.imshow(np.reshape(self.y, [self.yDim[0], self.yDim[1]]))
        plt.imshow(np.reshape((np.matmul(self.Dic, self.x)), [self.yDim[0], self.yDim[1]]))

