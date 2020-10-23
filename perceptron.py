# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class PerceptronOrigin():
    '''
    感知机算法的原始形式
    '''
    def __init__(self, x, y, a=1):
        self.x=x
        self.y=y
        # 初始化模型参数 w 和 b （这里将它们都设为0）
        self.w=np.zeros(x.shape[1])
        self.b=0
        # 学习率
        self.a=a

    def train(self):
        '''
        找出不会误分类任何训练样本的超平面
        :return: w, b
        '''
        error_exist=True
        current=0
        index_point=0
        while error_exist:
            # 如果存在误分类点，更新参数
            if (np.dot(self.x[current],self.w)+self.b)*self.y[current]<=0:
                index_point=current
                self.w=np.add(self.w,self.a*self.y[current]*self.x[current])
                self.b+=self.a*self.y[current]
            else:
                # 当前误分类点被正确分类后，从该点开始遍历所有点
                current=(current+1)%self.x.shape[0]
                # 直到索引再次回到该点，表示已无误分类点
                if current==index_point:
                    error_exist=False
        return self.w,self.b

    def predict(self, x):
        '''
        对测试样本进行预测
        :param x: input(feature) space
        :return: output label (+1 or -1)
        '''
        y=np.dot(self.w,x)+self.b
        return int(y)

class PerceptronDual():
    def __init__(self, x, y, a=1):
        self.x=x
        self.y=y
        self.w=np.zeros(x.shape[1])
        self.eta=np.zeros(x.shape[0])
        self.b=0
        self.a=a

    def get_Gram(self):
        '''
        计算 Gram 矩阵
        '''
        x=self.x.copy()
        G=x.dot(x.T)
        return G

    def train(self):
        G=self.get_Gram()
        error_exist = True
        current = 0
        index_point = 0
        while error_exist:
            if self.y[current]*(sum((self.eta[i]*self.y[i]*(G[current][i]+1)) for i in range(self.y.shape[0]))) <=0:
                ''' 注意这个判断公式 '''
                index_point = current
                self.eta[current]+=self.a
                self.b+=self.a*self.y[current]
            else:
                current=(current+1)%self.x.shape[0]
                if current==index_point:
                    error_exist=False

        for i in range(self.y.shape[0]):
            ''' 用公式计算出参数 w '''
            self.w=np.add(self.w,(self.eta[i]*self.x[i]*self.y[i]))
        return self.w,self.b

    def predict(self, x):
        y=np.dot(self.w,x)+self.b
        return int(y)

class Show():
    '''
    模型效果的可视化展示
    '''
    def __init__(self, w, b, x, y):
        self.x=x
        self.y=y
        self.w=w
        self.b=b

    def get_y(self, x):
        y=-(self.w[0]*x+self.b)/self.w[1]
        return y

    def draw_pic(self):
        plt.figure()
        x=np.linspace(0,5)
        plt.plot(x,self.get_y(x))
        for i in range(self.y.shape[0]):
            # 正类的样本点为红色，负类为蓝色
            if self.y[i]==1:
                plt.scatter(self.x[i,0],self.x[i,1],c='r')
            else:
                plt.scatter(self.x[i, 0], self.x[i, 1], c='b')
        plt.xlabel("$x^{(1)}$")
        plt.ylabel("$x^{(2)}$")
        plt.show()

if __name__ == '__main__':
    x=np.asarray([(3,3),(4,3),(1,1)])
    y=np.asarray([1,1,-1])

    ''' 原始形式 '''
    perceptron1=PerceptronOrigin(x,y)
    w1,b1=perceptron1.train()
    print("The parameter w is",w1,", b is",b1)

    pic=Show(w1,b1,x,y)
    pic.draw_pic()

    ''' 对偶形式 '''
    perceptron2=PerceptronDual(x,y)
    w2,b2=perceptron2.train()
    print("The parameter w is", w2, ", b is", b2)

    pic = Show(w2, b2, x, y)
    pic.draw_pic()