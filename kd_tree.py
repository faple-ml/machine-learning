# -*- coding: UTF-8 -*-
from math import sqrt

class KDNode():
    ''' kd树节点的数据结构 '''
    def __init__(self, root_node:list, depth:int, left, right):
        self.root_node=root_node # 父节点
        self.depth=depth # 深度
        self.left=left # 左子节点
        self.right=right # 右子节点

class KDTree():
    def __init__(self, xs:list):
        self.xs=xs
        self.k=len(xs[0]) # 特征空间的维度
        self.root=None

    def create_kd_node(self, xs:list, depth:int):
        ''' 递归生成子节点 '''

        ''' 如果数据集为空，递归停止 '''
        if len(xs)==0:
            return None

        xs.sort(key=lambda x:x[depth]) # 以坐标 x^(depth) 排序
        split=len(xs)//2 # 找到中位数的索引
        median=xs[split] # 中位数为当前子节点
        next_depth=(depth+1)%self.k # kd树深度+1，即当前子节点的深度

        return KDNode(median,depth,self.create_kd_node(xs[:split],next_depth),self.create_kd_node(xs[split+1:],next_depth))

    def build_kd_tree(self):
        ''' 从根节点开始生成kd树 '''
        self.root=self.create_kd_node(self.xs,0)
        return self.root

    def pre_order(self, root:KDNode):
        ''' 树的前序遍历 '''
        print(root.root_node)
        if root.left:
            self.pre_order(root.left)
        if root.right:
            self.pre_order(root.right)

    def find_nearest_point(self, x:list, node:KDNode, min_dist:float):
        ''' 递归搜索最近邻点 '''

        if node==None:
            # 当前最近点
            return [0]*self.k,float("inf")

        depth=node.depth # 当前维度
        s=node.root_node # 当前超平面（即切分点，当前子树的根节点）
        ''' 当前维度小于切分点，去左子树，大于去右子树 '''
        if x[depth]<s[depth]:
            next_node=node.left
            next_node_cousin=node.right # 兄弟节点
        else:
            next_node=node.right
            next_node_cousin = node.left

        ''' 从叶节点递归向上每个节点 '''
        nearest,dist=self.find_nearest_point(x,next_node,min_dist)
        if dist<min_dist:
            min_dist=dist # 最近点在以目标点为球心，min_dist为半径的超球体中

        s_dist=abs(s[depth]-x[depth]) # 第depth维目标点到超平面的距离
        if min_dist>s_dist:
            ''' 如果超球体与超平面相交 '''
            s_dist=sqrt(sum((p1-p2)**2 for p1,p2 in zip(s,x)))
            # 计算目标点与切分点的欧氏距离
            # 如果更近，则更新最近点
            if s_dist<dist:
                nearest=s
                dist=s_dist
                min_dist=dist
            ''' 检查兄弟节点对应区域是否有更近的点 '''
            nearest_cousin,cousin_dist=self.find_nearest_point(x,next_node_cousin,min_dist)
            if cousin_dist<dist:
                nearest=nearest_cousin
                dist=cousin_dist

        return nearest,dist # 返回最近邻点和最小距离

    def find_nearest(self, x:list, root:KDNode):
        ''' 搜索最近邻点（从根节点开始） '''
        return self.find_nearest_point(x,root,float("inf"))

if __name__ == '__main__':
    x=[[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    kd_tree=KDTree(x)
    kd_root=kd_tree.build_kd_tree()
    kd_tree.pre_order(kd_root)
    nearest,dist=kd_tree.find_nearest([3,4.5],kd_root)
    print("The nearest point is",nearest,", and their distance is",dist)