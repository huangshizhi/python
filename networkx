# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:23:50 2017

@author: huangshizhi
安装方式pip installl networkx
参考文献

http://networkx.github.io/examples.html
http://www.cnblogs.com/kaituorensheng/p/5423131.html
https://www.douban.com/note/331796003/
"""

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_node(1)
G.add_edge(2,3)
G.add_edge(3,2)

print("nodes:",G.nodes())
print("edges:",G.edges())
print("nubmer of edges:",G.number_of_edges())
nx.draw(G,with_labels=True)
#plt.savefig("wuxiangtu.jpg")
plt.show()


G2 = nx.DiGraph()
G2.add_node(1)
G2.add_node(2)                  #加点
G2.add_nodes_from([3,4,5,6])    #加点集合
G2.add_cycle([1,2,3,4])         #加环
G2.add_edge(1,3)     
G2.add_edges_from([(3,5),(3,6),(6,7)])  #加边集合

nx.draw(G2,with_labels=True)
plt.savefig("D:\天池\贵州交通\youxiangtu.png")
plt.show()
