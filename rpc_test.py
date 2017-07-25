# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:30:58 2017

@author: huangshizhi

https://docs.python.org/3/library/xmlrpc.server.html
http://developer.51cto.com/art/201407/446192.htm
python2对应的包为 xmlrpclib
python3对应的包为 xmlrpc

客户端代码

import xmlrpc.client
s = xmlrpc.client.ServerProxy('http://192.168.62.59:8000')
print(s.add(3,2))
print(s.subtract(3,2))
print(s.multiply(3,2))
print(s.divide(5,2))

"""
import jieba
import jieba.posseg
import jieba.analyse
import  time  
import re
from xmlrpc.server import SimpleXMLRPCServer
#from xmlrpc.server import SimpleXMLRPCRequestHandler

def add(x,y):   
    return x+y   
    
def subtract(x, y):   
    return x-y   
    
def multiply(x, y):   
    return x*y   
    
def divide(x, y):   
    return x/y  
    
def cut_words(obj):
    t1 = time.time()
    cut_words_list = []
    for row in obj:
        re_row=re.sub('(\d+[年月日])|(\d{11})','',row.strip())  #包含****年月替换为空
        wordlist = jieba.cut(re_row)
        cut_word_str =  '|'.join(wordlist)   #将列表生成器转为字符串
        cut_word= re.sub('_|\d{4,}|\d+[a-zA-Z]{1,}|\d(\.)[a-zA-Z0-9]{1,}|[a-zA-Z0-9]{20,}','',cut_word_str)
        cut_words_list.append(cut_word)
    t2 = time.time()
    print('切词耗时为:'+str(t2-t1)+'s')
    return cut_words_list    
    
def cut_str(obj):
    return  "|".join(jieba.cut(obj))  
    

# A simple server with simple arithmetic functions   
server = SimpleXMLRPCServer(("192.168.62.59", 8000))   #确定URL和端口
print ("Listening on port 8000..." )
server.register_multicall_functions()   
server.register_function(add, 'add')   #注册add函数
server.register_function(subtract, 'subtract')   
server.register_function(multiply, 'multiply')   
server.register_function(divide, 'divide')  
server.register_function(cut_words,'cut_words')

server.register_function(cut_str,'cut_str')

#server.register_function(predict_data,'predict_data')

server.serve_forever()#启动服务器,并使其对这个连接可用
  
  
