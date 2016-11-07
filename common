# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:47:27 2016

@author: huangshizhi

解析xml文件
"""

from xml.dom import minidom
import pymysql
import pandas as pd
import time
def get_attrvalue(node, attrname):
     return node.getAttribute(attrname) if node else ''

'''
或者为 if node
return node.childNodes[index].nodeValue if node else ''
'''
def get_nodevalue(node, index = 0):
    return node.childNodes[index].nodeValue if node.childNodes!=[] else ''

def get_xmlnode(node, name):
    return node.getElementsByTagName(name) if node else []

def get_xml_data(filename):
    start_time = time.time()
    doc = minidom.parse(filename) 
    end_time = time.time()
    print("加载时间为："+str(end_time-start_time))
    root = doc.documentElement

    user_nodes = get_xmlnode(root, 'ExternalPage')
    #print ("ExternalPage:", user_nodes)   
    #unodes = user_nodes[97608:97610]
    for node in user_nodes:
        url = get_attrvalue(node, 'about') 
        node_topic = get_xmlnode(node, 'topic')
        node_title = get_xmlnode(node, 'd:Title')
        node_description = get_xmlnode(node, 'd:Description')
        if(len(node_topic)>0 and len(node_title)>0 and len(node_description)>0):        
            topic =get_nodevalue(node_topic[0])
            title = get_nodevalue(node_title[0])
            description = get_nodevalue(node_description[0])
        else:
            topic=''
            title=''
            description=''
        item = {}
        item['url'] , item['topic'] , item['title'] , item['description']  = (
            url,topic,title,description 
        )
        item_list.append(item)
    return item_list

if __name__ == "__main__":
    start_time = time.time()
    filename = r"E:\worksapce\实用\common\test2.xml"
    item_list=[]
    item_list = get_xml_data(filename)
    end_time = time.time()
    print("加载时间:"+str(end_time-start_time))
    pdata = pd.DataFrame(item_list,columns=['url','topic','title','description'])
    conn = pymysql.connect(host="localhost",user="root",passwd="123456",db="test", port=3306,charset='utf8')
    insert_time = time.time()    
    pdata.to_sql('url_categeory',conn,flavor='mysql',if_exists='append',index=False,
             chunksize=50000)
    insert_end_time=time.time() 
    print("数据库时间:"+str(insert_end_time-insert_time))

      
    
