# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:43:36 2017

@author: huangshizhi
"""

import happybase
import logging
import numpy as np
import pandas as pd
import struct
import time 


'''
从Hbase数据库读取表及其相关数据等

'''
#print(conn.tables())
#[b'ad_portrait', b'emp', b'urlinfo']
#emp  b'1' {b'professinal:salary': b'66666'}
import happybase
conn = happybase.Connection('192.168.50.158')
conn.open()
table = conn.table('adx_portrait_0310')
rows = table.scan(limit=10)
for key,data in rows:
	  print(key,data)

conn.close()

'''
import happybase
conn = happybase.Connection('192.168.50.158')
conn.open()
table = conn.table('url_info')
rows = table.scan(limit=10)
for key,data in rows:
	  print(key,data)

conn.close()


'''

'''
在Hbase上建表,添加数据并查看数据
'''
import happybase
conn = happybase.Connection('192.168.50.158')
conn.open()
conn.create_table(
    'adx_portrait_0310',
		{'adx': dict(max_versions=10),})
table = conn.table('adx_portrait_0310')
table.put('123456', {'adx:attribute': "{'人口属性': {'身份职业/职员': 1.0}, '个人关注': {'娱乐/动漫': 1.0}, '地域': {'中国广东韶关': 1.0}}",
                      })			
			
table = conn.table('adx_portrait_0310')
rows = table.scan()
for key,data in rows:
	  print(key,data)

conn.close()

'''
在Hbase上删除表
'''
import happybase
conn = happybase.Connection('192.168.50.158')
conn.open()
conn.disable_table('adx_portrait_0310')
conn.delete_table('adx_portrait_0310')
conn.close()




'''
将数据从数据框写到Hbase数据库

'''
conn = happybase.Connection('192.168.50.158')
conn.open()
#读取pandas数据框，索引栏为'adxId'
table = conn.table('adx_portrait')
#外网服务器
df = pd.read_csv("/home/bjyfb/workspaces/hsz/test_hbase_0309.csv",encoding='utf-8',index_col=['adxId'])
start_time = time.time()
with table.batch(transaction=True) as b:
    for row in df.iterrows():
        row_key = str(row[0])
        row_value = dict()
        for column,value in row[1].iteritems():
            if not pd.isnull(value):
                row_value[':'.join(('adx',column))] = str(value)
        b.put(row_key,row_value) 
conn.close()
end_time = time.time()
print("插入Hbase数据库所需要的时间为:"+str((round(end_time-start_time),2)))

'''
#外网本地
df_test = pd.read_csv(r"D:\ad_portrait\test\data\test_hbase_0309.csv",encoding='utf-8',index_col=['adxId'])

df = df_test[6:7]

for row in df.iterrows():
    row_key = str(row[0])
    row_value = dict()
    for column,value in row[1].iteritems():
        if not pd.isnull(value):
            row_value[':'.join(('adx',column))] = str(value)
    print(row_value)
'''



'''
将数据从Hbase数据库写到pandas，文本等

'''
import happybase
import logging
import numpy as np
import pandas as pd
import struct
import time 
conn = happybase.Connection('192.168.50.158')
conn.open()
start_time = time.time()
table = conn.table('adx_portrait_0320')
df = pd.DataFrame(columns=['adxId','attribute'])
rows = table.scan(limit=1)

#键值对类型数据的读取
for k,d in rows:
    for key,value in d.items():
    	  attribute = value.decode()
    	  #print(key,attribute)    	  
    df.loc[len(df)] = [k,attribute]

df.to_csv('/home/bjyfb/project/data/temp/adx_portrait_0320.csv',index=None,encoding='utf-8')

#df.to_csv('/home/bjyfb/workspaces/jwkj/data/pdata.csv',index=None,encoding='utf-8')


#整张表的导出
df2 = pd.DataFrame(columns=['name','city','designation','salary'])
for row in rows:
    df_row = {key.decode().split(':')[1]: value.decode() for key, value in row[1].items()}
    df2 = df2.append(df_row, ignore_index=True)
df2.to_csv("/home/bjyfb/software/test/pdata.csv",index=None) 


conn.close()
end_time = time.time()
print("从Hbase数据库输出数据所需要的时间为:"+str(round((end_time-start_time),2)))
