# python

1.安装python第三方包，让Python连接Hbase服务器
下载相关第三方包，使用命令python setup.py install 进行安装
pip install thrift
pip install thriftpy
pip install happybase
其次，下载hbase文件解压之后的将文件夹复制到python的安装路径下，例如是
/.../anaconda3/lib/python3.5/site-packages
文件夹下面。
测试脚本：
import happybase
con = happybase.Connection('192.168.50.158')
没有报错即可！
