# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:58:41 2022
@author: Administrator
"""
 
import requests
 
def client_post(url, data):
    rep = requests.post(url, files=files)
    return rep.text 
 
if __name__ == '__main__':
    files = {
            "imagename":"videdo server demo",
            "image":("1.mp4",open('C:/Users/Z2 air/Desktop/test1.mp4','rb'),"application/octet-stream"),
            "imagetype":"jpg",
            "key1":"key1 content"
            }
    url = 'http://127.0.0.1:1111/model'
    res = client_post(url, files)
    print('127.0.0.1:1111/model(返回结果):', res)