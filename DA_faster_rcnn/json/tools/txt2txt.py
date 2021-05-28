#coding=utf-8
import os
import os.path #文件夹遍历函数
#获取目标文件夹的路径
filedir = '/home/ubuntu/Documents/June/datasets/pascal_voc/VOCdevkit/ImageSets/Main/１'
outtxt = "train.txt"
outtxtpath = os.path.join(filedir, outtxt)

#获取当前文件夹中的文件名称列表
filenames=os.listdir(filedir)
#打开当前目录下的result.txt文件，如果没有则创建
f=open(outtxtpath,'w')
#先遍历文件名

j = 0
for filename in filenames:
    if filename.endswith(".txt"):
        i = 0
        filepath = filedir+'/'+filename
        #遍历单个文件，读取行数
        for line in open(filepath):
            i += 1
            j +=1
            f.writelines(line)
        f.write('\n')
    print(i)
print(i)
#关闭文件
f.close()