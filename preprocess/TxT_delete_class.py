f=open('/media/meng2/disk11/ZRT/dataset/VSAC/novel/val_mini_withClass.txt','r').readlines()
f2=open('/media/meng2/disk11/ZRT/dataset/VSAC/novel/val_mini.txt','w')
for item in f:
    item=item.strip('\n')
    item = item.split(' ')[0]
    f2.write(item)
    f2.write('\n')
f2.close()