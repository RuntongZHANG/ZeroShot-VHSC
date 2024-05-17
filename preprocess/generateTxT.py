f1=open('/media/meng2/disk2/ZRT/dataset/voc_few_shot/VOCdevkit/'
        'VOC2012/ImageSets/Segmentation/train.txt','r').readlines()

f2=open('/media/meng2/disk2/ZRT/dataset/voc_few_shot/VOCdevkit/'
        'VOC2012/ImageSets/Segmentation/trainaug.txt','r').readlines()

f3=open('/media/meng2/disk2/ZRT/dataset/voc_few_shot/VOCdevkit/'
        'VOC2012/ImageSets/Segmentation/aug.txt','w')

f1_list=[]
f2_list=[]
for item in f1:
    item=item.strip('\n')
    f1_list.append(item)

for item in f2:
    item=item.strip('\n')
    f2_list.append(item)

for item in f2_list:
    if item not in f1_list:
        f3.write(item)
        f3.write('\n')
f3.close()