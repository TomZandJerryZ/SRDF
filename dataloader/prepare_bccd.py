import os
import random
import shutil
'''首先运行prepare.py 将文件名写入对应txt文件'''
def prepare_data(path,datatype=['train','val','test']):
    imgsetpath = os.path.join(path, "Mainn", "%s.txt")
    for dt in datatype:
        p = imgsetpath % dt
        im_path = os.path.join(path,dt,"Images")
        im = os.listdir(im_path)
        for i in im:
            file_name = i[:-4]
            with open(p,'a')as FFF:
                FFF.writelines(file_name+'\n')
        # print(p)

# bccd_p =  r'X:\Entertain\Gaming\Complete-Blood-Cell-Count-Dataset-master\Complete-Blood-Cell-Count-Dataset-master'# bccd文件夹地址
# prepare_data(path=bccd_p)


luna_p = 'F:\yxtxjc\lun16\data\Annotations'# 转换之后的luna数据地址
def prepare_luna(path=r'X:\Entertain\Gaming\luna',datatype=['train','val','test']):

    img_list = [x[:-4] for x in os.listdir(luna_p)]

    random.shuffle(img_list)
    train_set = img_list[0:int(len(img_list)*0.7)]
    testval_set = []
    for i in img_list:
        if i not  in train_set:
            testval_set.append(i)
    for dir in datatype:
        im_p = os.path.join(path,dir,"Images")
        label_p = os.path.join(path, dir, "Annotations")

        for dir_p in [im_p,label_p]:
            if os.path.exists(dir_p):
                continue
            else:
                os.mkdir(dir_p)
        if dir == 'train':
            for i in train_set:
                old_im_p = os.path.join(luna_p.replace('Annotations', 'LabelImages'), i + '.png')
                old_label_p = os.path.join(luna_p, i + '.xml')
                new_im_p = os.path.join(im_p, i + '.png')
                shutil.copy(old_im_p,new_im_p)
                new_label_p = os.path.join(label_p, i + '.xml')
                shutil.copy(old_label_p, new_label_p)
        else:
            for i in testval_set:
                old_im_p = os.path.join(luna_p.replace('Annotations', 'LabelImages'), i + '.png')
                old_label_p = os.path.join(luna_p, i + '.xml')
                new_im_p = os.path.join(im_p, i + '.png')
                shutil.copy(old_im_p,new_im_p)
                new_label_p = os.path.join(label_p, i + '.xml')
                shutil.copy(old_label_p, new_label_p)

    imgsetpath = os.path.join(path, "Mainn", "%s.txt")
    if not os.path.exists(os.path.join(path, "Mainn")):
        os.mkdir(os.path.join(path, "Mainn"))
    for dt in datatype:
        p = imgsetpath % dt
        im_path = os.path.join(path, dt, "Images")
        im = os.listdir(im_path)
        for i in im:
            file_name = i[:-4]
            with open(p, 'a') as FFF:
                FFF.writelines(file_name + '\n')



# prepare_luna()
import os
import shutil

def prepare_dota_eval():
    f_ = r'../evalutate/dota/ground-truth'
    shutil.rmtree(f_)
    os.makedirs(f_)
    # print(os.listdir(f_))

    image_set_file = r'C:\Users\sparrow\Desktop\dota_test\test.txt'
    with open(image_set_file) as f:

        label_list = [x.strip() for x in
                      f.readlines()]
        # new_list
    for i in label_list:
        old = os.path.join(r'C:\\Users\\sparrow\\Desktop\\dota_test\\labelTxt\\', i + '.txt')
        new = os.path.join(f_, i + '.txt')
        # shutil.copy(old,new)
        with open(old,'r') as ff:
            biu = ff.readlines()
        for line in biu:
            pt = line.split('\n')[0].split(' ')
            different = pt[-1]
            name = pt[-2]
            if different == '0':
                new_line = '{} {} {} {} {} {} {} {} {}\n'.format(
                                name,pt[0],pt[1],pt[2],pt[3],pt[4],pt[5],pt[6],pt[7])
            else:
                new_line = '{} {} {} {} {} {} {} {} {} difficult\n'.format(
                    name, pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7])
            with open(new,'a')as FFF:
                FFF.writelines(new_line)
prepare_dota_eval()
