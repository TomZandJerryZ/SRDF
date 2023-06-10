import os


# imagesource:GoogleEarth
# gsd:0.146343590398



label_p = r'C:\Users\60590\Desktop\CAR_\labelTxt'
b = ['imagesource:GoogleEarth\n','gsd:0.146343590398\n']
for x in os.listdir(label_p):
    l_p = os.path.join(label_p,x)
    with open(l_p,'r')as F:
        a = F.readlines()
    # print(a)
    aa = b+a
    with open(l_p,'w')as FF:
        FF.writelines(aa)
    # break
