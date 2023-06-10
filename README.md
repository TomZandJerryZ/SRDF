# SRDF

Python 3.6 / 3.9 
CUDA 10.2+
(torch 2.0.1+cu118(optional))

cd DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace

cd nms 
python setup.py build_ext --inplace

cd points_to_rect (optional)
python setup.py build_ext --inplace

cd locat_overlaps(optional)
python setup.py build_ext --inplace

cd overlaps(optional)
python setup.py build_ext --inplace

Cython,numpy,opencv-python,imgaug

#train
train_#(dataset).py
#eval
eval_test.py
#show
drew_res.py

-----data----
label
x y x y x y x y cls diff
image

train.txt / test.txt
[filename0\n
filename1\n
.....
]


