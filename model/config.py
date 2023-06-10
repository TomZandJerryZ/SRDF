class DefaultConfig():
    #backbone
    pretrained=True #使用imagenet预训练参数 可选
    freeze_stage_1=False #参数冻结 可选
    freeze_bn=True #参数冻结 可选

    #fpn
    fpn_out_channels=256 # FPN金字塔结构输出特征图的维度 128 64
    use_p5 = True # 决定 p6 p7
    
    #head
    class_num= 1 # 输出类别
    use_GN_head=True # BN OR GN
    prior=0.01 # 初始化分类输出头 所需参数
    add_centerness=False # 是否使用centerness head
    cnt_on_reg=False # centernesss head 是否与回归头共享输入特征

    #training 用于在FPN多层输出头上分配标签所使用的尺度
    # strides=[8,16,32,64,128]
    strides = [4,8, 16,32, 64, 128] #2 4567
    # strides = [4, 32, 128]# 2 56
    # limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    limit_range = [[1, 1024],[1024,4096], [4096, 16384], [16384, 65536], [65536,262144],[262144, 999999]]
    # limit_range = [[1,  16384], [16384, 65536], [65536, 262144], [262144, 999999]]#2 567
    # limit_range = [[1, 16384], [16384, 262144], [262144, 999999]]# 2 56

    # strides = [8,16,32]
    #
    T_a = 15
    # gauss_range = [0.95,0.8,0.7,0.6,0.4,0.2]
    gauss_range = [0.95,0.8,0.8,0.8,0.8,0.8]
    #inference
    score_threshold=0.25 # 分类阈值
    nms_iou_threshold=0.1 # nms阈值
    max_detection_boxes_num=1000 # 决定使用多少像素点进行预测

    # input_w = 600
    # input_h = 600

    # train_txt = r'C:\Users\60590\Desktop\dota\train.txt'
    # test_txt = r'C:\Users\60590\Desktop\dota\test_.txt'
    # train_image_p = r'C:\\Users\\60590\\Desktop\\dota\\images\\'

    source_path = 'C:/Users/60590/Desktop/SRDF/'
    # train_txt = r'C:\Users\60590\Desktop\dota\train.txt'
    # test_txt = r'C:\Users\60590\Desktop\dota\test_.txt'
    # train_image_p = r'C:\\Users\\60590\\Desktop\\dota\\images\\'
    # test_label = r'C:\Users\60590\Desktop\dota\labelTxt'
    # gt_path = r'C:\Users\60590\Desktop\dota\labelTxt'
    # test_img_p = r'C:\Users\60590\Desktop\dota\images'

    # train_txt = r'C:\Users\60590\Desktop\HRSC\train.txt'
    # test_txt = r'C:\Users\60590\Desktop\HRSC\test.txt'
    # train_image_p = r'C:\\Users\\60590\\Desktop\\HRSC\\images\\'
    # test_label = r'C:\Users\60590\Desktop\HRSC\labelTxt'
    # gt_path = r'C:\Users\60590\Desktop\HRSC\labelTxt'
    # test_img_p = r'C:\Users\60590\Desktop\HRSC\images'

    # train_txt = r'C:\Users\60590\Desktop\CAR_S\train.txt'
    # test_txt = r'C:\Users\60590\Desktop\CAR_S\test.txt'
    # train_image_p = r'C:\\Users\\60590\\Desktop\\CAR_S\\images\\'
    # test_label = r'C:\Users\60590\Desktop\CAR_S\labelTxt'
    # gt_path = r'C:\Users\60590\Desktop\CAR_S\labelTxt'
    # test_img_p = r'C:\Users\60590\Desktop\CAR_S\images'

    train_txt = r'C:\Users\60590\Desktop\RSDD\train.txt'
    test_txt = r'C:\Users\60590\Desktop\RSDD\test_offshore.txt'
    train_image_p = r'C:\\Users\\60590\\Desktop\\RSDD\\images\\'
    test_label = r'C:\Users\60590\Desktop\RSDD\labelTxt'
    gt_path = r'C:\Users\60590\Desktop\RSDD\labelTxt'
    test_img_p = r'C:\Users\60590\Desktop\RSDD\images'