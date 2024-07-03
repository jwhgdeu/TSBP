from pyemd import emd_with_flow
from sklearn.cluster import KMeans
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os

# 创建resNet50模型,将模型的倒数第二层输出作为图像特征
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_50 = models.resnet50(pretrained=True)
modules = list(resnet_50.children())[:-1]
resnet = nn.Sequential(*modules)
resnet.to(device)

# 计算图像的颜色直方图，需要的格式是BGR
def calc_hist(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def get_img_features_gland(image):
    # 调整图像大小为224x224,并进行标准化
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485,0.456,0.406],
                               std = [0.229,0.224,0.225])])
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    resnet.eval()
    with torch.no_grad():
        features = resnet(image).squeeze().cpu()
    return features

def get_img_features_cell(image):
    features1 = get_img_features_gland(image)
    image2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hist = calc_hist(image2) * 15
    features = torch.from_numpy(np.concatenate((hist, features1.numpy())))
    return features

def cal_min_len(bboxs_anno):

    feature = [d['feature'] for d in bboxs_anno]
    # 计算样本点之间的距离
    A = torch.stack(feature)
    distances = torch.cdist(A, A, p=2)

    # 将对角线元素设置为一个非常大的值，这样在寻找最小值时就会忽略它们
    distances.fill_diagonal_(float('inf'))

    # 计算每个点与其他所有点的最短距离
    min_distances, _ = torch.min(distances, dim=1)
    min_distances, _ = torch.sort(min_distances)
    min_dis = float(min_distances[0])
    dist_aver = float(torch.sum(min_distances) / len(feature))
    return dist_aver,min_dis


def K_means(bboxs_anno,num):

    features = [d['feature'] for d in bboxs_anno]
    features = torch.stack(features)
    # 定义聚类数量和模型
    k = num
    kmeans = KMeans(n_clusters=k,max_iter=500,n_init=k,random_state=1234)
    kmeans.fit(features)
    centers = kmeans.cluster_centers_
    # 打印结果
    bbox_anno=[]
    bbox_fea = []
    for i, cluster in range(k):
        tmp = dict()
        tmp['img_name'] = 'from_kmeans'
        tmp['bbox'] = (1.0,1.0,1.0,1.0)
        tmp['bbox_with_score'] =(1.0,1.0,1.0,1.0,1.0)
        tmp['feature'] = torch.from_numpy(centers[i])
        bbox_fea.append(tmp['feature'])
        bbox_anno.append(tmp)
    return bbox_anno,bbox_fea
    pass

def get_bbox_anno(bbox_folder):
    # 定义正样本、负样本、候选框
    boxes_TP_orign = []
    boxes_FP_orign = []
    boxes_Candidate = []
    bbox_dir = os.listdir(bbox_folder)
    image_Name = []
    for txt_name in bbox_dir:
        txt_path = os.path.join(bbox_folder, txt_name)
        img_name = txt_name.split('.')[0]
        image_Name.append(img_name)
        # 需要裁剪的图像
        img_path = f'/your/img_path/{img_name}.png'
        img = Image.open(img_path)
        with open(txt_path) as f:
            bboxs = f.readlines()
            for id, bbox in enumerate(bboxs):
                bbox = bbox.strip()
                # 用于记录框坐标，框特征，图像名
                boxes = dict()
                x_f, y_f, w_f, h_f, score = bbox.split(' ')
                x1 = int(float(x_f))
                y1 = int(float(y_f))
                x2 = int(float(x_f) + float(w_f) + 0.50)
                y2 = int(float(h_f) + float(y_f) + 0.50)
                score1 = float(score)
                if (x2 > (x1 + 2)) and (y2 > (y1 + 2)):
                    tmp = (x1, y1, x2, y2)
                    tmp_f = (x_f, y_f, w_f, h_f, score)
                    img1_crop = img.crop(tmp)
                    # 获取框内图像的特征
                    img1_fea = get_img_features_gland(img1_crop)
                    boxes['img_name'] = img_name
                    boxes['bbox'] = tmp
                    boxes['bbox_with_score'] = tmp_f
                    boxes['feature'] = img1_fea
                    if score1 >= 0.50:
                        boxes_TP_orign.append(boxes)
                    if score1 < 0.30:
                        boxes_FP_orign.append(boxes)
                    if score1 >= 0.30 and score1 < 0.50:
                        boxes_Candidate.append(boxes)
    return boxes_TP_orign, boxes_FP_orign, boxes_Candidate, image_Name

def TSBP(bbox_folder,start_TP_NUM,start_FP_NUM):

    boxes_TP_orign, boxes_FP_orign, boxes_Candidate, image_Name = get_bbox_anno(bbox_folder)
    # 利用K-means
    boxes_TP,TP_fea = K_means(boxes_TP_orign,num=start_TP_NUM)
    boxes_FP,FP_fea=K_means(boxes_FP_orign,num=start_FP_NUM)
    del_index_FP = []
    del_index_TP = []
    # 用于计算K-means之前同类样本最小距离的平均和最小距离
    dist_sum_TP, glob_dis_TP = cal_min_len(boxes_TP_orign)
    dist_sum_FP, glob_dis_FP= cal_min_len(boxes_FP_orign)
    thresh_dist_TP = dist_sum_TP
    thresh_dist_FP = dist_sum_FP
    boxes_FP_sub = []
    boxes_TP_sub = []
    for index in range(0,len(TP_fea)):
        for index1 in range(0,len(FP_fea)):
            d = torch.dist(TP_fea[index],FP_fea[index1])
            if d < glob_dis_TP:
               del_index_FP.append(index1)
               del_index_TP.append(index)
    del_index_TP = list(set(del_index_TP))
    del_index_FP = list(set(del_index_FP))
    for item in del_index_TP:
        boxes_TP_sub.extend([x for x in boxes_TP if x['feature'].tolist() == TP_fea[item].tolist()])
        boxes_TP = [x for x in boxes_TP if x['feature'].tolist() != TP_fea[item].tolist()]

    for item in del_index_FP:
        boxes_FP_sub.extend([x for x in boxes_FP if x['feature'].tolist() == FP_fea[item].tolist()])
        boxes_FP = [x for x in boxes_FP if x['feature'].tolist() != FP_fea[item].tolist()]

    loop=0
    use_close=True
    while(len(boxes_Candidate)>0):
        loop+=1
        print(f'第{loop}次循环')
        # EMD匹配
        len_FP = len(boxes_FP)
        len_TP = len(boxes_TP)
        len_leaf1 = len(boxes_Candidate)
        full_len = len_leaf1+len_TP+len_FP
        # 定义P和Q的分布,以及初始化距离函数
        # P的范围：[0,(len_leaf1-1)],正样本框的范围：[len_leaf1,(len_leaf1+len_TP-1)],负样本框的范围：[(len_leaf1+len_TP),full_len-1]
        P = np.array([1.0 if i < len_leaf1 else 0.0 for i in range(full_len)])
        Q = np.array([0.0 if i < len_leaf1 else 1.0 for i in range(full_len)])
        Q_feature = boxes_TP+boxes_FP
        D = np.zeros((full_len,full_len))
        # 对D进行填充
        for i in range(len_leaf1):
            for j in range(len_leaf1,full_len):
                dist = torch.dist(boxes_Candidate[i]["feature"],Q_feature[j-len_leaf1]["feature"])
                D[i][j] = dist
                D[j][i] = dist

        emd,flow = emd_with_flow(P, Q, D,extra_mass_penalty=0)
        print(f'before EMD the Candidate_len is:{len_leaf1},the TP_len is:{len_TP},the FP_len is:{len_FP}')
        # 储存匹配上的候选框,方便对boxes_Candidate做删减操作

        TP_tem = []
        FP_tem = []
        for i in range(len_leaf1):
            for j in range(len_leaf1,full_len):
                if (flow[i][j])>0 and j<len_leaf1+len_TP:
                        distance = torch.dist(boxes_TP[j - len_leaf1]['feature'], boxes_Candidate[i]['feature'])
                        TP_Candi = dict()
                        TP_Candi['tp_candi'] = boxes_Candidate[i]
                        TP_Candi['distance'] = distance
                        TP_tem.append(TP_Candi)

                elif (flow[i][j])>0 and j>=len_leaf1+len_TP:
                        distance1 = torch.dist(boxes_FP[j - (len_leaf1 + len_TP)]['feature'], boxes_Candidate[i]['feature'])
                        FP_Candi = dict()
                        FP_Candi['fp_candi'] = boxes_Candidate[i]
                        FP_Candi['distance'] = distance1
                        FP_tem.append(FP_Candi)

        #给匹配的样本按距离排序
        TP_tem = sorted(TP_tem, key = lambda x : x['distance'])
        FP_tem = sorted(FP_tem, key = lambda x : x['distance'])

        print(f'TP匹配的正候选框个数：{len(TP_tem)},FP负候选框个数：{len(FP_tem)}')
        if len(TP_tem) > 0:
            print(f'TP min dist：{(TP_tem[0]["distance"])},TP max dist：{TP_tem[-1]["distance"]}')
        if len(FP_tem) > 0:
            print(f'FP min dist：{(FP_tem[0]["distance"])},FP max dist：{FP_tem[-1]["distance"]}')
        boxes_Candidate_tmp = []
        TP_add = 0
        FP_add = 0
        for item_tp in TP_tem:
            if item_tp['distance'] <= thresh_dist_TP:
                tmp_tp_sample = item_tp['tp_candi']
                boxes_Candidate_tmp.append(tmp_tp_sample)
                boxes_TP_orign.append(tmp_tp_sample)
                boxes_TP.append(tmp_tp_sample)
                TP_add += 1

        for item_fp in FP_tem:
            if item_fp['distance'] <= thresh_dist_FP:
                tmp_fp_sample = item_fp['fp_candi']
                boxes_Candidate_tmp.append(tmp_fp_sample)
                boxes_FP.append(tmp_fp_sample)
                FP_add += 1
        for item in boxes_Candidate_tmp:
            boxes_Candidate = [x for x in boxes_Candidate if x['bbox'] != item['bbox']]

        print(f'TP_add is:{TP_add},FP_add is:{FP_add}')
        if TP_add == 0 and FP_add == 0 and not use_close:
            #第二阶段放宽距离限制
            thresh_dist_TP = float('inf')
            thresh_dist_FP = float('inf')

        if TP_add == 0 and FP_add == 0 and use_close:
            if len(boxes_FP_sub):
               boxes_FP += boxes_FP_sub
            if len(boxes_TP_sub):
               boxes_TP += boxes_TP_sub
            use_close = False
    #保存匹配结果
    for img_name in image_Name:
        folder1='/save/bbox_result_folder/path'
        if not os.path.exists(folder1):
            os.makedirs(folder1)
        with open(folder1 + f'/{img_name}.txt', 'w') as fbbox:
            for i, bbox in enumerate(boxes_TP_orign):
                if bbox['img_name'] == img_name:
                    bbox_str = bbox['bbox_with_score'][0] + ' ' + bbox['bbox_with_score'][1] + ' ' + \
                               bbox['bbox_with_score'][2] + ' ' + bbox['bbox_with_score'][3] + ' ' + \
                               bbox['bbox_with_score'][4] + '\n'
                    fbbox.write(bbox_str)


if __name__ == '__main__':
    #记录每一张图像预测结果.txt的文件夹
    bbox_folder = '/your/path/bbox_txt'
    #二分类任务，将高置信度的框记作'TP'，称作正样本,低置信度的框记作'FP'，称作负样本
    #初始时刻TP、FP经K-means处理簇的数量
    start_TP_NUM = 25
    start_FP_NUM = 25
    TSBP(bbox_folder,start_TP_NUM,start_FP_NUM)


