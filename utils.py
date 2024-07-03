import json
import os

# image的id与image_name一一对应
def read_coco_json():
    b = {}
    c = {}
    val_path = '/your/anno_path/instances_val2017.json'
    with open(val_path) as f:
        data_dict = json.load(f)
        for img_mess in data_dict['images']:
            b[img_mess['id']] = img_mess['file_name'].split('.')[0]
            c[img_mess['file_name'].split('.')[0]] = img_mess['id']
    return b,c

# 为每一张图像的预测结果生成一个txt文件，记录bbox、score
def coco_to_txt():
    img_id = 0
    result_path = '/your/predict_result_path/coco_instances_results.json'
    a,_ = read_coco_json()
    with open(result_path) as f:
        imgs_anns = json.load(f)
        for img_ann in imgs_anns:
            if(img_ann['image_id'] == img_id+1):
                img_id += 1
            if(img_ann['image_id'] == img_id):
                    x, y, w, h = img_ann['bbox']
                    score = img_ann['score']
                    bbox = f'{x}'+' '+f'{y}'+' '+f'{w}'+' '+f'{h}'+' '+f'{score}'+'\n'
                    bbox_path = r'/your/save_bbox_path/bbox_txt'+f'{a[img_id]}.txt'
                    with open(bbox_path,'a') as bf:
                        bf.write(bbox)

# 将bbox.txt转化为json格式
def txt_to_coco():
    img_annos = []
    b, _ = read_coco_json()
    bbox_folder = r'/your/bbox_txt/path'
    for i in range(len(os.listdir(bbox_folder))):
        bbox_name = b[i]
        bbox_name = bbox_name + '.txt'
        bbox_path = os.path.join(bbox_folder, bbox_name)
        with open(bbox_path, 'r') as f:
            bboxs = f.readlines()
            for bbox in bboxs:
                if bbox == '\n':
                    continue
                img_anno = {}
                x, y, w, h, score = bbox.split(' ')
                img_anno["image_id"] = i
                img_anno["category_id"] = 0
                img_anno["bbox"] = [float(x), float(y), float(w), float(h)]
                img_anno["score"] = float(score)
                img_annos.append(img_anno)
    json_anno = json.dumps(img_annos)
    with open(r'/your/save_json/path.json','w') as f1:
        f1.write(json_anno)


if __name__ =='__main__':
    coco_to_txt()
    txt_to_coco()


