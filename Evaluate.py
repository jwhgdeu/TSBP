# 导入相关库和模块
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def bbox_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    inter_area = max(0,min(x1+w1,x2+w2)-max(x1,x2))*max(0,min(y1+h1,y2+h2)-max(y1,y2))
    union_area = w1*h1+w2*h2-inter_area
    iou = inter_area/union_area if union_area>0 else 0
    return iou
    pass


def get_tp_fp_fn(gt_anns, dt_anns, iou_thr):
    """
     计算当前图像预测框与真实框的iou
    """
    tp = 0
    fp = 0
    fn = 0
    if len(gt_anns) == 0:
        fp = len(dt_anns)
        return tp,fp,fn
    if len(dt_anns) == 0:
        fn = len(gt_anns)
        return tp, fp, fn
    gt_anns = sorted(gt_anns,key=lambda x:-x['area'])
    dt_anns = sorted(dt_anns,key=lambda x:-x['score'])
    iou_match = np.zeros(len(dt_anns))

    for gt_anno in gt_anns:
        max_iou = -np.inf
        max_idx = -1
        for idx,dt_anno in enumerate(dt_anns):
            if iou_match[idx] == 0 and gt_anno['category_id'] == dt_anno['category_id']:
                iou = bbox_iou(gt_anno['bbox'], dt_anno['bbox'])
                if iou >= iou_thr and iou > max_iou:
                    max_iou = iou
                    max_idx = idx
        if max_idx != -1:
            iou_match[max_idx] = 1
            tp+=1

    fp = len(dt_anns)-tp
    fn = len(gt_anns)-tp
    return tp, fp, fn

if __name__ =='__main__':
    TP = 0
    FP = 0
    FN = 0
    annFile = '/your/img_anno/path/instances_val2017_gland.json'
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes('/your/img_predict/path/instances_results.json')
    # 初始化 COCOeval对象
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    img_ids = sorted(cocoGt.getImgIds())
    for img_id in img_ids:
        # 获取图片的标注信息和预测结果
        ann_ids = cocoGt.getAnnIds(imgIds=img_id)
        gt_anns = cocoGt.loadAnns(ann_ids)
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img_id)
        dt_anns = cocoDt.loadAnns(dt_ann_ids)
        # tp,fp,fn
        tp, fp, fn = get_tp_fp_fn(gt_anns, dt_anns, iou_thr=0.50)
        TP += tp
        FP += fp
        FN += fn
        print(f'tp:{tp},fp:{fp},fn:{fn}')
    print(f'the total TP:{TP},FP:{FP},FN:{FN}')
    P = float(TP * 1.0 / (TP + FP))
    R = float(TP * 1.0) / (TP + FN)
    F = float(2 * P * R / (P + R))
    print(f'the P is:{P},the R is:{R},the F is:{F}')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

