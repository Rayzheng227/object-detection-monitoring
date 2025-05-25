import numpy as np
import cv2
import sys
from time import time
import os
import math
from sklearn.metrics import pairwise_distances
from numba import jit, njit

from ..tracker.kcftracker_mod import KCFTracker
from ..extracotrs.shuffenet import SemanticExtractor
from PIL import Image

# Intersection over Union函数是计算两个边界框的交并比（用于衡量两个边界框的重叠程度）
@jit
def intersection_over_union(bbox1, bbox2):
    """
    :param bbox1: bounding-box with [left, top, right, bottom]
    :param bbox2: bounding-box with [left, top, right, bottom]
    :return: IoU of bbox1 and bbox2
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    # print(bbox1, bbox2)
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    # compute the area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    bbox1Area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2Area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(bbox1Area + bbox2Area - interArea)
    # return the intersection over union value
    return iou

# Martix_iou函数是计算两个边界框的交并比矩阵，用于Intersection over Union的批量计算
@jit
def matrix_iou(arr1, arr2):
    """
    :param arr1: array of bboxes with shape (m, 4)
    :param arr2: array of bboxes with shape (n, 4)
    :return: IoU between arr1 and arr2, with shape (m, n)
    """
    assert arr1.shape[1] == arr2.shape[1] == 4
    ious = np.zeros([arr1.shape[0], arr2.shape[0]], dtype=np.float32)
    for m in range(arr1.shape[0]):
        for n in range(arr2.shape[0]):
            ious[m][n] = intersection_over_union(arr1[m], arr2[n])
    return ious


# _ratio函数是计算两个数的比值（计算宽高比）
def _ratio(param1, param2):
    assert param1 > 0 and param2 > 0
    return min(param1, param2) / max(param1, param2)

# wh_ratio函数是计算两个边界框的宽高比矩阵，用于宽高比的批量计算
@jit
def wh_ratio(arr1, arr2):
    assert arr1.shape[1] == arr2.shape[1] == 4
    ratio1 = (arr1[:, 2] - arr1[:, 0]) / (arr1[:, 3] - arr1[:, 1])
    ratio2 = (arr2[:, 2] - arr2[:, 0]) / (arr2[:, 3] - arr2[:, 1])
    ratios = np.zeros([arr1.shape[0], arr2.shape[0]], dtype=np.float32)
    for m in range(arr1.shape[0]):
        for n in range(arr2.shape[0]):
            ratios[m, n] = min(ratio1[m], ratio2[n]) / max(ratio1[m], ratio2[n])
    return ratios


# Class Monitor是一个用于目标跟踪的类
class Monitor(object):
    # 初始化：frame_size是帧的大小，max_to_keep是最大保留的轨迹数，max_age是最大的丢失帧数，min_iou是最小的交并比
    # 该函数会初始化一些变量，包括帧的宽高、列数、行数、最大宽度、帧数、目标数、当前轨迹数、上一帧的目标数、最大保留轨迹数、最大丢失帧数、最小交并比、最小宽高比、轨迹列表、语义提取器(来自shufflenet）
    def __init__(self, frame_size, max_to_keep=50, max_age=2, min_iou=0.05):
        self._frame_width = frame_size[0]
        self._frame_height = frame_size[1]
        self._cols = 0
        self._rows = 0

        self._max_width = 0
        self._frame_count = 0
        self._object_count = 0
        self._current_tracks = 0
        self._last_objects = 0

        self._max_to_keep = max_to_keep
        self._max_age = max_age
        self._min_iou = min_iou
        self._min_ratio = 0.5
        self._tracks = []

        self.net = SemanticExtractor()

    # detect_lo函数是检测丢失的目标，返回丢失的目标列表
    # 具体来说，该函数会遍历所有的轨迹，如果某个轨迹的匹配结果中没有匹配到任何目标，且该轨迹的丢失帧数小于1，且目标的中心点在某个区域内，那么就将该轨迹加入到丢失的目标列表中
    def _detect_lo(self, predictions, match_result):
        t0 = time()
        matches = np.sum(match_result, axis=1) # 对每个轨迹的匹配结果求和
        # print(np.where(matches == 0)[0])
        ret_flag = False
        lost = []
        if np.where(matches == 0)[0].size == 0: # 如果轨迹的匹配结果中没有匹配到任何目标，那么直接返回None
            return None
        t1 = time()
        # print('lo1:', t1 - t0)
        for track in np.where(matches == 0)[0]:
            t2 = time()
            # print('lo2:', t2 - t1)
            cx = (predictions[track][0] + predictions[track][2]) / 2
            cy = (predictions[track][1] + predictions[track][3]) / 2

            col = int(cx / (1242 / 9))
            row = int(cy / (374 / 6))
            '''
            NOTE : OLD VERSION： 缺失了label以及score的信息
            if self._tracks[track]['miss_age'] < 1 and 0 < col < 9 - 1 and 0 < row < 6 - 1: # 如果目标中心点在区域内
                lost_track = {'id': self._tracks[track]['id'], 'left': predictions[track][0],
                              'top': predictions[track][1], 'right': predictions[track][2],
                              'bottom': predictions[track][3]}
                # XXX: debug; 看看predict的具体格式如何
                print('---------------')
                print('predict:', predictions[track])
                print('self._tracks:', self._tracks[track])
                print('==============')
                lost.append(lost_track)
                ret_flag = True
            '''
            '''
            XXX: NEW VERSION: 增加label和score两个
            '''
            if self._tracks[track]['miss_age'] < 1 and 0 < col < 9 - 1 and 0 < row < 6 - 1:
                lost_track = {'id': self._tracks[track]['id'], 'label': self._tracks[track]['label'][-1],
                              'score': self._tracks[track]['score'][-1], 'left': predictions[track][0],
                              'top': predictions[track][1], 'right': predictions[track][2],
                              'bottom': predictions[track][3]}  
                lost.append(lost_track)
                ret_flag = True
        # print('lo3:', time() - t2)
        if ret_flag:
            # print('lo:', time() - t0)
            return lost
        # print('lo:', time() - t0)
        return None

    # temporal_anomaly_detect函数是检测轨迹的异常
    def temporal_anomaly_detect(self, detections, predictions, match_result):
        warns = {'sharp_change': None,
                 'lost_object': self._detect_lo(predictions, match_result),
                 'label_switch': None}
        return warns, detections

    # 在第一帧中初始化目标跟踪器
    # @jit
    def init(self, frame, detections):
        # first frame track construction
        num_objects = detections['num_objects']
        for obj in range(num_objects):
            # 遍历检测到的每一个对象
            # 创建对象字典（Track，包含对象id、标签、得分、跟踪器、出现帧数、丢失帧数、左上角坐标、右下角坐标）
            track = {'id': self._object_count, 'label': [], 'score': [],
                     'tracker': KCFTracker(True, True, True), 'appear': self._frame_count, 'miss_age': 0,
                     'left': detections['2d_bbox_left'][obj], 'top': detections['2d_bbox_top'][obj],
                     'right': detections['2d_bbox_right'][obj], 'bottom': detections['2d_bbox_bottom'][obj]}
            # 将检测对象添加到列表中
            track['label'].append(detections['label'][obj])
            track['score'].append(detections['score'][obj])
            track['tracker'].init(
                [int(detections['2d_bbox_left'][obj] + 0.5), int(detections['2d_bbox_top'][obj] + 0.5),
                 int(detections['2d_bbox_right'][obj] - detections['2d_bbox_left'][obj] + 0.5),
                 int(detections['2d_bbox_bottom'][obj] - detections['2d_bbox_top'][obj] + 0.5)], frame)
            self._tracks.append(track)
            self._object_count += 1
            self._current_tracks += 1
        self._last_objects = num_objects

    # trackmatch用于匹配匹配当前帧的检测结果和跟踪器
    # @jit
    def trackMatch(self ,detections, predictions,pred_img ,det_img):
        """
        Match predictions from tracks and current detections
        :param detections: dict
        :param predictions: dict
        :return: match matrix with
        """
        # Get center points
        # cp_pred = np.zeros([self._current_tracks, 2])
        # miss_ages = np.zeros(self._current_tracks, dtype=int)
        # diag_dist = np.zeros(self._current_tracks)
        # cp_det = np.zeros([detections['num_objects'], 2])
        # bbox_pred = np.zeros([self._current_tracks, 4], dtype=float)
        # bbox_det = np.zeros([detections['num_objects'], 4], dtype=float)
        # match_matrix = np.zeros([self._current_tracks, detections['num_objects']], dtype=int)
        # for track in range(self._current_tracks):
        
        # 初始化变量(pred预测结果，det当前帧检测结果)
        cp_pred = np.zeros([self._current_tracks, 2]) # cp_pred 预测边框的中心点坐标
        miss_ages = np.zeros(self._current_tracks, dtype=int) # miss_ages 每个跟踪器的丢失帧数
        diag_dist = np.zeros(self._current_tracks) # diag_dist 对角线距离(预测边界框的对角线)
        cp_det = np.zeros([detections['num_objects'], 2]) #cp_det 存储检测边界框的中心点坐标
        bbox_pred = np.zeros([self._current_tracks, 4], dtype=float) # 存储预测边界框
        bbox_det = np.zeros([detections['num_objects'], 4], dtype=float) # 存储检测边界框
        match_matrix = np.zeros([self._current_tracks, detections['num_objects']], dtype=int) # 匹配矩阵
        for track in range(self._current_tracks):
            # 遍历每一个跟踪器，获取预测边界框的中心点坐标、丢失帧数、预测边界框的坐标
            miss_ages[track] = self._tracks[track]['miss_age']
            bbox_pred[track][0] = predictions[track][0]
            bbox_pred[track][1] = predictions[track][1]
            bbox_pred[track][2] = predictions[track][2]
            bbox_pred[track][3] = predictions[track][3] 
            # bbox_pred[track]=[left,top,right,bottom] 四个点的坐标
            cp_pred[track][0] = (predictions[track][0] + predictions[track][2]) / 2
            cp_pred[track][1] = (predictions[track][1] + predictions[track][3]) / 2
            # cp_pred[track]=[x,y] 中心点坐标
            diag_dist[track] = pairwise_distances(np.array([[predictions[track][0], predictions[track][1]]]),
                                                  cp_pred[track:track + 1]) # 对角线距离
        '''
        检查：打印调试信息，确保数组长度和索引范围正确（报错原因可能是数组长度和索引范围不匹配）
        # '''
        # print("detections['num_objects']:", detections['num_objects'])
        # print("detections['2d_bbox_left'] length:", len(detections['2d_bbox_left']))
        # print("detections['2d_bbox_top'] length:", len(detections['2d_bbox_top']))
        # print("detections['2d_bbox_right'] length:", len(detections['2d_bbox_right']))
        # print("detections['2d_bbox_bottom'] length:", len(detections['2d_bbox_bottom']))
        # 对检测结果detection进行遍历
        for obj in range(detections['num_objects']):
            bbox_det[obj][0] = detections['2d_bbox_left'][obj]
            bbox_det[obj][1] = detections['2d_bbox_top'][obj]
            bbox_det[obj][2] = detections['2d_bbox_right'][obj]
            bbox_det[obj][3] = detections['2d_bbox_bottom'][obj]
            cp_det[obj][0] = (detections['2d_bbox_left'][obj] + detections['2d_bbox_right'][obj]) / 2
            cp_det[obj][1] = (detections['2d_bbox_top'][obj] + detections['2d_bbox_bottom'][obj]) / 2
        # Compute prediction-detection bbox distance
        dist = pairwise_distances(cp_pred, cp_det, metric='euclidean')
        # Compute priority score
        # 计算优先级分数（优先级在这里是做什么的？）
        # 优先级分数是通过计算预测边界框中心点到图像中心点的欧式距离，然后通过欧式距离和图像对角线的比值计算得到的
        pr_score = np.squeeze(
            pairwise_distances(cp_pred, np.array([[self._frame_width / 2, self._frame_height / 2]]),
                               metric='euclidean'), axis=-1)
        pr_score = 1 - pr_score / math.sqrt(
            (self._frame_width / 2) * (self._frame_width / 2) + (self._frame_height / 2 * self._frame_height / 2))
        # 生成匹配矩阵
        for age in range(self._max_age + 1):
            # 遍历max_age（即最大的丢失帧数）
            indices = np.where(miss_ages == age)[0] # 获取丢失帧数为age的跟踪器（即丢失帧数为age的跟踪器的索引）
            if indices.shape[0] > 0:
                # 如果丢失帧数为age的跟踪器的数量大于0，即存在丢失帧数为age的跟踪器
                temp_pr_score = pr_score[indices] # pr_score中丢失帧数为age的跟踪器的优先级分数
                pr_score_sort = np.argsort(temp_pr_score) # 对优先级分数进行排序
                for rank in range(indices.shape[0]):
                    # rank是优先级分数的排序，即按优先级从小到大，即优先处理丢失帧较少的预测结果
                    origin_index = indices[pr_score_sort[rank]]
                    valid_detections = np.where(dist[origin_index] <= diag_dist[origin_index])[0]
                    # 获取距离小于对角线距离的检测结果（若距离小于对角线距离，则认为是同一个目标）
                    # print(valid_detections)
                    if valid_detections.shape[0] > 0:
                        # 若存在距离小于对角线距离的检测结果，即同一个目标
                        ious = np.zeros(valid_detections.shape[0])
                        for obj in range(valid_detections.shape[0]):
                            # 计算交并比ious(两个边界框的交集面积除以两个边界框的并集面积)
                            ious[obj] = intersection_over_union(predictions[origin_index], np.array(
                                [detections['2d_bbox_left'][valid_detections[obj]],
                                 detections['2d_bbox_top'][valid_detections[obj]],
                                 detections['2d_bbox_right'][valid_detections[obj]],
                                 detections['2d_bbox_bottom'][valid_detections[obj]]]))
                        # print(ious)
                        candidates = np.argsort(ious)
                        # candidates代表iou从小到大的索引（iou越大 预测结果越好）
                        # print(candidates)
                        for obj in range(-1, -(candidates.shape[0] + 1), -1):
                            # 遍历iou从大到小的检测结果
                            candidate = valid_detections[candidates[obj]] # 获取iou最大的检测结果
                            # print(obj, candidate)
                            width_pred = bbox_pred[origin_index, 2] - bbox_pred[origin_index, 0] # 预测边界框的宽度
                            width_det = bbox_det[candidate, 2] - bbox_det[candidate, 0] # 检测边界框的宽度
                            height_pred = bbox_pred[origin_index, 3] - bbox_pred[origin_index, 1] # 预测边界框的高度
                            height_det = bbox_det[candidate, 3] - bbox_det[candidate, 1] # 检测边界框的高度
                            crop_pred = self.net.crop(pred_img,[(bbox_pred[origin_index, 0],bbox_pred[origin_index, 1],bbox_pred[origin_index, 2],bbox_pred[origin_index, 3])])
                            # crop_pred是预测边界框的图像, crop_det是检测边界框的图像
                            crop_det = self.net.crop(det_img,[(bbox_det[candidate, 0],bbox_det[candidate, 1],bbox_det[candidate, 2],bbox_det[candidate, 3])])
                            # 将两个crop的图像输入到语义提取器中，得到两个向量
                            pred_vector = self.net.semantic_extraction(crop_pred)
                            det_vector = self.net.semantic_extraction(crop_det)
                            # 通过余弦相似度计算两个向量的相似度
                            similar = self.net.cosine_similarity(pred_vector, det_vector)
                        
                            #if similar > 0.85:
                            if np.sum(match_matrix, axis=0)[candidate] == 0 \
                                    and similar > 0.85 \
                                    and ious[candidates[obj]] >= self._min_iou \
                                    and np.argwhere(np.argsort(dist[:, candidate]) == origin_index)[0][0] < 2:
                            # 若满足：1，检测结果没有被匹配过；2，相似度大于0.85；3，iou大于最小iou；4，距离最近的预测结果是自己
                            #         and _ratio(width_pred, width_det) > self._min_ratio \
                            #         and _ratio(height_pred, height_det) > self._min_ratio \
                            #         and abs(width_pred / height_pred - width_det / height_det) < 0.5\

                                # print(origin_index)
                                # print(np.argmin(dist[:, candidate]))
                                # print(np.argsort(dist[:, candidate]))
                                # print(_ratio(width_pred, width_det), _ratio(height_pred, height_det), width_pred / height_pred - width_det / height_det)
                                match_matrix[origin_index][candidate] = 1
                                # 即match_matrix[origin_index][candidate] = 1，表示预测结果origin_index和检测结果candidate匹配
                                break
                    else:
                        continue
            else:
                continue

        return match_matrix


    # @jit
    # update函数是更新目标跟踪器
    def update(self, pred_frame , frame, detections):
        '''
        input：pred_frame是预测帧，frame是当前帧，detections是当前帧的检测结果
        output：warns1是静态异常检测的结果，warns2是动态异常检测的结果，detections是更新后的检测结果，duration是更新的时间
        '''
        self._frame_count += 1
        # 每次更新，frame_count加1
        # print('frame:', frame.shape[1], frame.shape[0])
        # Tracker new location predict
        t0 = time()
        pred_locs = []

        # print('tracks:', self._current_tracks, 'detections:', detections['num_objects'])
        # current_tracks是当前的轨迹，detections['num_objects']是当前帧的目标数
        self._current_tracks = detections['num_objects'] # 更新当前轨迹数
        detections['errors'] = np.zeros((detections['num_objects'], 3), dtype=np.int32) #detecions['errors']用于存储错误信息
        
        # 遍历每一个轨迹，获取预测边界框的位置
        for track in range(self._current_tracks):
            loc = self._tracks[track]['tracker'].get_pred(frame) # 获取预测边界框的位置
            loc[2] += loc[0] # 右下角的x坐标加上左上角的x坐标
            loc[3] += loc[1] # 右下角的y坐标加上左上角的y坐标
            pred_locs.append(loc)
        
        pred_locs = np.array(pred_locs)
        t1 = time()
        print('track predict: {}s'.format(t1 - t0))
        duration0 = t1 - t0
        warns1 = {'wrong_location': None, 'unusual_size': None} # 初始化静态异常检测的结果：错误位置、异常大小
        t2 = time()
        print('static: {}s'.format(t2 - t1))
        duration1 = t2 - t1
        pred_image = Image.fromarray(cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB)) # 将预测帧转换为Image对象
        current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # 将当前帧转换为Image对象
        if detections['num_objects'] > 0 and self._current_tracks > 0:
            #如果当前帧的目标数大于0且当前轨迹数大于0，即存在目标且存在轨迹
            match_matrix = self.trackMatch(detections, pred_locs, pred_image, current_image) # 通过trackMatch函数匹配当前帧的检测结果和跟踪器
        elif detections['num_objects'] == 0:
            # 如果不存在目标
            match_matrix = np.zeros([self._current_tracks, 1], dtype=int)
        else:
            # 如果不存在轨迹
            match_matrix = np.zeros([1, detections['num_objects']], dtype=int)
        t3 = time()
        print('match: {}s'.format(t3 - t2))
        duration2 = t3 - t2
        if self._current_tracks > 0:
            # 如果当前轨迹数大于0，考虑另外三种异常：sharp_change、lose_object、label_switch（位置的突变、目标的丢失、标签的切换）
            warns2, detections = self.temporal_anomaly_detect(detections, pred_locs, match_matrix)
        else:
            warns2 = {'sharp_change': None, 'lost_object': None, 'label_switch': None}
        t4 = time()
        print('temporal: {}s'.format(t4 - t3))
        duration3 = t4 - t3
        self._last_objects = detections['num_objects'] # last_objects是上一帧的目标数
        sum_track = np.sum(match_matrix, axis=1)
        sum_det = np.sum(match_matrix, axis=0)

        # match update
        if self._current_tracks > 0 and detections['num_objects'] > 0:
            # 如果当前轨迹数大于0且当前帧的目标数大于0（即trackmatch函数匹配到了目标）
            # 更新轨迹（更新轨迹的标签、得分、位置、丢失帧数）
            # 问题在于 出现丢失目标时，虽然将丢失目标的边界框信息加了上去，但没有增加丢失目标的label以及score信息
            for track in np.where(sum_track == 1)[0]:
                index = np.where(match_matrix[track] == 1)[0][0]
                if self._max_to_keep <= len(self._tracks[track]['label']):
                    self._tracks[track]['label'].pop(0)
                    self._tracks[track]['score'].pop(0)
                self._tracks[track]['label'].append(detections['label'][index])
                self._tracks[track]['score'].append(detections['score'][index])
                self._tracks[track]['miss_age'] = 0
                self._tracks[track]['left'] = detections['2d_bbox_left'][index]
                self._tracks[track]['top'] = detections['2d_bbox_top'][index]
                self._tracks[track]['right'] = detections['2d_bbox_right'][index]
                self._tracks[track]['bottom'] = detections['2d_bbox_bottom'][index]
                self._tracks[track]['tracker'].update(frame, [detections['2d_bbox_left'][index],
                                                              detections['2d_bbox_top'][index],
                                                              detections['2d_bbox_right'][index],
                                                              detections['2d_bbox_bottom'][index]], use_detec=True)
        t5 = time()
        print('update: {}s'.format(t5 - t4))
        duration4 = t5 - t4
        # lost delete
        if self._current_tracks > 0:
            track_to_delete = []
            for track in np.where(sum_track == 0)[0]:
                if self._max_age <= self._tracks[track]['miss_age']: \
                        # or (self._tracks[track]['left'] +self._tracks[track]['right'])/2 < self._limits[0] * self._frame_width  \
                    # or (self._tracks[track]['left'] +self._tracks[track]['right'])/2 > self._limits[2] * self._frame_width:
                    track_to_delete.append(track)
                else:
                    self._tracks[track]['left'] = pred_locs[track][0]
                    self._tracks[track]['top'] = pred_locs[track][1]
                    self._tracks[track]['right'] = pred_locs[track][2]
                    self._tracks[track]['bottom'] = pred_locs[track][3]
                    self._tracks[track]['tracker'].update(frame, [pred_locs[track][0], pred_locs[track][1],
                                                                  pred_locs[track][2], pred_locs[track][3]],
                                                          use_detec=True)
                    self._tracks[track]['miss_age'] += 1
            for track in reversed(track_to_delete):
                self._tracks.pop(track)
                self._current_tracks -= 1
        t6 = time()
        print('delete: {}s'.format(t6 - t5))
        duration5 = t6 - t5

        # new track
        if detections['num_objects'] > 0:
            for det in np.where(sum_det == 0)[0]:
                track = {'id': self._object_count, 'label': [detections['label'][det]],
                         'score': [detections['score'][det]],
                         'tracker': KCFTracker(True, True, True), 'appear': self._frame_count, 'miss_age': 0,
                         'left': detections['2d_bbox_left'][det], 'top': detections['2d_bbox_top'][det],
                         'right': detections['2d_bbox_right'][det], 'bottom': detections['2d_bbox_bottom'][det]}
                track['tracker'].init(
                    [int(detections['2d_bbox_left'][det] + 0.5), int(detections['2d_bbox_top'][det] + 0.5),
                     int(detections['2d_bbox_right'][det] - detections['2d_bbox_left'][det] + 0.5),
                     int(detections['2d_bbox_bottom'][det] - detections['2d_bbox_top'][det] + 0.5)], frame)
                self._tracks.append(track)
                self._object_count += 1
                self._current_tracks += 1
        t7 = time()
        print('new: {}s'.format(t7 - t6))
        duration6 = t7 - t6
        return warns1, warns2, detections, [duration0, duration1, duration2, duration3, duration4, duration5, duration6]
