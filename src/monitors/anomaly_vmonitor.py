import numpy as np
import cv2
import sys
from time import time
import os
import math
from sklearn.metrics import pairwise_distances
from numba import jit, njit

from ..tracker.kcftracker_mod import KCFTracker
from ..extracotrs.extractor import Extractor

# Vmonitor主要功能：
# 1. 初始化：构建先验知识，初始化跟踪器
# 2. 更新：更新跟踪器，删除丢失的跟踪器，添加新的跟踪器
# 3. 匹配：匹配跟踪器和检测器
# 4. 检测：检测静态异常和动态异常
# 5. 监控：监控跟踪器的状态，输出异常信息

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


def _ratio(param1, param2):
    assert param1 > 0 and param2 > 0
    return min(param1, param2) / max(param1, param2)


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


class Monitor(object):
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

        self.ext = None

    def _detect_wl(self, detections):
        valid_count = detections['num_objects']
        wrong_loc = []
        for det in range(detections['num_objects']):
            cls = detections['label'][det]
            cx = (detections['2d_bbox_left'][det] + detections['2d_bbox_right'][det]) / 2
            cy = (detections['2d_bbox_top'][det] + detections['2d_bbox_bottom'][det]) / 2
            col = int(cx / self.ext._x_batch)
            row = int(cy / self.ext._y_batch)
            # if cy <= self._frame_height * self._limits[1]:
            if cls == 0 or cls == 1 or cls == 2:
                _map = np.array(self.ext.region_map[0:3]).sum(axis=0)
            elif cls == 3 or cls == 4:
                _map = np.array(self.ext.region_map[3:5]).sum(axis=0)
            else:
                _map = np.array(self.ext.region_map[5:6].sum(axis=0))
            if _map[col, row] == 0:
                # if self.ext.region_map[cls, col, row] == 0:
                detections['errors'][det, 0] = 1
                wl = {'left': detections['2d_bbox_left'][det], 'top': detections['2d_bbox_top'][det],
                      'right': detections['2d_bbox_right'][det], 'bottom': detections['2d_bbox_bottom'][det]}
                wrong_loc.append(wl)
                valid_count -= 1
        if valid_count < detections['num_objects']:
            return wrong_loc, detections
        return None, detections

    # def _detect_cr(self):
    #     pass

    def _detect_us(self, detections, ep=0.7, ep2=0.5):
        valid_count = detections['num_objects']
        wrong_size = []
        for det in range(detections['num_objects']):
            cls = detections['label'][det]
            cx = (detections['2d_bbox_left'][det] + detections['2d_bbox_right'][det]) / 2
            cy = (detections['2d_bbox_top'][det] + detections['2d_bbox_bottom'][det]) / 2
            width = detections['2d_bbox_right'][det] - detections['2d_bbox_left'][det]
            height = detections['2d_bbox_bottom'][det] - detections['2d_bbox_top'][det]
            ratio = width / height
            col = int(cx / self.ext._x_batch)
            row = int(cy / self.ext._y_batch)
            _limit = self.ext.bbox_limit[cls, col + row * self.ext._cols]
            if np.sum(_limit) != 0 and (width < _limit[0, 0] * (1 - ep) or width > _limit[0, 1] * (1 + ep) \
                                        or height < _limit[1, 0] * (1 - ep) or height > _limit[1, 1] * (1 + ep) \
                                        or ratio < _limit[2, 0] * (1 - ep2) or ratio > _limit[2, 1] * (1 + ep2)):
                detections['errors'][det, 1] = 1
                us = {'left': detections['2d_bbox_left'][det], 'top': detections['2d_bbox_top'][det],
                      'right': detections['2d_bbox_right'][det], 'bottom': detections['2d_bbox_bottom'][det]}
                wrong_size.append(us)
                valid_count -= 1
        if valid_count < detections['num_objects']:
            return wrong_size, detections
        return None, detections

    def _detect_sc(self, detections):
        return abs(detections['num_objects'] - self._last_objects)
        # return np.exp(self.ext.kde.score_samples([[abs(detections['num_objects'] - self._last_objects)]]))[0]

    # def _detect_so(self):
    #     pass

    def _detect_lo(self, predictions, match_result):
        t0 = time()
        matches = np.sum(match_result, axis=1)
        # print(np.where(matches == 0)[0])
        ret_flag = False
        lost = []
        if np.where(matches == 0)[0].size == 0:
            return None
        t1 = time()
        # print('lo1:', t1 - t0)
        for track in np.where(matches == 0)[0]:
            t2 = time()
            # print('lo2:', t2 - t1)
            cx = (predictions[track][0] + predictions[track][2]) / 2
            cy = (predictions[track][1] + predictions[track][3]) / 2

            col = int(cx / self.ext._x_batch)
            row = int(cy / self.ext._y_batch)
            if self._tracks[track]['miss_age'] < 1 and 0 < col < self.ext._cols - 1 and 0 < row < self.ext._rows - 1:
                lost_track = {'id': self._tracks[track]['id'], 'left': predictions[track][0],
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

    # @jit
    def _detect_ls(self, detections, match_result):
        t0 = time()
        if detections['num_objects'] > 0:
            matches = np.sum(match_result, axis=1)
            if np.where(matches == 1)[0].size == 0:
                return None, detections
            switch = []
            for track in np.where(matches == 1)[0]:
                index = np.where(match_result[track] == 1)[0]
                # print(index)
                if detections['label'][index] != self._tracks[track]['label'][-1]:
                    detections['errors'][index, 2] = 1
                    label_switch = {'id': self._tracks[track]['id'], 'left': detections['2d_bbox_left'][index],
                                    'top': detections['2d_bbox_top'][index],
                                    'right': detections['2d_bbox_right'][index],
                                    'bottom': detections['2d_bbox_bottom'][index]}
                    switch.append(label_switch)
            if switch:
                # print('ls:', time() - t0)
                return switch, detections
        # print('ls:', time() - t0)
        return None, detections

    # def _detect_uo(self):
    #     pass

    def static_anomaly_detect(self, detections):
        warn1, det1 = self._detect_wl(detections)
        warn2, det2 = self._detect_us(det1)
        warns = {'wrong_location': warn1, 'unusual_size': warn2}
        return warns, det2

    def temporal_anomaly_detect(self, detections, predictions, match_result):
        warn1, det1 = self._detect_ls(detections, match_result)
        warns = {'sharp_change': self._detect_sc(detections),
                 'lost_object': self._detect_lo(predictions, match_result),
                 'label_switch': warn1}
        return warns, det1

    # @jit
    def init(self, frame, detections, file_path: str, rows: int, cols: int, labels):
        # construct prior knowledge
        self.ext = Extractor([self._frame_width, self._frame_height], rows=rows, cols=cols, labels=labels)
        self.ext.parse(file_path)
        self.ext.construct()
        # first frame track construction
        num_objects = detections['num_objects']
        for obj in range(num_objects):
            track = {'id': self._object_count, 'label': [], 'score': [],
                     'tracker': KCFTracker(True, True, True), 'appear': self._frame_count, 'miss_age': 0,
                     'left': detections['2d_bbox_left'][obj], 'top': detections['2d_bbox_top'][obj],
                     'right': detections['2d_bbox_right'][obj], 'bottom': detections['2d_bbox_bottom'][obj]}
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
        # print(self._tracks)

    # @jit
    def trackMatch(self, detections, predictions):
        """
        Match predictions from tracks and current detections
        :param detections: dict
        :param predictions: dict
        :return: match matrix with
        """
        # Get center points
        # print(detections['num_objects'])
        cp_pred = np.zeros([self._current_tracks, 2])
        miss_ages = np.zeros(self._current_tracks, dtype=int)
        diag_dist = np.zeros(self._current_tracks)
        cp_det = np.zeros([detections['num_objects'], 2])
        bbox_pred = np.zeros([self._current_tracks, 4], dtype=float)
        bbox_det = np.zeros([detections['num_objects'], 4], dtype=float)
        match_matrix = np.zeros([self._current_tracks, detections['num_objects']], dtype=int)
        for track in range(self._current_tracks):
            miss_ages[track] = self._tracks[track]['miss_age']
            bbox_pred[track][0] = predictions[track][0]
            bbox_pred[track][1] = predictions[track][1]
            bbox_pred[track][2] = predictions[track][2]
            bbox_pred[track][3] = predictions[track][3]
            cp_pred[track][0] = (predictions[track][0] + predictions[track][2]) / 2
            cp_pred[track][1] = (predictions[track][1] + predictions[track][3]) / 2
            # print(np.array([predictions[track][0], predictions[track][1]]), cp_pred[track])
            diag_dist[track] = pairwise_distances(np.array([[predictions[track][0], predictions[track][1]]]),
                                                  cp_pred[track:track + 1])

        for obj in range(detections['num_objects']):
            bbox_det[obj][0] = detections['2d_bbox_left'][obj]
            bbox_det[obj][1] = detections['2d_bbox_top'][obj]
            bbox_det[obj][2] = detections['2d_bbox_right'][obj]
            bbox_det[obj][3] = detections['2d_bbox_bottom'][obj]
            cp_det[obj][0] = (detections['2d_bbox_left'][obj] + detections['2d_bbox_right'][obj]) / 2
            cp_det[obj][1] = (detections['2d_bbox_top'][obj] + detections['2d_bbox_bottom'][obj]) / 2
        # print(cp_pred.shape)

        # Compute prediction-detection bbox distance
        # print(cp_det.shape, cp_pred.shape)
        dist = pairwise_distances(cp_pred, cp_det, metric='euclidean')

        # Compute prediction-detection ious
        # ious = matrix_iou(bbox_pred, bbox_det)

        # Compute width-height ratio
        # ratio = wh_ratio(bbox_pred, bbox_det)

        # Compute priority score
        pr_score = np.squeeze(
            pairwise_distances(cp_pred, np.array([[self._frame_width / 2, self._frame_height / 2]]),
                               metric='euclidean'), axis=-1)
        # print(pr_score.shape)
        pr_score = 1 - pr_score / math.sqrt(
            (self._frame_width / 2) * (self._frame_width / 2) + (self._frame_height / 2 * self._frame_height / 2))
        # print(pr_score)
        # for track in range(self._current_tracks):
        #     obj_index = np.argmin(dist[track])
        #     if dist[track][obj_index] <= diag_dist[track] and np.sum(match_matrix, axis=0)[obj_index] == 0:
        #         match_matrix[track][obj_index] = 1
        for age in range(self._max_age + 1):
            indices = np.where(miss_ages == age)[0]
            # print(indices)
            if indices.shape[0] > 0:
                temp_pr_score = pr_score[indices]
                # print(temp_pr_score)
                pr_score_sort = np.argsort(temp_pr_score)
                # print(pr_score_sort)
                for rank in range(indices.shape[0]):
                    origin_index = indices[pr_score_sort[rank]]
                    # print(origin_index)
                    # print(np.where(dist[origin_index] <= diag_dist[origin_index])[0])
                    valid_detections = np.where(dist[origin_index] <= diag_dist[origin_index])[0]
                    print(valid_detections)
                    if valid_detections.shape[0] > 0:
                        ious = np.zeros(valid_detections.shape[0])
                        # print(ious)
                        for obj in range(valid_detections.shape[0]):
                            ious[obj] = intersection_over_union(predictions[origin_index], np.array(
                                [detections['2d_bbox_left'][valid_detections[obj]],
                                 detections['2d_bbox_top'][valid_detections[obj]],
                                 detections['2d_bbox_right'][valid_detections[obj]],
                                 detections['2d_bbox_bottom'][valid_detections[obj]]]))
                        print(ious)
                        candidates = np.argsort(ious)
                        print(candidates)
                        for obj in range(-1, -(candidates.shape[0] + 1), -1):
                            candidate = valid_detections[candidates[obj]]
                            print(obj, candidate)
                            width_pred = bbox_pred[origin_index, 2] - bbox_pred[origin_index, 0]
                            width_det = bbox_det[candidate, 2] - bbox_det[candidate, 0]
                            height_pred = bbox_pred[origin_index, 3] - bbox_pred[origin_index, 1]
                            height_det = bbox_det[candidate, 3] - bbox_det[candidate, 1]
                            if np.sum(match_matrix, axis=0)[candidate] == 0 \
                                    and ious[candidates[obj]] >= self._min_iou \
                                    and _ratio(width_pred, width_det) > self._min_ratio \
                                    and _ratio(height_pred, height_det) > self._min_ratio \
                                    and abs(width_pred / height_pred - width_det / height_det) < 0.5\
                                    and np.argwhere(np.argsort(dist[:, candidate]) == origin_index)[0][0] < 2:
                                print(origin_index)
                                print(np.argmin(dist[:, candidate]))
                                print(np.argsort(dist[:, candidate]))
                                print(_ratio(width_pred, width_det), _ratio(height_pred, height_det), width_pred / height_pred - width_det / height_det)
                                match_matrix[origin_index][candidate] = 1
                                break
                    else:
                        continue
            else:
                continue

        return match_matrix

    # @jit
    def update(self, frame, detections):
        self._frame_count += 1
        print('frame:', frame.shape[1], frame.shape[0])
        # Tracker new location predict
        t0 = time()
        pred_locs = []
        print('tracks:', self._current_tracks, 'detections:', detections['num_objects'])
        detections['errors'] = np.zeros((detections['num_objects'], 3), dtype=np.int32)
        for track in range(self._current_tracks):
            loc = self._tracks[track]['tracker'].get_pred(frame)
            loc[2] += loc[0]
            loc[3] += loc[1]
            # if loc[2] > frame.shape[1]:
            #     loc[2] = frame.shape[1]
            # if loc[3] > frame.shape[0]:
            #     loc[3] = frame.shape[0]
            pred_locs.append(loc)
        pred_locs = np.array(pred_locs)
        t1 = time()
        print('track predict: {}s'.format(t1 - t0))
        duration0 = t1 - t0
        if detections['num_objects'] > 0:
            warns1, detections = self.static_anomaly_detect(detections)
        else:
            warns1 = {'wrong_location': None, 'unusual_size': None}
        t2 = time()
        print('static: {}s'.format(t2 - t1))
        duration1 = t2 - t1
        if detections['num_objects'] > 0 and self._current_tracks > 0:
            match_matrix = self.trackMatch(detections, pred_locs)
        elif detections['num_objects'] == 0:
            match_matrix = np.zeros([self._current_tracks, 1], dtype=int)
        else:
            match_matrix = np.zeros([1, detections['num_objects']], dtype=int)
        t3 = time()
        print('match: {}s'.format(t3 - t2))
        duration2 = t3 - t2
        if self._current_tracks > 0:
            warns2, detections = self.temporal_anomaly_detect(detections, pred_locs, match_matrix)
        else:
            warns2 = {'sharp_change': None, 'lost_object': None, 'label_switch': None}
        t4 = time()
        print('temporal: {}s'.format(t4 - t3))
        duration3 = t4 - t3
        self._last_objects = detections['num_objects']
        sum_track = np.sum(match_matrix, axis=1)
        sum_det = np.sum(match_matrix, axis=0)

        # match update
        if self._current_tracks > 0 and detections['num_objects'] > 0:
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
                # print(len(self._tracks))
                # print(self._tracks[track])
                if self._max_age <= self._tracks[track]['miss_age']: \
                        # or (self._tracks[track]['left'] +self._tracks[track]['right'])/2 < self._limits[0] * self._frame_width  \
                    # or (self._tracks[track]['left'] +self._tracks[track]['right'])/2 > self._limits[2] * self._frame_width:
                    track_to_delete.append(track)
                else:
                    self._tracks[track]['left'] = pred_locs[track][0]
                    self._tracks[track]['top'] = pred_locs[track][1]
                    self._tracks[track]['right'] = pred_locs[track][2]
                    self._tracks[track]['bottom'] = pred_locs[track][3]
                    # print('this', self._tracks[track]['tracker']._roi)
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
