import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit, njit
from sklearn.neighbors import KernelDensity


# Extractor class的作用是从KITTI数据集的label文件中提取出有用的信息，包括每个类别在每个区域的数量、每个区域的bbox大小、bbox的变化等等
class Extractor(object):
    def __init__(self, frame_size, rows: int, cols: int, labels: list):
        # 
        # protected
        # 初始化参数：帧的宽度、高度、行数、列数、每个批次的宽度和高度、类别标签
        assert frame_size[0] > 0 and frame_size[1] > 0
        self._frame_width = frame_size[0]
        self._frame_height = frame_size[1]
        assert rows > 0 and cols > 0
        self._rows = rows
        self._cols = cols
        self._x_batch = self._frame_width / self._cols
        self._y_batch = self._frame_height / self._rows
        self._labels = labels

        #初始化：变化统计、核密度估计、场景、地图map、区域地图region_map、转移矩阵transfer、边界框的尺寸及大小bbox_size、bbox_limit
        # public
        self.changes = None
        self.kde = None
        self.scenes = []
        self.map = np.zeros((len(self._labels), self._frame_width, self._frame_height), dtype=int)
        self.region_map = np.zeros((len(self._labels), self._cols, self._rows), dtype=int)
        self.transfer = np.zeros((len(self._labels), self._cols * self._rows, self._cols * self._rows), dtype=float)
        self.bbox_size = [[[[], [], []] for j in range(self._cols * self._rows)] for i in range(len(self._labels))]
        self.bbox_limit = np.zeros((len(self._labels), self._cols * self._rows, 3, 2), dtype=float)

    # @jit
    def parse(self, file_path: str):
        # parse函数的作用是从指定的文件路径中读取跟踪数据文件，解析每一帧中的对象信息，并将这些信息存储在一个结构化的字典中
        print(file_path)
        for file in os.listdir(file_path):
            with open(os.path.join(file_path, file)) as f:
                content = f.readlines()
            content = [x.strip().split(' ') for x in content]
            max_frame = int(content[-1][0]) + 1
            detections = {'max_frame': max_frame, 'obj_count': []}
            num_objects = 0
            for x in content:
                num_objects = max(num_objects, int(x[1]))
            # print(file, num_objects)
            detections['num_objects'] = num_objects
            tracks = [{'id': i, 'class': -1, 'frames': [], 'truncated': [],
                       'occluded': [], 'bbox': []} for i in range(num_objects + 1)]
            print(tracks)
            for frame_id in range(max_frame):
                obj_cnt = 0
                print('file', file, 'frame', frame_id)
                for x in content:
                    if int(x[0]) == frame_id and int(x[1]) != -1:
                        obj_cnt += 1

                        _frame = int(x[0])
                        _id = int(x[1])
                        _class = self._labels.index(str(x[2]).lower())
                        _truncated = int(x[3])
                        _occluded = int(x[4])
                        _bbox_left = float(x[6])
                        _bbox_top = float(x[7])
                        _bbox_right = float(x[8])
                        _bbox_bottom = float(x[9])

                        tracks[_id]['frames'].append(_frame)
                        tracks[_id]['class'] = _class
                        tracks[_id]['truncated'].append(_truncated)
                        tracks[_id]['occluded'].append(_occluded)
                        tracks[_id]['bbox'].append([_bbox_left, _bbox_top, _bbox_right, _bbox_bottom])
                detections['obj_count'].append(obj_cnt)
            detections['tracks'] = tracks
            self.scenes.append(detections)

    # @jit
    def construct(self):
        # Construct通过遍历解析的数据，计算对象数量的变化、对象的空间分布、对象的转移矩阵、对象的边界框尺寸等统计信息，并将这些信息存储在相应的数据结构中。
        _changes = []
        _max = []
        for scene in self.scenes:
            _changes.append(np.abs(np.array(scene['obj_count'][1:]) - np.array(scene['obj_count'][:-1])))
            _max.append(np.max(np.abs(np.array(scene['obj_count'][1:]) - np.array(scene['obj_count'][:-1]))))
            for track in scene['tracks']:
                f_cx = None
                f_cy = None
                for i in range(len(track['bbox'])):
                    bbox = track['bbox'][i]
                    # map
                    cx = int((bbox[0] + bbox[2]) / 2 + 0.5)
                    cy = int((bbox[1] + bbox[3]) / 2 + 0.5)
                    self.map[track['class'], cx, cy] += 1
                    # transfer
                    col = int(cx / self._x_batch)
                    row = int(cy / self._y_batch)
                    if f_cx is not None and f_cy is not None:
                        f_col = int(f_cx / self._x_batch)
                        f_row = int(f_cy / self._y_batch)
                        self.transfer[track['class'], f_col + f_row * self._cols, col + row * self._cols] += 1
                    f_cx = cx
                    f_cy = cy
                    # bbox
                    if track['truncated'][i] < 2:
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        ratio = width / height
                        self.bbox_size[track['class']][col + row * self._cols][0].append(width)
                        self.bbox_size[track['class']][col + row * self._cols][1].append(height)
                        self.bbox_size[track['class']][col + row * self._cols][2].append(ratio)

        # region_map
        for j in range(self._rows):
            for k in range(self._cols):
                self.region_map[:, k, j] = np.sum(self.map[:, int(k * self._x_batch): int((k + 1) * self._x_batch),
                                                  int(j * self._y_batch): int((j + 1) * self._y_batch)], axis=(1, 2))

        # sharp change stats
        self.changes = np.zeros(max(_max) + 1, dtype=int)
        _combination = np.zeros(0, dtype=int)
        for change in _changes:
            _combination = np.concatenate((_combination, change), axis=0)
            for i in range(change.shape[0]):
                self.changes[change[i]] += 1
        self.kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(_combination.reshape(-1, 1))
        print(np.exp(self.kde.score_samples([[1]])))

        # bbox limit
        for i in range(len(self._labels)):
            for j in range(self._cols * self._rows):
                if self.bbox_size[i][j][0]:
                    self.bbox_limit[i, j, 0, 0] = min(self.bbox_size[i][j][0])
                    self.bbox_limit[i, j, 0, 1] = max(self.bbox_size[i][j][0])
                    self.bbox_limit[i, j, 1, 0] = min(self.bbox_size[i][j][1])
                    self.bbox_limit[i, j, 1, 1] = max(self.bbox_size[i][j][1])
                    self.bbox_limit[i, j, 2, 0] = min(self.bbox_size[i][j][2])
                    self.bbox_limit[i, j, 2, 1] = max(self.bbox_size[i][j][2])
                else:
                    self.bbox_limit[i, j, 0, 0] = 0
                    self.bbox_limit[i, j, 0, 1] = 0
                    self.bbox_limit[i, j, 1, 0] = 0
                    self.bbox_limit[i, j, 1, 1] = 0
                    self.bbox_limit[i, j, 2, 0] = 0
                    self.bbox_limit[i, j, 2, 1] = 0

        # transfer rate


if __name__ == '__main__':
    start = time.time()
    ext = Extractor(frame_size=(1242, 374), rows=6, cols=9,
                    labels=['car', 'van', 'truck', 'pedestrian', 'cyclist', 'tram', 'dontcare'])
    ext.parse('./label_02-bak/')
    ext.construct()
    sns.set(font='Tahoma', font_scale=2)
    f, ax = plt.subplots(figsize=(12.42, 3.74))
    ax = sns.heatmap(np.sum(ext.region_map[:, :, :], axis=0).transpose(1, 0), cmap='jet', ax=ax)
    ax.set_xlabel('column')
    ax.set_ylabel('row')
    plt.show()
    plt.close()
    ax2 = sns.heatmap(np.sum(ext.region_map[3:5, :, :], axis=0).transpose(1, 0), cmap='jet')
    ax2.set_xlabel('column')
    ax2.set_ylabel('row')
    plt.show()
    plt.close()
    ax3 = sns.heatmap(np.sum(ext.region_map[5:6, :, :], axis=0).transpose(1, 0), cmap='jet')
    ax3.set_xlabel('column')
    ax3.set_ylabel('row')
    plt.show()
    plt.close()
    print('{:.4f}s'.format(time.time() - start))
    print()
