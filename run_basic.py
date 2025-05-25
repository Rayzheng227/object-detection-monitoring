import numpy as np
import cv2
import sys
from time import time
import os

import src.tracker.kcftracker_mod as kcftracker_mod
from src.monitors.anomaly_vmonitor import Monitor

#在img上绘制pt1和pt2的连线
def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=5):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5#pt1和pt2的距离
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)#pt1和pt2之间固定间隔的点的list


    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

#绘制多边形
def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)

#绘制矩形
def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)


initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 100
duration = 0.01

font = cv2.FONT_HERSHEY_TRIPLEX
labels = ['car', 'van', 'truck', 'pedestrian', 'cyclist', 'tram', 'dontcare']
colors = {'car': (0, 255, 0), 'van': (255, 255, 0), 'truck': (212, 255, 127),
          'pedestrian': (205, 235, 255), 'cyclist': (105, 168, 227), 'tram': (135, 184, 222)}

if __name__ == '__main__':

    # if (len(sys.argv) == 1):
    #     cap = cv2.VideoCapture(0)
    # elif (len(sys.argv) == 2):
    #     if (sys.argv[1].isdigit()):  # True if sys.argv[1] is str of a nonnegative integer
    #         cap = cv2.VideoCapture(int(sys.argv[1]))
    #     else:re
    #         cap = cv2.VideoCapture(sys.argv[1])
    #         inteval = 100
    # else:
    #     assert (0), "too many arguments"
    # -----------------------------------------------------------------------------------------------
    # 导入视频
    video_name = '0000'
    cap = cv2.VideoCapture('./data/videos/{}.mp4'.format(video_name))

    cv2.namedWindow('tracking')

    initTracking = True

    # 导入标签
    label_dir = './data/detections/detection_{}.txt'.format(video_name)
    # label_dir = './label_02/{}.txt'.format(video_name)
    print(label_dir)
    with open(label_dir) as f:
        content = f.readlines()
    # print(content)
    content = [x.strip().split(' ') for x in content]
    detections = []
    max_frame = int(content[-1][0]) + 1
    print(max_frame)
    fig = np.zeros((375, 1242, 3), dtype=np.uint8)
    # fig.fill(255)
    count = []
    # 读取每一帧
    for frame_id in range(max_frame):
        detect = {}
        _label = []
        _2d_bbox_left = []
        _2d_bbox_top = []
        _2d_bbox_right = []
        _2d_bbox_bottom = []
        _score = []
        num_count = 0
        # 包含每个框的位置（上下左右）
        for x in content:
            if int(x[0]) == frame_id and float(x[6]) >= 0.5: #
                _label.append(int(x[1]) - 1)
                _2d_bbox_left.append(float(x[2]))
                _2d_bbox_top.append(float(x[3]))
                _2d_bbox_right.append(float(x[4]))
                _2d_bbox_bottom.append(float(x[5]))
                _score.append(float(x[6]))
                num_count += 1
                cv2.circle(fig,
                           (int((float(x[2]) + float(x[4])) / 2 + 0.5), int((float(x[3]) + float(x[5])) / 2 + 0.5)), 2,
                           colors[labels[int(x[1]) - 1]], 0)
                # plt.scatter((float(x[2])+float(x[4]))/2, (float(x[3])+float(x[5]))/2)
            else:
                continue
        count.append(num_count)
        detect['num_objects'] = int(num_count)
        detect['label'] = np.array(_label)
        detect['2d_bbox_left'] = np.array(_2d_bbox_left)
        detect['2d_bbox_top'] = np.array(_2d_bbox_top)
        detect['2d_bbox_right'] = np.array(_2d_bbox_right)
        detect['2d_bbox_bottom'] = np.array(_2d_bbox_bottom)
        detect['score'] = np.array(_score)
        detections.append(detect)
    cv2.imwrite('density{}.png'.format(video_name), fig)
    print('max_count', max(count))
    # print(annotations[0]['label'].shape)

    # ix = int(detections[0]['2d_bbox_left'][1] + 0.5)
    # iy = int(detections[0]['2d_bbox_top'][1] + 0.5)
    # w = int(detections[0]['2d_bbox_right'][1] - detections[0]['2d_bbox_left'][1] + 0.5)
    # h = int(detections[0]['2d_bbox_bottom'][1] - detections[0]['2d_bbox_top'][1] + 0.5)

    monitor = Monitor(frame_size=(1242, 374), max_to_keep=100, max_age=2)
    current_frame = 0
    duration = []
    overhead = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # index = np.where(detections[current_frame]['label'] == 5)
        if initTracking:
            # for obj in range(detections[current_frame]['num_objects']):
            #     label_len = len(labels[int(detections[current_frame]['label'][obj])] + ' ' + '{:.3f}'.format(
            #         detections[current_frame]['score'][obj]))
            #     cv2.rectangle(frame, (int(detections[current_frame]['2d_bbox_left'][obj] + 0.5),
            #                           int(detections[current_frame]['2d_bbox_top'][obj] + 0.5)),
            #                   (int(detections[current_frame]['2d_bbox_right'][obj] + 0.5),
            #                    int(detections[current_frame]['2d_bbox_bottom'][obj] + 0.5)),
            #                   colors[labels[int(detections[current_frame]['label'][obj])]], 2)
            #                   # (0, 255, 0), 2)
            #     cv2.rectangle(frame, (int(detections[current_frame]['2d_bbox_left'][obj] + 0.5) - 1,
            #                           int(detections[current_frame]['2d_bbox_top'][obj] + 0.5) - 15),
            #                   (int(detections[current_frame]['2d_bbox_left'][obj] + 0.5) - 2 + int(label_len * 7.5),
            #                    int(detections[current_frame]['2d_bbox_top'][obj] + 0.5)),
            #                   colors[labels[int(detections[current_frame]['label'][obj])]], -1)
            #                   # (0, 255, 0), -1)
            #     cv2.putText(frame, labels[int(detections[current_frame]['label'][obj])] + ' ' + '{:.3f}'.format(
            #         detections[current_frame]['score'][obj]),
            #                 (int(detections[current_frame]['2d_bbox_left'][obj] + 0.5),
            #                  int(detections[current_frame]['2d_bbox_top'][obj] + 0.5) - 5), font, 0.4, (0, 0, 0), 1)
            monitor.init(frame, detections[current_frame], file_path='./data/labels/label_02-bak', cols=9, rows=6,
                         labels=['car', 'van', 'truck', 'pedestrian', 'cyclist', 'tram', 'dontcare'])
            initTracking = False
            onTracking = True
        elif onTracking:
            t0 = time()
            warns1, warns2, dets, oh = monitor.update(frame, detections[current_frame])
            # dets['errors'] = np.zeros((detections[current_frame]['num_objects'], 3), dtype=np.int32)
            print('warns1', warns1, '\n', 'warns2', warns2)
            t1 = time()
            for obj in range(dets['num_objects']):
                dot_colors = [(0, 0, 255), (255, 111, 131), (225, 105, 65)]
                label_len = len(labels[int(dets['label'][obj])] + ' ' + '{:.3f}'.format(
                    dets['score'][obj]))
#TODO from here
                # if dets['errors'][obj].sum() == 0:
                #     # pass
                #     cv2.rectangle(frame, (int(dets['2d_bbox_left'][obj] + 0.5),
                #                           int(dets['2d_bbox_top'][obj] + 0.5)),
                #                   (int(dets['2d_bbox_right'][obj] + 0.5),
                #                    int(dets['2d_bbox_bottom'][obj] + 0.5)),
                #                   colors[labels[int(dets['label'][obj])]], 2)
                #                   # (0, 255, 0), 2)
                #     cv2.rectangle(frame, (int(dets['2d_bbox_left'][obj] + 0.5) - 1,
                #                           int(dets['2d_bbox_top'][obj] + 0.5) - 20),
                #                   (int(dets['2d_bbox_left'][obj] + 0.5) - 2 + int(label_len * 12),
                #                    int(dets['2d_bbox_top'][obj] + 0.5)),
                #                   colors[labels[int(dets['label'][obj])]], -1)
                #                   # (0, 255, 0), -1)
                #     cv2.putText(frame, labels[int(dets['label'][obj])] + ' ' + '{:.3f}'.format(
                #         dets['score'][obj]),
                #                 (int(dets['2d_bbox_left'][obj] + 0.5),
                #                  int(dets['2d_bbox_top'][obj] + 0.5) - 5), font, 0.6, (0, 0, 0), 1)
                # if dets['errors'][obj][0] == 1:
                #     # cv2.circle(frame, (int(dets['2d_bbox_left'][obj] + 0.5) + 5, int(dets['2d_bbox_top'][obj] + 0.5) + 5 + error_count*10), 5, dot_colors[i], -1)
                #     cv2.rectangle(frame, (int(dets['2d_bbox_left'][obj] + 0.5) - 1,
                #                           int(dets['2d_bbox_top'][obj] + 0.5) - 20),
                #                   (int(dets['2d_bbox_left'][obj] + 0.5) - 2 + int(
                #                       label_len * 12), int(dets['2d_bbox_top'][obj] + 0.5)),
                #                   colors[labels[int(dets['label'][obj])]], -1)
                #     cv2.rectangle(frame,
                #                   (int(dets['2d_bbox_left'][obj] + 0.5), int(dets['2d_bbox_top'][obj] + 0.5)),
                #                   (int(dets['2d_bbox_right'][obj] + 0.5), int(dets['2d_bbox_bottom'][obj] + 0.5)),
                #                   dot_colors[0], 2)

                #     cv2.putText(frame, labels[int(dets['label'][obj])] + ' ' + '{:.3f}'.format(
                #                 dets['score'][obj]),
                #                 (int(dets['2d_bbox_left'][obj] + 0.5),
                #                  int(dets['2d_bbox_top'][obj] + 0.5) - 5), font, 0.6,
                #                 (0, 0, 0), 1)
                # if dets['errors'][obj][1] == 1:
                #     cv2.rectangle(frame,
                #                   (int(dets['2d_bbox_left'][obj] + 0.5), int(dets['2d_bbox_top'][obj] + 0.5)),
                #                   (int(dets['2d_bbox_right'][obj] + 0.5), int(dets['2d_bbox_bottom'][obj] + 0.5)),
                #                   dot_colors[1], 2)
                #     cv2.rectangle(frame, (int(dets['2d_bbox_left'][obj] + 0.5) - 1,
                #                           int(dets['2d_bbox_top'][obj] + 0.5) - 20),
                #                   (int(dets['2d_bbox_left'][obj] + 0.5) - 2 + int(
                #                       label_len * 12), int(dets['2d_bbox_top'][obj] + 0.5)),
                #                   dot_colors[1], -1)
                #     if dets['errors'][obj][2] == 0:
                #         cv2.putText(frame, labels[int(dets['label'][obj])] + ' ' + '{:.3f}'.format(
                #                     dets['score'][obj]),
                #                     (int(dets['2d_bbox_left'][obj] + 0.5),
                #                      int(dets['2d_bbox_top'][obj] + 0.5) - 5), font, 0.6,
                #                     (255, 255, 255), 1)
                # if dets['errors'][obj][2] == 1:
                #     cv2.rectangle(frame, (int(dets['2d_bbox_left'][obj] + 0.5),
                #                           int(dets['2d_bbox_top'][obj] + 0.5)),
                #                   (int(dets['2d_bbox_right'][obj] + 0.5),
                #                    int(dets['2d_bbox_bottom'][obj] + 0.5)),
                #                   colors[labels[int(dets['label'][obj])]], 2)
                #                   # (0, 255, 0), 2)
                #     cv2.rectangle(frame, (int(dets['2d_bbox_left'][obj] + 0.5) - 1,
                #                           int(dets['2d_bbox_top'][obj] + 0.5) - 20),
                #                   (int(dets['2d_bbox_left'][obj] + 0.5) - 2 + int(label_len * 12) + 8,
                #                    int(dets['2d_bbox_top'][obj] + 0.5)),
                #                   colors[labels[int(dets['label'][obj])]], -1)
                #                   # (0, 255, 0), -1)
                #     cv2.circle(frame, (int(dets['2d_bbox_left'][obj] + 0.5) + 4,
                #                int(dets['2d_bbox_top'][obj] + 0.5) - 8), 4, (0, 0, 255), -1)
                #     if dets['errors'][obj][1] == 0:
                #         cv2.putText(frame, labels[int(dets['label'][obj])] + ' ' + '{:.3f}'.format(
                #             dets['score'][obj]),
                #                     (int(dets['2d_bbox_left'][obj] + 0.5) + 8,
                #                      int(dets['2d_bbox_top'][obj] + 0.5) - 5), font, 0.6, (0, 0, 0), 1)
                #     else:
                #         cv2.putText(frame, labels[int(dets['label'][obj])] + ' ' + '{:.3f}'.format(
                #             dets['score'][obj]),
                #                     (int(dets['2d_bbox_left'][obj] + 0.5) + 8,
                #                      int(dets['2d_bbox_top'][obj] + 0.5) - 5), font, 0.6, (0, 0, 0), 1)
                        #TODO end here
            # if warns1['wrong_location'] is not None:
            #     for obj in warns1['wrong_location']:
            #         # cv2.rectangle(frame, (int(obj['left'] + 0.5), int(obj['top'] + 0.5)),
            #         #               (int(obj['right'] + 0.5), int(obj['bottom'] + 0.5)), (0, 0, 255), 2)
            #         cv2.putText(frame, 'wl', (int(obj['left'] + 0.5), int(obj['top'] + 0.5) + 10), font, 0.5,
            #                     (0, 0, 255), 2)
            # if warns1['unusual_size'] is not None:
            #     for obj in warns1['unusual_size']:
            #         # cv2.rectangle(frame, (int(obj['left'] + 0.5), int(obj['top'] + 0.5)),
            #         #               (int(obj['right'] + 0.5), int(obj['bottom'] + 0.5)), (0, 0, 255), 2)
            #         cv2.putText(frame, 'us', (int(obj['left'] + 0.5), int(obj['top'] + 0.5) + 20), font, 0.5,
            #                     (0, 0, 255), 2)
            if warns2['lost_object'] is not None:
                for obj in warns2['lost_object']:
                    # cv2.rectangle(frame, (int(obj['left'] + 0.5), int(obj['top'] + 0.5)),
                    #               (int(obj['right'] + 0.5), int(obj['bottom'] + 0.5)), (0, 0, 255), 2)
                    drawrect(frame, (int(obj['left'] + 0.5), int(obj['top'] + 0.5)),
                             (int(obj['right'] + 0.5), int(obj['bottom'] + 0.5)), (0, 0, 255), 2)
            # if warns2['label_switch'] is not None:
            #     for obj in warns2['label_switch']:
            #         cv2.putText(frame, 'ls', (int(obj['left']+0.5), int(obj['top']+0.5)+30), font, 0.5, (0, 0, 255), 2)

            print('current_frame:', current_frame)
            # for i in range(9):
            #     cv2.line(frame, (int(i*1242/9+0.5), 0), (int(i*1242/9+0.5), 374), (0, 255, 0))
            # for i in range(6):
            #     cv2.line(frame, (0, int(i*374/6+0.5)), (1241, int(i*374/6+0.5)), (0, 255, 0))

            # duration = 0.8 * duration + 0.2 * (t1 - t0)
            if current_frame > 3:
                duration.append(t1 - t0)
                overhead.append(oh)
            print('duration: {}s\n'.format(t1 - t0))
            # cv2.putText(frame, 'SC: ' + str(warns2['sharp_change']), (8, 20), font, 0.6,
            #             (0, 0, 255), 2)

        cv2.imshow('tracking', frame)
        cv2.imwrite('./results1/{}/{}.png'.format(video_name, current_frame), frame)
        current_frame += 1
        c = cv2.waitKey(inteval) & 0xFF
        if c == 27 or c == ord('q'):
            break
        if current_frame == max_frame:
            break
    duration = np.array(duration)
    print(np.min(duration), np.max(duration), np.mean(duration))
    print(np.argmax(duration))
    overhead = np.array(overhead)
    for i in range(7):
        print('{:.4f}'.format(np.min(overhead[:, i])), '{:.4f}'.format(np.max(overhead[:, i])),
              '{:.4f}'.format(np.mean(overhead[:, i])))
    cap.release()
    cv2.destroyAllWindows()
