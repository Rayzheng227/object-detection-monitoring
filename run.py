import numpy as np
import cv2
import sys
from time import time
import os

import src.tracker.kcftracker_mod as kcftracker_mod
from src.monitors.vmonitor import Monitor



# Draw line between pt1 and pt2 with specified style
def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=5):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)


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

# Draw polygon
def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)

# Draw rectangle
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
labels = ['car', 'van', 'truck','pedestrian' ,'person_sitting' , 'cyclist', 'tram', 'dontcare']
colors = {
    'car': (0, 255, 0), 
    'van': (255, 255, 0), 
    'truck': (212, 255, 127),
    'pedestrian': (205, 235, 255), 
    'person_sitting': (255, 192, 203),  
    'cyclist': (105, 168, 227), 
    'tram': (135, 184, 222),
    'dontcare': (128, 128, 128) 
}
if __name__ == '__main__':

    '''
    step1.Load video and labels
    '''
    # ------------------------------------
    video_name = '0000'
    video_dir = os.path.join(os.getcwd(), './data/videos', '{}.mp4'.format(video_name))

    # cap = cv2.VideoCapture('E:/Tracking/my_data/{}.mp4'.format(video_name))
    if not os.path.exists(video_dir):
        # print(video_dir)
        print("Video file does not exist.")
    else:
        cap = cv2.VideoCapture(video_dir)
        # print(cap.isOpened())
    # cap=cv2.VideoCapture('./{}.mp4'.format(video_name))
    # print(cap.isOpened())
    ret, pre_frame = cap.read() # ret is a bool, if the frame is read correctly, it is True; pre_frame is the frame
    # print(ret) # whether the frame is read correctly
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {total_frames}")   # TOTAL FRAME =154

    cv2.namedWindow('tracking')

    initTracking = True

    # label_dir = 'E:/Tracking/my_data/label_0000.txt'
    label_dir = './data/detections/detection_{}.txt'.format(video_name)
    # label_dir = './label_0000.txt'
    with open(label_dir) as f:
        content = f.readlines()

    content = [x.strip().split(' ') for x in content] # split the content by space 
    print(content[0]) # print the first line of the content
    # the first line of the content is:['0', '2', '296.744956', '161.752147', '455.226042', '292.372804', '0.99', '0']
    # the elements of content is [frame_id, label, (left,top,right,bottom),score]
    
    # --------------------------------------

    detections = [] 

    max_frame = int(content[-1][0]) + 1  
  


    fig = np.zeros((375, 1242, 3), dtype=np.uint8) 

    count = []
    for frame_id in range(max_frame):
        detect = {}
        _label = []
        _2d_bbox_left = []
        _2d_bbox_top = []
        _2d_bbox_right = []
        _2d_bbox_bottom = []
        _score = []
        num_count = 0
        for x in content:
            if int(x[0]) == frame_id and float(x[6]) >= 0.5: # Filter by confidence score >= 0.5
                _label.append(int(x[1]) - 1)
                _2d_bbox_left.append(float(x[2]))
                _2d_bbox_top.append(float(x[3]))
                _2d_bbox_right.append(float(x[4]))
                _2d_bbox_bottom.append(float(x[5]))
                _score.append(float(x[6]))
                num_count += 1
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


    '''
    step2.Initialize monitor with frame size (1242, 374), max 100 frames, max age 3s
    '''

    monitor = Monitor(frame_size=(1242, 374), max_to_keep=100, max_age=3)
    print('init monitor')
    current_frame = 0
    duration = []
    overhead = []
    temp_warns2 = {'lost_object':None}
    # print(cap.isOpened())
    while cap.isOpened():
        ret, frame = cap.read()
        # print(ret)
        if not ret:
            print(f"Failed to read frame at index {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1)
            continue
            # break
        # index = np.where(detections[current_frame]['label'] == 5)
        if initTracking:
            # print('-------------------')
            # print('init tracking,current frame is: ',current_frame)
            monitor.init(frame, detections[current_frame])
            initTracking = False
            onTracking = True
        elif onTracking:
            # print('-------------------')
            # print("start tracking, current frame is",current_frame)
            t0 = time()
            #loss_infos = []
            # Handle lost objects by adding them to current frame detections
            if temp_warns2['lost_object'] is not None:
                # print('LOST OBJECT NUM IS:', len(temp_warns2['lost_object']))
                for loss_obj in temp_warns2['lost_object']:
                    #loss_info = []
                    '''
                    # XXX:OLD VERSION: USE np.concatenate() to add the lost object to the detections
                    np.concatenate(detections[current_frame]['2d_bbox_left'],np.array(int(loss_obj['left']+0.5)))
                    np.concatenate(detections[current_frame]['2d_bbox_top'],np.array(int(loss_obj['top']+0.5)))
                    np.concatenate(detections[current_frame]['2d_bbox_right'],np.array(int(loss_obj['right']+0.5)))
                    np.concatenate(detections[current_frame]['2d_bbox_bottom'],np.array(int(loss_obj['bottom']+0.5)))
                    #loss_infos.append(loss_info)
                detections[current_frame]['num_objects'] = detections[current_frame]['num_objects']+len(temp_warns2['lost_object'])
                '''
                # XXX:NEW VERSION：USE np.append() instead of np.concatenate()
                    # print('-------------------')
                    # print('lose infor:',loss_obj) # lose infor: {'id': 0, 'left': 326.25599371009906, 'top': 189.68469631114954, 'right': 411.1296057100991, 'bottom': 232.14777231114954}
                    detections[current_frame]['2d_bbox_left'] = np.append(detections[current_frame]['2d_bbox_left'], int(loss_obj['left'] + 0.5))
                    detections[current_frame]['2d_bbox_top'] = np.append(detections[current_frame]['2d_bbox_top'], int(loss_obj['top'] + 0.5))
                    detections[current_frame]['2d_bbox_right'] = np.append(detections[current_frame]['2d_bbox_right'], int(loss_obj['right'] + 0.5))
                    detections[current_frame]['2d_bbox_bottom'] = np.append(detections[current_frame]['2d_bbox_bottom'], int(loss_obj['bottom'] + 0.5))
                    detections[current_frame]['label'] = np.append(detections[current_frame]['label'], loss_obj['label'])
                    detections[current_frame]['score'] = np.append(detections[current_frame]['score'], loss_obj['score'])
                detections[current_frame]['num_objects'] += len(temp_warns2['lost_object'])
    

            #detections[current_frame].append()
            print('det num obj is:',detections[current_frame]['num_objects'] )
            # XXX: use TRY ACCEPT to find why out of bound error occurs:
            # TODO :Remeber to delete the try accept after the problem is solved
            try:
                warns1, warns2, dets, oh = monitor.update(pre_frame,frame, detections[current_frame]) #通过update函数得到警告信息、检测结果、以及每个时间
                
                # print('-------------------\n')
                # print('num_obj:',detections[current_frame]['num_objects'])
                # print('len of label:',len(detections[current_frame]['label']))
                # print('2d_bbox_left:',len(detections[current_frame]['2d_bbox_left']))
                # print('2d_bbox_top:',len(detections[current_frame]['2d_bbox_top']))
                # print('2d_bbox_right:',len(detections[current_frame]['2d_bbox_right']))
                # print('2d_bbox_bottom:',len(detections[current_frame]['2d_bbox_bottom']))
                # print('score:',len(detections[current_frame]['score']))
                # print('errors:',len(detections[current_frame]['errors']))
            except Exception as e:
                print('error:',e)
                # print('pre_frame',pre_frame.shape)
                # print('frame',frame.shape)
                # print('detections[current_frame]',detections[current_frame])
                # print('-------------------\n')
                # print('num_obj:',detections[current_frame]['num_objects'])
                # print('len of label:',len(detections[current_frame]['label']))
                # print('2d_bbox_left:',len(detections[current_frame]['2d_bbox_left']))
                # print('2d_bbox_top:',len(detections[current_frame]['2d_bbox_top']))
                # print('2d_bbox_right:',len(detections[current_frame]['2d_bbox_right']))
                # print('2d_bbox_bottom:',len(detections[current_frame]['2d_bbox_bottom']))
                # print('score:',len(detections[current_frame]['score']))
                # print('errors:',len(detections[current_frame]['errors']))
                # detections[current_frame] is{'num_objects': 2, 'label': array([5]), '2d_bbox_left': array([765.129111, 326.      ]), '2d_bbox_top': array([147.124585, 190.      ]), '2d_bbox_right': array([966.438106, 411.      ]), '2d_bbox_bottom': array([374., 232.]), 'score': array([0.99]), 'errors': array([[0, 0, 0],[0, 0, 0]])}
                
                
                for track in monitor._tracks:
                    print('Track:', track)
                # print('frame:',frame)
                # print('pre_frame:',pre_frame)
                cv2.imshow('frame',frame)
                cv2.imshow('pre_frame',pre_frame)
                cv2.waitKey(0)
                break
        
            # print("shape of 2dbox is:",len(detections[current_frame]['2d_bbox_left']))
            warns1, warns2, dets, oh = monitor.update(pre_frame, frame, detections[current_frame])  # 通过update函数得到警告信息、
            
            temp_warns2 = warns2
            # dets['errors'] = np.zeros((detections[current_frame]['num_objects'], 3), dtype=np.int32)
            # print('warns1', warns1, '\n', 'warns2', warns2)
            t1 = time()
            '''
            step3.Draw detection results
            '''
            for obj in range(dets['num_objects']):
                dot_colors = [(0, 0, 255), (255, 111, 131), (225, 105, 65)]
                label_len = len(labels[int(dets['label'][obj])] + ' ' + '{:.3f}'.format(
                    dets['score'][obj]))
#TODO from here
                if dets['errors'][obj].sum() == 0:
                    # pass
                    cv2.rectangle(frame, (int(dets['2d_bbox_left'][obj] + 0.5),
                                          int(dets['2d_bbox_top'][obj] + 0.5)),
                                  (int(dets['2d_bbox_right'][obj] + 0.5),
                                   int(dets['2d_bbox_bottom'][obj] + 0.5)),
                                  colors[labels[int(dets['label'][obj])]], 2)
                                  # (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(dets['2d_bbox_left'][obj] + 0.5) - 1,
                                          int(dets['2d_bbox_top'][obj] + 0.5) - 20),
                                  (int(dets['2d_bbox_left'][obj] + 0.5) - 2 + int(label_len * 12),
                                   int(dets['2d_bbox_top'][obj] + 0.5)),
                                  colors[labels[int(dets['label'][obj])]], -1)
                                  # (0, 255, 0), -1)
                    cv2.putText(frame, labels[int(dets['label'][obj])] + ' ' + '{:.3f}'.format(
                        dets['score'][obj]),
                                (int(dets['2d_bbox_left'][obj] + 0.5),
                                 int(dets['2d_bbox_top'][obj] + 0.5) - 5), font, 0.6, (0, 0, 0), 1)
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

            # print('current_frame:', current_frame)
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
        # cv2.imwrite('E:/Tracking/my_data/results/sem/{}.png'.format( current_frame), frame)
        cv2.imwrite('./sem/{}.png'.format(current_frame), frame)
        current_frame += 1
        # print('frame:',current_frame)
        c = cv2.waitKey(inteval) & 0xFF
        if c == 27 or c == ord('q'):
            break
        if current_frame == max_frame-1:
            break
        pre_frame = frame
    # duration = np.array(duration)
    # #print(np.min(duration), np.max(duration), np.mean(duration))
    # #print(np.argmax(duration))
    # overhead = np.array(overhead)
    # for i in range(7):
    #     print('{:.4f}'.format(np.min(overhead[:, i])), '{:.4f}'.format(np.max(overhead[:, i])),
    #           '{:.4f}'.format(np.mean(overhead[:, i])))
    cap.release()
    cv2.destroyAllWindows()
