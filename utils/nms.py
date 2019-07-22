import numpy as np

def cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def cpu_soft_nms(dets, thresh=0.001, method=0):
    Nt = 0.3
    sigma = 0.5
    N = dets.shape[0]
    for i in range(N):
        maxscore = dets[i, 4]
        maxpos = i

        tx1 = dets[i,0]
        ty1 = dets[i,1]
        tx2 = dets[i,2]
        ty2 = dets[i,3]
        ts = dets[i,4]

        pos = i + 1
        while pos < N:
            if maxscore < dets[pos, 4]:
                maxscore = dets[pos, 4]
                maxpos = pos
            pos = pos + 1

        dets[i,0] = dets[maxpos,0]
        dets[i,1] = dets[maxpos,1]
        dets[i,2] = dets[maxpos,2]
        dets[i,3] = dets[maxpos,3]
        dets[i,4] = dets[maxpos,4]

        dets[maxpos,0] = tx1
        dets[maxpos,1] = ty1
        dets[maxpos,2] = tx2
        dets[maxpos,3] = ty2
        dets[maxpos,4] = ts

        tx1 = dets[i,0]
        ty1 = dets[i,1]
        tx2 = dets[i,2]
        ty2 = dets[i,3]
        ts = dets[i,4]

        pos = i + 1
        while pos < N:
            x1 = dets[pos, 0]
            y1 = dets[pos, 1]
            x2 = dets[pos, 2]
            y2 = dets[pos, 3]
            s = dets[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    dets[pos, 4] = weight*dets[pos, 4]

                    if dets[pos, 4] < thresh:
                        dets[pos,0] = dets[N-1, 0]
                        dets[pos,1] = dets[N-1, 1]
                        dets[pos,2] = dets[N-1, 2]
                        dets[pos,3] = dets[N-1, 3]
                        dets[pos,4] = dets[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep


def nms(dets, thresh, soft_nms=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    elif soft_nms:
        return cpu_soft_nms(dets, thresh, method = 1)
    else:
        return cpu_nms(dets, thresh)