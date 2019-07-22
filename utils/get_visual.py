import cv2
import glob

def get_visual(capture, interval):
    cascade_file = glob.glob('**/haarcascade_frontalface_alt.xml', recursive=True)[0]
    cascade = cv2.CascadeClassifier(cascade_file)
    color = (255, 255, 0)
    pen_w = 2
    number, count = 1, 0
    while True:
        _, frame = capture.read()
        # frame = cv2.flip(frame, 1) # 画像の反転
        if not _:
            cv2.destroyAllWindows()
            capture.release()
            break
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(image_gray)
        if interval > 0 and count == 0 and face_list is not ():
            cv2.imwrite('imgs/img_{:010}.png'.format(number), frame)
            number += 1
            count = interval
        if count > 0:
            count -= 1
        for (x, y, w, h) in face_list:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness = pen_w)
        _, image = cv2.imencode('.jpg', frame)
        yield b'--boundary\r\nContent-Type: image/jpeg\r\n\r\n' + image.tostring() + b'\r\n\r\n'

def draw_detection(img, bboxes, scores, cls_inds, colors, labels, thr=0.2):
    h, w, _ = img.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(img, mess, (box[0], box[1] - 7), 0, 1e-3 * h, colors[cls_indx], thick // 3)
    return img

def get_visual_m2det(capture, interval):
    from models.m2det import build_net
    import torch
    import torch.backends.cudnn as cudnn
    import numpy as np
    from utils import Config, anchors, init_net, PriorBox, BaseTransform, Detect, to_color, nms
    config_file = glob.glob('**/m2det512_vgg.py', recursive=True)[0]
    param_file = glob.glob('**/m2det512_vgg.pth', recursive=True)[0]
    label_file = glob.glob('**/coco_labels.txt', recursive=True)[0]
    cfg = Config.fromfile(config_file)
    anchor_config = anchors(cfg)
    net = build_net('test', size=cfg.model.input_size, config=cfg.model.m2det_config)
    init_net(net, cfg, param_file, cfg.test_cfg.cuda)
    net.eval()
    with torch.no_grad():
        priors = PriorBox(anchor_config).forward()
        if cfg.test_cfg.cuda:
            net = net.cuda()
            priors = priors.cuda()
            cudnn.benchmark = True
        else:
            net = net.cpu()
    preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
    detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)
    base = int(np.ceil(pow(cfg.model.m2det_config.num_classes, 1. / 3)))
    colors = [to_color(x, base) for x in range(cfg.model.m2det_config.num_classes)]
    cats = [_.strip().split(',')[-1] for _ in open(label_file, 'r').readlines()]
    labels = tuple(['__background__'] + cats)
    number, count = 1, 0
    while True:
        _, frame = capture.read()
        # frame = cv2.flip(frame, 1) # 画像の反転
        if not _:
            cv2.destroyAllWindows()
            capture.release()
            break
        w, h = frame.shape[1], frame.shape[0]
        img = preprocess(frame).unsqueeze(0)
        scale = torch.Tensor([w, h, w, h])
        if cfg.test_cfg.cuda:
            img = img.cuda()
            scale = scale.cuda()
        out = net(img)
        boxes, scores = detector.forward(out, priors)
        boxes = (boxes[0] * scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []
        for j in range(1, cfg.model.m2det_config.num_classes):
            inds = np.where(scores[:, j] > cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(c_dets, cfg.test_cfg.iou, cfg.test_cfg.soft_nms)
            keep = keep[:cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allboxes.extend([_.tolist() + [j] for _ in c_dets])
        allboxes = np.array(allboxes)
        if len(allboxes) == 0:
            continue
        boxes = allboxes[:, :4]
        scores = allboxes[:, 4]
        cls_inds = allboxes[:, 5]
        if interval > 0 and count == 0 and 1 in cls_inds:
            cv2.imwrite('imgs/img_{:010}.png'.format(number), frame)
            number += 1
            count = interval
        if count > 0:
            count -= 1
        frame = draw_detection(frame, boxes, scores, cls_inds, colors, labels)
        _, image = cv2.imencode('.jpg', frame)
        yield b'--boundary\r\nContent-Type: image/jpeg\r\n\r\n' + image.tostring() + b'\r\n\r\n'
