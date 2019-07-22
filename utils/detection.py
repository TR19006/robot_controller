import torch
from torch.autograd import Function

def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

class Detect(Function):
    def __init__(self, num_classes, bkg_label, cfg):
        self.num_classes = num_classes
        self.background_label = bkg_label

        self.variance = cfg['variance']

    def forward(self, predictions, prior):
        loc, conf = predictions

        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(1, self.num_priors, 4)
        self.scores = torch.zeros(1, self.num_priors, self.num_classes)
        if loc_data.is_cuda:
            self.boxes = self.boxes.cuda()
            self.scores = self.scores.cuda()

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)

        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)
            self.boxes.expand_(num, self.num_priors, 4)
            self.scores.expand_(num, self.num_priors, self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()

            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores

        return self.boxes, self.scores
