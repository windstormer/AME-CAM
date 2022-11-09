import torch
import numpy as np
import torch.nn.functional as F

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, use_cosine_similarity=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.use_cosine_similarity = use_cosine_similarity

    def forward(self, zis, zjs, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda')
        features = torch.cat([zis.unsqueeze(1), zjs.unsqueeze(1)], dim=1)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, feature_dim],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # print("mask.shape", mask.shape)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print("contrast_feature.shape", contrast_feature.shape)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        logits = anchor_feature, contrast_feature
        if self.use_cosine_similarity:
            cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            logits = cosine_similarity(anchor_feature.unsqueeze(1), contrast_feature.unsqueeze(0))
            logits = torch.div(logits, self.temperature)
        else:
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
        # print("logits.shape", logits.shape)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0
        # )
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * anchor_count).to(device)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        p_i = mask.sum(1)
        # p_i = torch.where(p_i==0, torch.ones(1, dtype=torch.float).cuda(), p_i)
        mean_log_prob_pos = (mask * log_prob).sum(1) / p_i

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def dice_loss(gt, gen):
    # print(gen.dtype, gt.dtype)
    gt = torch.flatten(gt)
    gen = torch.flatten(gen)
    intersection = torch.sum(gt * gen)
    dice = (2. * intersection + 1e-5) / (torch.sum(gt) + torch.sum(gen) + 1e-5)
    return dice