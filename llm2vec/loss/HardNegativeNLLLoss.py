import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather


class HardNegativeNLLLoss:
    def __init__(
            self,
            scale: float = 30.0,
            similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
            self,
            q_reps: Tensor,
            d_reps_pos: Tensor,
            d_reps_neg: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        d_reps_pos_last8_loss1 = full_d_reps_pos[:, -1, :]
        d_reps_neg_last8_loss1 = full_d_reps_neg[:, -1, :]

        d_reps_last8_loss1 = torch.cat([d_reps_pos_last8_loss1, d_reps_neg_last8_loss1], dim=0)
        scores_loss1 = self.similarity_fct(full_q_reps, d_reps_last8_loss1)
        scores_loss1 = scores_loss1 * self.scale
        d_reps_pos_last8 = full_d_reps_pos
        d_reps_neg_last8 = full_d_reps_neg

        d_reps_last8 = torch.cat([d_reps_pos_last8, d_reps_neg_last8], dim=0)
        q_b = full_q_reps.size(0)
        d_b = d_reps_last8.size(0)
        d_views = d_reps_last8.size(1)
        h_dim = d_reps_last8.size(2)

        d_reps_last8 = d_reps_last8.reshape(d_b * d_views, h_dim)
        sim_scores = self.similarity_fct(full_q_reps, d_reps_last8)
        max_scores, _ = torch.max(sim_scores.view(q_b, d_b, d_views), dim=2)
        scores = max_scores * self.scale

        labels_scores = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )

        loss1 = self.cross_entropy_loss(scores, labels_scores)

        teacher_targets = torch.softmax(scores.detach(), dim=-1)
        loss2 = - torch.mean(
            torch.sum(torch.log_softmax(scores_loss1, dim=-1) * teacher_targets, dim=-1))

        loss = (loss1 + loss2) / 2

        return loss
