from utils.pyutils import AverageMeter


class LossUtil:
    def __init__(self, weight_dict, **kwargs) -> None:
        self.loss_weight_dict = weight_dict
        self.avg_loss = dict()
        self.avg_loss['total_loss'] = AverageMeter('total_loss')
        # for k in weight_dict.keys():
        #     self.avg_loss[k] = AverageMeter(k)

    def add_loss(self, loss, loss_dict):
        self.avg_loss['total_loss'].add({'total_loss': loss.item()})
        for k, v in loss_dict.items():
            meter = self.avg_loss.get(k, None)
            if meter is None:
                meter = AverageMeter(k)
                self.avg_loss[k] = meter

            self.avg_loss[k].add({k: v})

    def pretty_out(self):
        d = {}
        total_loss = self.avg_loss['total_loss'].pop('total_loss')
        d["Total_Loss"] = total_loss
        f = 'Total_Loss:%.4f, ' % total_loss
        for k in self.avg_loss.keys():
            if k == 'total_loss':
                continue
            tmp = self.avg_loss[k].pop(k)
            d[k] = tmp
            t = '%s:%.4f, ' % (k, tmp)
            f += t
        return f,d
