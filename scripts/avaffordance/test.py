import torch
import torch.nn
import os
from mmengine import Config
import argparse
from utils import pyutils
from utility import mask_iou, Eval_Fmeasure, save_mask
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset


def main():
    # logger
    logger = getLogger(None, __name__)
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    logger.info(cfg.pretty_text)

    # model
    model = build_model(**cfg.model)
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    logger.info('Load trained model %s' % args.weights)

    # Test data
    test_dataset = build_dataset(**cfg.dataset.val)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.dataset.test.batch_size*8,
                                                  shuffle=False,
                                                  num_workers=cfg.process.num_works,
                                                  pin_memory=True,drop_last=True)
    avg_meter_miou = pyutils.AverageMeter('miou_func','miou_dep')
    avg_meter_F = pyutils.AverageMeter('F_score_func','F_score_dep')

    # Test
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio, func_gt, dep_gt, img_label, video_name_list = batch_data

            imgs = imgs.cuda()
            audio = audio.cuda()
            func_gt = func_gt.cuda()
            dep_gt = dep_gt.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            func_gt = func_gt.view(B * frame, H, W)
            dep_gt = dep_gt.view(B * frame, H, W)
            audio = audio.view(-1, audio.shape[2],
                               audio.shape[3], audio.shape[4])

            mask_func, mask_dep, _, _  = model(audio, imgs)
            bs = mask_func.shape[0]
            mask_func = mask_func.squeeze(1)  # [bs, 512, 512]
            mask_dep = mask_dep.squeeze(1)

            func_gt = func_gt.squeeze(1)  # [bs, 512, 512]
            dep_gt = dep_gt.squeeze(1)

            if args.save_pred_mask:
                mask_save_path = os.path.join(args.save_dir, dir_name, 'pred_masks_val')
                video_name_list_func = [f"{v}_func" for v in video_name_list]
                video_name_list_dep = [f"{v}_dep" for v in video_name_list]
                save_mask(mask_func, mask_save_path, video_name_list_func, type="func")
                save_mask(mask_dep, mask_save_path, video_name_list_dep, type="dep")

            # 逐样本计算指标
            for i in range(bs):
                one_mask_func = mask_func[i].unsqueeze(0)  # [1, H, W]
                one_mask_dep = mask_dep[i].unsqueeze(0)
                one_func_gt = func_gt[i].unsqueeze(0)
                one_dep_gt = dep_gt[i].unsqueeze(0)

                miou_func = mask_iou(one_mask_func, one_func_gt)
                F_score_func = Eval_Fmeasure(one_mask_func, one_func_gt)

                if one_dep_gt.sum() > 0:
                    miou_dep = mask_iou(one_mask_dep, one_dep_gt)
                    F_score_dep = Eval_Fmeasure(one_mask_dep, one_dep_gt)
                    avg_meter_miou.add({'miou_func': miou_func, 'miou_dep': miou_dep})
                    avg_meter_F.add({'F_score_func': F_score_func, 'F_score_dep': F_score_dep})
                else:
                    avg_meter_miou.add({'miou_func': miou_func})
                    avg_meter_F.add({'F_score_func': F_score_func})

                logger.info('n_iter: {}, sample: {}, iou_func: {:.4f}, iou_dep: {}, F_func: {:.4f}, F_dep: {}'.format(
                    n_iter, i,
                    miou_func,
                    f"{miou_dep:.4f}" if one_dep_gt.sum() > 0 else "N/A",
                    F_score_func,
                    f"{F_score_dep:.4f}" if one_dep_gt.sum() > 0 else "N/A"))
        miou_func = (avg_meter_miou.pop('miou_func'))
        miou_dep = (avg_meter_miou.pop('miou_dep'))
        F_score_func = (avg_meter_F.pop('F_score_func'))
        F_score_dep = (avg_meter_F.pop('F_score_dep'))
        logger.info(f'test miou func: {miou_func}')
        logger.info(f'test F func: {F_score_func}')
        logger.info(f'test miou dep: {miou_dep}')
        logger.info(f'test F dep: {F_score_dep}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('weights', type=str, help='model weights path')
    parser.add_argument("--save_pred_mask", action='store_true',
                        default=False, help="save predited masks or not")
    parser.add_argument('--save_dir', type=str,
                        default='work_dir', help='save path')

    args = parser.parse_args()
    main()
