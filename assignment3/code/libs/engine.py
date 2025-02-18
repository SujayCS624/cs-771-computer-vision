import os
import json
import time

import torch
from torchvision.utils import draw_bounding_boxes, save_image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .utils import AverageMeter, convert_to_xywh, gen_distinct_colors


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    device,
    scaler=None,
    tb_writer=None,
    print_freq=10,
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, (imgs, targets) in enumerate(train_loader, 0):
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # zero out optimizer
        optimizer.zero_grad()
        if scaler != None:
            # mixed precision training
            # forward / backward the model
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                losses = model(imgs, targets)
            scaler.scale(losses["final_loss"]).backward()
            # step optimizer / scheduler
            scaler.step(optimizer)
            scheduler.step()
            # update the scaler
            scaler.update()
        else:
            # forward / backward the model
            losses = model(imgs, targets)
            losses["final_loss"].backward()
            # step optimizer / scheduler
            optimizer.step()
            scheduler.step()

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensorboard
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar("train/learning_rate", lr, global_step)
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars("train/all_losses", tag_dict, global_step)
                # final loss
                tb_writer.add_scalar(
                    "train/final_loss", losses_tracker["final_loss"].val, global_step
                )

            # print to terminal
            block1 = "Epoch: [{:03d}][{:05d}/{:05d}]".format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = "Time {:.2f} ({:.2f})".format(batch_time.val, batch_time.avg)
            block3 = "Loss {:.2f} ({:.2f})\n".format(
                losses_tracker["final_loss"].val, losses_tracker["final_loss"].avg
            )
            block4 = ""
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4 += "\t{:s} {:.2f} ({:.2f})".format(key, value.val, value.avg)

            print("\t".join([block1, block2, block3, block4]))

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return


def evaluate(
    val_loader,
    model,
    output_file,
    gt_json_file,
    device,
    print_freq=10,
    viz=False,
    viz_box_thr=0.5,
):
    """Test the model on the validation set"""
    # an output file will be used to save all results
    assert output_file is not None

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    cpu_device = torch.device("cpu")

    # prepare color maps / output folder for visualization
    if viz:
        category_color_maps = gen_distinct_colors(model.num_classes)
        category_names = val_loader.dataset.get_cls_names()
        viz_folder = os.path.join(os.path.dirname(output_file), 'viz')
        if not os.path.exists(viz_folder):
            os.mkdir(viz_folder)

    # loop over validation set
    start = time.time()
    det_results = []
    for iter_idx, data in enumerate(val_loader, 0):
        imgs, targets = data
        imgs_device = list(img.to(device) for img in imgs)
        # forward the model (wo. grad)
        with torch.no_grad():
            outputs = model(imgs_device, None)

        # unpack the results
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            boxes = convert_to_xywh(output["boxes"]).tolist()
            scores = output["scores"].tolist()
            labels = output["labels"].tolist()
            for box, score, label in zip(boxes, scores, labels):
                det_results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box,
                        "score": score,
                    }
                )

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (after sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            if viz:
                # visualize the detection results
                for img, target, output in zip(imgs, targets, outputs):
                    # select boxes with confidence scores larger than a threshold
                    box_inds = (output["scores"]>=viz_box_thr)
                    boxes = output["boxes"][box_inds]
                    # again an offset -1 is needed
                    labels = (output["labels"] - 1)[box_inds].tolist()
                    boxes_labels = [category_names[label] for label in labels]
                    boxes_colors = [category_color_maps[label] for label in labels]
                    # draw the box on the image
                    img = (img * 255).to(torch.uint8)
                    viz_img = draw_bounding_boxes(
                        img,
                        boxes,
                        boxes_labels,
                        boxes_colors,
                        width=2,
                        font="FreeMono.ttf",
                        font_size=16
                    )
                    # save the results
                    image_id = int(target["image_id"].item())
                    save_image(
                        # this convertion is needed thanks to inconsistent torchvision API
                        viz_img.to(torch.float) / 255,
                        os.path.join(viz_folder,"{:06d}.png".format(image_id))
                    )

            # print timing
            print(
                "Test: [{0:05d}/{1:05d}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})".format(
                    iter_idx, len(val_loader), batch_time=batch_time
                )
            )

    # save results to json file
    with open(output_file, "w") as outfile:
        json.dump(det_results, outfile)

    # use COCO API for evaluation
    coco_gt = COCO(gt_json_file)
    coco_dt = coco_gt.loadRes(output_file)
    cocoEval = COCOeval(coco_gt, coco_dt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return
