from __future__ import print_function
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
from .utils import AverageMeter, BCELossWeighted, image_to_gpu, label_to_gpu, exchange_temp_channel_axes
import config


def train(model, data_loader, optimizer, epoch, args, criterion=torch.nn.CrossEntropyLoss().cuda()):
    loss_meter = AverageMeter()
    model.train()
    epoch_begin = time.time()
    begin_data_time = time.time()
    total_data_time = 0
    begin_loop_time = time.time()
    total_loop_time = 0
    total_gpu_time = 0
    for idx, data in enumerate(data_loader):
        data_time = time.time() - begin_data_time
        begin_gpu_time = time.time()

        (face_img, depression_labels) = data
        
        # build Variables, and transfer to GPU
        if not args.cpu:
            face_img = image_to_gpu(face_img)
            depression_labels = label_to_gpu(depression_labels)
        
        
        face_img = Variable(face_img)
        depression_labels = Variable(depression_labels)
        

        # foward predictions
        depression_pred = model.forward(face_img)
        
        # calculate loss
        loss = criterion(depression_pred, depression_labels)
#         if np.isnan(loss.data[0]):
        if np.isnan(loss.item()):
            print("Warning! This loss is NaN")
            continue
        loss_meter.update(loss.item())

        # backward gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        gpu_time = time.time() - begin_gpu_time
        loop_time = time.time() - begin_loop_time
        begin_loop_time = time.time()

        total_loop_time += loop_time
        total_data_time += data_time
        total_gpu_time += gpu_time
        if idx % args.print_freq == 0:
            if idx % 100 == 0:
                print("sum of pred is: ", np.sum(depression_pred.data.cpu().numpy()), np.sum(depression_labels.data.cpu().numpy()))
            print("Training epoch {} [{:4}/{:4}]: {:.4f}s/batch, {:.4f}s on gpu, {:.4f}s on data    "
                  "loss: {:.8f}    loss_avg:{:.8f}".format(epoch, idx, len(data_loader),
                                                           total_loop_time/args.print_freq,
                                                           total_gpu_time/args.print_freq,
                                                           total_data_time/args.print_freq,
                                                           loss_meter.value(), loss_meter.average()))
            total_loop_time = 0
            total_data_time = 0
            total_gpu_time = 0
        begin_data_time = time.time()
    epoch_time = time.time() - epoch_begin
    # print epoch summary
    print("Epoch {} training summary: {:.4f}s in total, loss_avg: {:.8f}\n".format(epoch, epoch_time,
                                                                                   loss_meter.average()))
    return loss_meter.average()


def validate(model, data_loader, epoch, args, criterion=torch.nn.BCELoss().cuda()):
    loss_meter = AverageMeter()
    model.eval()
    epoch_begin = time.time()
    begin_data_time = time.time()
    total_data_time = 0
    begin_loop_time = time.time()
    total_loop_time = 0
    total_gpu_time = 0
    
    # Find percentage
    correct = 0
    total = 0
    
    for idx, data in enumerate(data_loader):
        data_time = time.time() - begin_data_time
        begin_gpu_time = time.time()

        (face_img, depression_labels) = data
        
        # build Variables, and transfer to GPU
        if not args.cpu:
            face_img = image_to_gpu(face_img)
            depression_labels = label_to_gpu(depression_labels)
        
        
        face_img = Variable(face_img, volatile=True)
        depression_labels = Variable(depression_labels, volatile=True)

        # foward predictions
        depression_pred = model.forward(face_img)

        # calculate loss
        loss = criterion(depression_pred, depression_labels)
        
        if np.isnan(loss.item()):
            print("Warning! This loss is NaN")
            continue
        loss_meter.update(loss.item())

        total += depression_labels.size(0)
        _, predicted = torch.max(depression_pred.data, 1)
        correct += (predicted == depression_labels).sum().item()
        
        # measure elapsed time
        gpu_time = time.time() - begin_gpu_time
        loop_time = time.time() - begin_loop_time
        begin_loop_time = time.time()

        total_loop_time += loop_time
        total_data_time += data_time
        total_gpu_time += gpu_time
        if idx % args.print_freq == 0:
            print("Validation epoch {} [{:4}/{:4}]: {:.4f}s/batch, {:.4f}s on gpu, {:.4f}s on data    "
                  "loss: {:.8f}    loss_avg:{:.8f}    accuracy {}".format(epoch, idx, len(data_loader),
                                                           total_loop_time/args.print_freq,
                                                           total_gpu_time/args.print_freq,
                                                           total_data_time/args.print_freq,
                                                           loss_meter.value(), loss_meter.average(),
                                                           (100.0 * correct / total)))
            total_loop_time = 0
            total_data_time = 0
            total_gpu_time = 0
        begin_data_time = time.time()
    epoch_time = time.time() - epoch_begin
    # print epoch summary
    print("Epoch {} validation summary: {:.4f}s in total, loss_avg: {:.8f}\n".format(epoch, epoch_time,
                                                                                     loss_meter.average()))
    return loss_meter.average(), (100.0 * correct / total)


def test(model, data_loader, epoch, args, criterion = torch.nn.BCELoss().cuda()):
    loss_meter = AverageMeter()
    model.eval()
    epoch_begin = time.time()
    for idx, data in enumerate(data_loader):
        start_time = time.time()

        (face_img, depression_labels) = data
        
        # build Variables, and transfer to GPU
        if not args.cpu:
            face_img = image_to_gpu(face_img)
            depression_labels = label_to_gpu(depression_labels)
        
        
        face_img = Variable(face_img, volatile=True)
        depression_labels = Variable(depression_labels, volatile=True)

        # foward predictions
        depression_pred = model.forward(face_img)

        # calculate loss
        loss = criterion(depression_pred, depression_labels)
        
        if np.isnan(loss.item()):
            print("Warning! This loss is NaN")
            continue
        loss_meter.update(loss.item())

        # measure elapsed time
        batch_time = time.time() - start_time

        # save prediction
        save_data = exchange_temp_channel_axes(depression_pred.data).cpu().numpy()
        for i in range(len(index_list)):
            raw_prediction_folder = config.PREDICTION_RAW_FOLDER.format(predict_path=args.output, exp=exp_folders[i])
            if not os.path.exists(raw_prediction_folder):
                os.makedirs(raw_prediction_folder)
            if index_list[i] == 0:
                for count in range(0, (config.LEN_SAMPLE + config.NONOVERLAP) // 2):
                    f = os.path.join(raw_prediction_folder, str(count).zfill(config.N_DIGITS))
                    np.save(f, save_data[i][count])
            else:
                for count in range((config.LEN_SAMPLE - config.NONOVERLAP) // 2,
                                   (config.LEN_SAMPLE + config.NONOVERLAP) // 2):
                    f = os.path.join(raw_prediction_folder,
                                     str(index_list[i] * config.NONOVERLAP + count).zfill(config.N_DIGITS))
                    np.save(f, save_data[i][count])

        if idx % args.print_freq == 0:
            print("Validation epoch {} [{:4}/{:4}]: {:.4f}s/batch    "
                  "loss: {:.6f}    loss_avg:{:.6f}".format(epoch, idx, len(data_loader), batch_time,
                                                           loss_meter.value(), loss_meter.average()))
    epoch_time = time.time() - epoch_begin
    # print epoch summary
    print("Epoch {} validation summary: {:.4f}s in total, loss_avg: {:.6f}\n".format(epoch, epoch_time,
                                                                                     loss_meter.average()))
    return loss_meter.average()
