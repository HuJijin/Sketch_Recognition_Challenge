import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import time
import torch
import torchvision
import torchvision.transforms as transforms
import logging
import shutil
from log import Log
from Dataset import ImageSet
import opts_evaluation as opts
from processing import *


# 128 x 3 x 256 x256 -> 1280 x 3 x 256 x 256
def do_multiview_crop(data):
    crop_img_size = 224
    orig_img_size = 256

    n = data.shape[0]   # n = batch_size
    xs = [0, 0, orig_img_size - crop_img_size, orig_img_size - crop_img_size]  # 0, 0, 32, 32    256-224=32
    ys = [0, orig_img_size - crop_img_size, 0, orig_img_size - crop_img_size]  # 0, 32, 0, 32

    new_data = torch.zeros(n * 10, 3, crop_img_size, crop_img_size)   # 每张图取四个角加中心一共五张图，再进行翻转，得到十张图
    y_cen = int((orig_img_size - crop_img_size) * 0.5)  # 16
    x_cen = int((orig_img_size - crop_img_size) * 0.5)  # 16

    for i in range(n):  # 128
        for (k, (x, y)) in enumerate(zip(xs, ys)):
            new_data[i * 10 + k, :, :, :] = data[i, :, y:y + crop_img_size, x:x + crop_img_size]  # 4个角
        new_data[i * 10 + 4, :, :, :] = data[i, :, y_cen:y_cen + crop_img_size, x_cen:x_cen + crop_img_size]  # 中心

        for k in range(5):  # 翻转
            new_data[i * 10 + k + 5, :, :, :] = flip(new_data[i * 10 + k, :, :, :], -1)

    return new_data


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, device=x.device)
    return x[tuple(indices)]


def backup(opt):
    global log
    current_time = time.strftime("%Y-%m-%d-%H-%M")

    if opt['debug']:
        opt['evaluation_path'] = "debug_log"
    else:
        opt['evaluation_path'] = "_".join([opt['evaluation_path'], current_time])
    log_file_name = os.path.join(opt['evaluation_path'], current_time + '.log')
    os.makedirs(opt['evaluation_path'], exist_ok=True)
    log = Log(loggername=__name__, loglevel=logging.DEBUG, file_path=log_file_name).getlog()
    log.info(opt)
    ##copy file
    shutil.copyfile(__file__, os.path.join(opt['evaluation_path'], os.path.basename(__file__)))


def best_model_func(model, best_resnet):
    state_dict = model.state_dict()

    for key in state_dict.keys():
        if key in best_resnet.keys():
            state_dict[key] = best_resnet[key]
        else:
            log.info('missing best key: %s' % key)

    model.load_state_dict(state_dict)
    return model


def pred(model, test_loader, opt):
    model.eval()
    if opt['test_type'] == "single":
        # center crop
        with torch.no_grad():
            pred_id = None
            img_names = ()
            for datas, names in test_loader:  # 经过loader后，数据输出格式为batsize*channel*w*h,即128x3x224x224
                img_names += names
                centercrop_pred = model(datas.cuda())
                centercrop_label = torch.argmax(centercrop_pred, dim=1).cpu()
                if pred_id is None:
                    pred_id = centercrop_label
                else:
                    pred_id = torch.cat((pred_id, centercrop_label))
            pred_id = pred_id.numpy()
            img_names = list(img_names)
            pred_labels = classid2label(pred_id)
    else:
        with torch.no_grad():
            # multi crop
            pred_id = None
            img_names = ()
            for datas, names in test_loader:
                img_names += names
                b, c, w, h = datas.shape   # 128 x 3 x 256 x 256
                multicrop_datas = do_multiview_crop(datas)  # 1280 x 3 x 256 x 256
                multicrop_labels = model(multicrop_datas.cuda()).squeeze()
                avg_label = torch.argmax(multicrop_labels.view(b, 10, -1).mean(dim=1), dim=1).cpu()
                if pred_id is None:
                    pred_id = avg_label
                else:
                    pred_id = torch.cat((pred_id, avg_label))
            pred_id = pred_id.numpy()
            img_names = list(img_names)
            pred_labels = classid2label(pred_id)

    return img_names, pred_labels


def classid2label(pred_id):
    pred_labels = []
    id2labels = dict([(0,'airplane'),(1,'alarm_clock'),(2,'ambulance'),(3,'ant'),(4,'apple'),
                        (5,'backpack'),(6,'basket'),(7,'butterfly'),(8,'cactus'),(9,'calculator'),
                        (10,'campfire'),(11,'candle'),(12,'coffee_cup'),(13,'crab'),(14,'duck'),
                        (15,'face'),(16,'ice_cream'),(17,'pig'),(18,'pineapple'),(19,'suitcase')])
    for i in range(len(pred_id)):
        for key,value in id2labels.items():
            if pred_id[i] == key:
                pred_labels.append(id2labels[key])
    return pred_labels


def evaluate(excelpath, opt, seen=1):
    # --------------------------------------------------------------------
    # 将test_unseen_label.txt的数据存为字典
    # unseen数据集在目前阶段用不到
    # rate1 = open('test_unseen_label.txt', 'r', encoding='utf-8')
    # test_unseen_label = dict()
    # for line in rate1:
    #     line = line.strip().split(' ')
    #     test_unseen_label[line[0]] = line[1]
    # rate1.close()

    # 将test_seen_label.txt的数据存为字典
    rate2 = open(opt['file_path'], 'r', encoding='utf-8')
    test_seen_label = dict()
    for line in rate2:
        line = line.strip().split(' ')
        test_seen_label[line[0]] = line[1]
    rate2.close()
    # --------------------------------------------------------------------
    file_path = excelpath
    wb = load_workbook(file_path)
    sheet_list = wb.sheetnames
    ws = wb[sheet_list[0]]
    total_list = []
    for row in ws.rows:
        row_list = []
        row_list.append(str(row[0].value))
        row_list.append(str(row[1].value))
        total_list.append(row_list)
    pred = dict(total_list)

    # --------------------------------------------------------------------
    y_true = []
    y_pred = []
    x = pred
    # seen =1
    if seen == 0:
        y = test_unseen_label
    else:
        y = test_seen_label
    z = y.keys()-y.keys()^x.keys()
    for key in z:
        for i, j in y.items():
            if i == key:
                y_pred.append(x[key])
                y_true.append(j)
    str1 = '------------------测试集上得分：---------------------—-'
    str2 = '------------------- 混淆矩阵: ----------------------—-'
    str3 = '矩阵类别序列同上'
    str4 = str1+'\n'+classification_report(y_true, y_pred)+'\n'+str2+'\n'+str3+'\n'
    log.info(str1+'\n'+classification_report(y_true, y_pred))
    # print(str4)
    output_file = os.path.join(opt['evaluation_path'], 'evaluation.txt')
    write_text(output_file, str4)

    with open(output_file, 'a') as f:
        np.savetxt(f, np.column_stack(confusion_matrix(y_true, y_pred).T), fmt='%10.f')
    # confusion_matrix(y_true, y_pred)
    # print(confusion_matrix(y_true, y_pred))


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpus']
    backup(opt)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),   # single时需要center crop, multi时不需要
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ImageSet(opt['testset_path'], opt['file_path'], transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=8
    )

    if opt['backbone'] == 'resnet50' or opt['backbone'] == 'Resnet50':
        model = torchvision.models.resnet50(num_classes=opt['num_classes']).cuda()

        if os.path.exists(opt['best']):
            best_model = torch.load(opt['best'])
            for key in list(best_model.keys()):
                if 'module.' in key:
                    best_model[key.replace('module.', '')] = best_model.pop(key)
            if 'network' in best_model.keys():
                best_model = best_model['network']
            elif 'swav' in opt['best'] or 'deepcluster' in opt['best']:
                model = torch.nn.DataParallel(model)
            model = best_model_func(model, best_model)
        else:
            log.info("could not find best file %s"%(opt['best']))
    else:
        print('Could not find backbone %s ' % (opt['backbone']))
        exit()
    # model = torch.load(opt['best'])
    model = torch.nn.DataParallel(model)
    img_names, pred_labels = pred(model, test_loader, opt)
    pred_file = write2excel(img_names, pred_labels, opt)
    evaluate(pred_file, opt)


if __name__ == "__main__":
    opt = opts.parse_opt()
    opt = vars(opt)  # vars()返回字典
    main(opt)


