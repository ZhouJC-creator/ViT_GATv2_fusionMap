import csv
import math
import os


def data_process(train_txt, valid_txt, outer_path, train_rate):
    train_txt = open(train_txt, 'w+', encoding="utf - 8")
    valid_txt = open(valid_txt, 'w+', encoding="utf - 8")
    # train_writer = csv.writer(train_csv)
    # train_writer.writerow(['image', 'label'])

    # valid_csv = open(valid_csv, 'w', newline="")
    # valid_writer = csv.writer(valid_csv)
    # valid_writer.writerow(['image', 'label'])

    folderlist = os.listdir(outer_path)

    for folder in folderlist:
        label = 1

        if (folder == 'Alternaria leaf spot'):
            label = 1
        elif (folder == 'Brown spot'):
            label = 2
        elif (folder == 'Frogeye leaf spot'):
            label = 3
        elif (folder == 'Grey spot'):
            label = 4
        elif (folder == 'Health'):
            label = 5
        elif (folder == 'Mosaic'):
            label = 6
        elif (folder == 'Powdery mildew'):
            label = 7
        elif (folder == 'Rust'):
            label = 8

        files = os.listdir(outer_path + '/' + folder)
        # 去掉.jpg即可根据数字排序而不是字符串排序
        # files.sort(key=lambda x: int(x[:-4]))
        data_len = len(files)
        train_len = math.floor(data_len * train_rate)

        for i in range(train_len):
            train_txt.write(files[i] + ', ' + str(label) + '\n')

        for j in range(train_len, data_len):
            valid_txt.write(files[j] + ', ' + str(label) + '\n')

data_process('../result/train.txt', '../result/test.txt', 'F:/DataSet/fine-grained/AppleLeaf9', 0.8)
