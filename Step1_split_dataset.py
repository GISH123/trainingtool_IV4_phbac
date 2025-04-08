import os
import random
import shutil
from configuration import TRAIN_SET_RATIO, TEST_SET_RATIO
import cv2



class SplitDataset():
    def __init__(self, dataset_dir, saved_dataset_dir, train_ratio=TRAIN_SET_RATIO, test_ratio=TEST_SET_RATIO, show_progress=False):
        self.dataset_dir = dataset_dir
        self.saved_dataset_dir = saved_dataset_dir
        self.saved_train_dir = saved_dataset_dir + "/train/"
        self.saved_valid_dir = saved_dataset_dir + "/valid/"
        self.saved_test_dir = saved_dataset_dir + "/test/"


        self.train_ratio = train_ratio
        self.test_radio = test_ratio
        self.valid_ratio = 1 - train_ratio - test_ratio

        self.train_file_path = []
        self.valid_file_path = []
        self.test_file_path = []

        self.index_label_dict = {}

        self.show_progress = show_progress

        if not os.path.exists(self.saved_train_dir):
            os.mkdir(self.saved_train_dir)
        if not os.path.exists(self.saved_test_dir):
            os.mkdir(self.saved_test_dir)
        if not os.path.exists(self.saved_valid_dir):
            os.mkdir(self.saved_valid_dir)


    def __get_label_names(self):
        label_names = []
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path):
                label_names.append(item)
        return label_names

    def __get_all_file_path(self):
        all_file_path = []
        index = 0
        for file_type in self.__get_label_names():
            self.index_label_dict[index] = file_type
            index += 1
            type_file_path = os.path.join(self.dataset_dir, file_type)
            file_path = []
            for file in os.listdir(type_file_path):
                single_file_path = os.path.join(type_file_path, file)
                file_path.append(single_file_path)
            all_file_path.append(file_path)
        return all_file_path

    def __copy_files(self, type_path, type_saved_dir):
        for item in type_path:
            src_path_list = item[1]
            dst_path = type_saved_dir + "%s/" % (item[0])
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            for src_path in src_path_list:

                if self.show_progress:
                    print("Copying file "+src_path+" to "+dst_path)
                #----------------------
                img_per=cv2.imread(src_path)
                h,w ,dim= img_per.shape
                if h<300 or w <300 : 
                    if h <300: h=299
                    if w <300: w=299
                    filename= os.path.split(src_path)[1]
                    img299=cv2.resize(img_per,[h,w])
                    cv2.imwrite(src_path,img299)
                    shutil.copy(src_path, dst_path)
                else:
                    isCrop=True
                    if  isCrop:
                        x_start=150
                        x_end=x_start+400
                        if   w >400 :
                            img_per_crop=img_per[ 0:  ,x_start:x_end]
                            #cv2.imshow('img_per_crop',img_per_crop)
                            #cv2.waitKey(0)
                            cv2.imwrite(src_path,img_per_crop)
                    
                    shutil.copy(src_path, dst_path)
                #-----------------

    def __split_dataset(self):
        all_file_paths = self.__get_all_file_path()
        for index in range(len(all_file_paths)):
            file_path_list = all_file_paths[index]
            file_path_list_length = len(file_path_list)
            random.shuffle(file_path_list)

            train_num = int(file_path_list_length * self.train_ratio)
            test_num = int(file_path_list_length * self.test_radio)

            self.train_file_path.append([self.index_label_dict[index], file_path_list[: train_num]])
            self.test_file_path.append([self.index_label_dict[index], file_path_list[train_num:train_num + test_num]])
            self.valid_file_path.append([self.index_label_dict[index], file_path_list[train_num + test_num:]])

    def start_splitting(self):
        self.__split_dataset()
        self.__copy_files(type_path=self.train_file_path, type_saved_dir=self.saved_train_dir)
        self.__copy_files(type_path=self.valid_file_path, type_saved_dir=self.saved_valid_dir)
        self.__copy_files(type_path=self.test_file_path, type_saved_dir=self.saved_test_dir)


if __name__ == '__main__':

    dataset_dir="original_dataset"
    saved_dataset_dir="dataset"
    if os.path.exists(saved_dataset_dir) :  shutil.rmtree(saved_dataset_dir)
    
    os.mkdir(saved_dataset_dir)
    split_dataset = SplitDataset(dataset_dir=dataset_dir,
                                 saved_dataset_dir=saved_dataset_dir,
                                 show_progress=True)
    
    
    split_dataset.start_splitting()


    import to_tfrecord
    from configuration import train_dir, valid_dir, test_dir, train_tfrecord, valid_tfrecord, test_tfrecord
    from prepare_data import get_images_and_labels
    import random
    to_tfrecord.dataset_to_tfrecord(dataset_dir=train_dir, tfrecord_name=train_tfrecord)
    to_tfrecord.dataset_to_tfrecord(dataset_dir=valid_dir, tfrecord_name=valid_tfrecord)
    to_tfrecord.dataset_to_tfrecord(dataset_dir=test_dir, tfrecord_name=test_tfrecord)