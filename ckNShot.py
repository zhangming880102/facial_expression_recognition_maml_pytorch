from    ckmaml import CK
import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np


CHANNEL=3

class CKNShot:
    def labelOrder(self,data):
        temp=dict()
        for (img,label) in data:
            if label in temp.keys():
                temp[label].append(img)
            else:
                temp[label]=[img]

        new_data=[]
        for label,imgs in temp.items():
            new_data.append(np.array(imgs))

        return len(temp),new_data

    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz
        # if root/data.npy does not exist, just download it
        self.x = CK(root,transform=transforms.Compose([lambda x: x.resize((imgsz, imgsz)),
                                                            lambda x: np.array(x),
                                                            lambda x: np.transpose(x, [2, 0, 1]),
                                                            lambda x: x/255]))
        temp=[]
        for img,label in self.x:
            temp.append([img,label])

        train_number=len(temp)
        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        self.train_clss,self.x_train=self.labelOrder(temp[:train_number])
        self.test_clss,self.x_test =self.labelOrder(temp[train_number:])

        # self.normalization()

        self.batchsz = batchsz
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", len(self.x_train),'class number',self.train_clss, "test", len(self.x_test),'class number',self.test_clss)

        self.datasets_cache = {"train": (self.train_clss,self.load_data_cache(self.datasets["train"],self.train_clss)),  # current epoch data cached
                               "test": (self.train_clss,self.load_data_cache(self.datasets["train"],self.train_clss))}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack,class_number):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(class_number, self.n_way, False)

                for j, cur_class in enumerate(selected_cls):
                    selected_c=data_pack[cur_class]
                    selected_img = np.random.choice(selected_c.shape[0], self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(selected_c[selected_img[:self.k_shot]])
                    x_qry.append(selected_c[selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, CHANNEL, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, CHANNEL, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, CHANNEL, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int64).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, CHANNEL, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int64).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        model_clss,mode_data=self.datasets_cache[mode]
        if self.indexes[mode] >= len(mode_data):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = (model_clss,self.load_data_cache(self.datasets[mode],model_clss))
        
        model_clss,mode_data=self.datasets_cache[mode]
        next_batch = mode_data[self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch





if __name__ == '__main__':

    import  time
    import  torch
    import  visdom

    # plt.ion()

    ck_db = CKNShot('db/ck', batchsz=20, n_way=5, k_shot=5, k_query=15, imgsz=64)
    for i in range(1000):
        x_spt, y_spt, x_qry, y_qry = ck_db.next('train')


        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        x_spt = torch.from_numpy(x_spt)
        x_qry = torch.from_numpy(x_qry)
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)
        batchsz, setsz, c, h, w = x_spt.size()





