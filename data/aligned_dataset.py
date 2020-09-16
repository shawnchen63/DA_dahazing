import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import util.util as util


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_A = os.path.join(opt.dataroot, 'train_a')
        self.dir_B = os.path.join(opt.dataroot, 'train_b')
        self.dir_C = os.path.join(opt.dataroot, 'unlabeled')
        # self.dir_C = os.path.join(opt.dataroot, 'verify_haze_img')
        # self.dir_C = os.path.join(opt.dataroot, 'HazeRD_dataset', 'hazy')
        # self.dir_C = os.path.join(opt.dataroot, 'Dense_Haze_NTIRE19', 'hazy')
        #self.dir_D = os.path.join(opt.dataroot, 'train_depth')
        #self.dir_E = os.path.join(opt.dataroot, 'unlabeled_depth')
        #self.dir_F = os.path.join(opt.dataroot, 'test_depth')
        #self.dir_AB = os.path.join(opt.dataroot, 'train')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))
        #self.D_paths = sorted(make_dataset(self.dir_D))
        #self.E_paths = sorted(make_dataset(self.dir_E))
        #self.F_paths = sorted(make_dataset(self.dir_F))

        self.transformPIL = transforms.ToPILImage()
        transform_list1 = [transforms.ToTensor()]
        transform_list2 = [transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        #transform_list2 = [transforms.Normalize((0.5,), (0.5,))]

        self.transform1 = transforms.Compose(transform_list1)
        self.transform2 = transforms.Compose(transform_list2)

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        if self.opt.phase == 'train':
            A_path = self.A_paths[index]
            B_path = self.B_paths[index]
            #D_path = self.D_paths[index]
            # and C is the unlabel hazy image
            C_ind = random.randint(0, int((len(self.A_paths)-1)/6))
            C_path = self.C_paths[C_ind]
            #E_path = self.E_paths[C_ind]
            # C_path = self.C_paths[random.randint(0, len(self.AB_paths)-2200)]
            A = Image.open(A_path).convert('RGB')
            B = Image.open(B_path).convert('RGB')
            C = Image.open(C_path).convert('RGB')

            ori_w = A.width
            ori_h = A.height
            A = A.resize((ori_w, ori_h), Image.BICUBIC)
            B = B.resize((ori_w, ori_h), Image.BICUBIC)
            #D = D.resize((D.width, D.height), Image.BICUBIC)

            C_w = C.width
            C_h = C.height
            ## resize the real image without label
            C = C.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
            #E = E.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
            #A = A.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
            #B = B.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
            #A_half_full = A_half_full.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)

            #ori_w = AB.width
            #ori_h = AB.height

            A = self.transform1(A)
            B = self.transform1(B)
            C = self.transform1(C)
            #D = self.transform1(D)
            #E = self.transform1(E)

            ######### crop the training image into fineSize ########
            w = A.size(2)
            h = A.size(1)
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

            A = A[:, h_offset:h_offset + self.opt.fineSize,
                   w_offset:w_offset + self.opt.fineSize]
            B = B[:, h_offset:h_offset + self.opt.fineSize,
                   w_offset:w_offset + self.opt.fineSize]
            #D = D[:, h_offset:h_offset + self.opt.fineSize,
            #    w_offset:w_offset + self.opt.fineSize]

            w = C.size(2)
            h = C.size(1)
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

            #C = C[:, h_offset:h_offset + self.opt.fineSize,
            #       w_offset:w_offset + self.opt.fineSize]

            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(2, idx)
                B = B.index_select(2, idx)
                #D = D.index_select(2, idx)
                #C = C.index_select(2, idx)


            A = self.transform2(A)
            B = self.transform2(B)
            C = self.transform2(C)
            #D = self.transform2(D)
            #E = self.transform2(E)

            if random.random()<0.5:
                noise = torch.randn(3, self.opt.fineSize, self.opt.fineSize) / 100
                #A = A + noise

            return {'A': A, 'B': B, 'C': C, 'C_paths': C_path,
                    'A_paths': A_path, 'B_paths': B_path}

        elif self.opt.phase == 'test':
            #if self.opt.test_type == 'syn':
            A_path = self.A_paths[index]
            B_path = self.B_paths[index]
            #F = Image.open(F_path)
            A = Image.open(A_path).convert('RGB')
            B = Image.open(B_path).convert('RGB')
            ori_w = A.width
            ori_h = B.height
            #ori_fw = F.width
            #ori_fh = F.height


            # new_w = int(np.floor(ori_w/32)*32)
            # new_h = int(np.floor(ori_h/16)*16)
            # new_w = ori_w
            # new_h = ori_h
            new_w = 512
            new_h = 512
            A = A.resize((int(new_w), int(new_h)), Image.BICUBIC)
            B = B.resize((int(new_w), int(new_h)), Image.BICUBIC)
            A = self.transform1(A)
            B = self.transform1(B)
            A = self.transform2(A)
            B = self.transform2(B)
            #F = F.resize((ori_fw, ori_fh), Image.BICUBIC)
            #F = self.transform1(F)
            #F = self.transform2(F)
            #A = AB[:,:,0:int(new_w/2)]
            #B = AB[:,:,int(new_w/2):new_w]
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}


            elif self.opt.test_type == 'real':
                C_path = self.C_paths[index]
                C = Image.open(C_path).convert('RGB')
                C_w = C.width
                C_h = C.height
                # C = C.resize((C_w, C_h), Image.BICUBIC)

                new_w = int(np.floor(C_w / 16) * 16)
                new_h = int(np.floor(C_h / 16) * 16)
                C = C.resize((int(new_w), int(new_h)), Image.BICUBIC)
                C = self.transform1(C)
                C = self.transform2(C)
                return {'C': C, 'C_paths': C_path}
        #A = self.transformPIL(A)
        #B = self.transformPIL(B)
        #A_half = self.transformPIL(A_half)
        #A.save('A.png')
        #B.save('B.png')
        #A_half.save('A_half.png')



    def __len__(self):

        if self.opt.phase == 'train':
            return len(self.A_paths)
        elif self.opt.phase == 'test':
            if self.opt.test_type == 'syn':
                return len(self.A_paths)
            elif self.opt.test_type == 'real':
                return len(self.C_paths)

    def name(self):
        return 'AlignedDataset'
