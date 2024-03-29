# two cyclegan :syn2real, real2syn
import torch
import itertools
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import losses

class CycleGanmodel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_C', type=float, default=0.1,
                                help='weight for perceptual loss')
            parser.add_argument('--lambda_D', type=float, default=0.5,
                                help='weight for exposure loss')
            parser.add_argument('--lambda_identity', type=float, default=0.1, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--which_model_netG_A', type=str, default='resnet_9blocks_exposure',
                                help='selects model to use for netG_A')
        parser.add_argument('--which_model_netG_B', type=str, default='resnet_9blocks',
                                help='selects model to use for netG_B')
        parser.add_argument('--use_exposure', action='store_true',
                                help='use_exposure')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        #visual_names_A = ['real_A', 'depth', 'fake_B', 'rec_A']
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'input_B']
        #visual_names_B = ['real_B', 'real_B_depth', 'fake_A', 'rec_B']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        use_parallel = False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG_A, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual, self.use_exposure)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG_B, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual)
        
        if self.isTrain:
            use_sigmoid = False
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)

        # load/define networks
        if self.isTrain and opt.continue_train:
            self.load_networks(opt.which_epoch)

        elif not self.isTrain:
            self.init_with_pretrained_model('G_A', self.opt.g_s2r_premodel)
            self.init_with_pretrained_model('G_B', self.opt.g_r2s_premodel)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = losses.GANLoss(use_ls=True).to(self.device)
            self.discLoss, self.contentLoss, self.loss_L1, self.loss_ssim, self.L_exp = losses.init_loss(opt, self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'C'].to(self.device)
        #self.depth =  input['D'].to(self.device)
        self.real_B = input['C' if AtoB else 'A'].to(self.device)
        self.input_B = input['B'].to(self.device)
        #self.real_B_depth = input['E'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'C_paths']

    def forward(self, e):
        #self.fake_B = self.netG_A(self.real_A, self.depth, True)
        self.fake_B = self.netG_A(self.real_A, e)

        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A, e)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self, e):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
        lambda_D = self.opt.lambda_D
        # Identity loss
        if lambda_idt >= 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B, e)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fcriterionIdted.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = torch.Tensor([0])
            self.loss_idt_B = torch.Tensor([0])

        # GAN loss D_A(G_A(A))
        if self.use_exposure:
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) + lambda_D * torch.mean(self.L_exp(self.fake_B, e)) + lambda_C * self.contentLoss.get_loss(self.fake_B, self.real_A)
        else:
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) + lambda_C * self.contentLoss.get_loss(self.fake_B, self.real_A)
            
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) + lambda_C * self.contentLoss.get_loss(self.fake_A, self.real_B)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = 2*self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        if self.use_exposure:
            e = -1.0 + 1.6*np.random.rand()
        else:
            e = 0
        self.forward(e)
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G(e)
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
