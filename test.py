import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from util.metrics import SSIM
from PIL import Image
from tqdm import tqdm

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.phase = 'test'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(
    web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase,
                                                          opt.which_epoch))
# test
avgPSNR = 0.0
avgPSNR_b = 0.0
#avgSSIM = 0.0
#avgSSIM_b = 0.0
counter = 0

exposure_list = [0.0000001, 0.01, 0.1, 0.2, 0.3]
exposure_label = ['00', '01', '1', '2', '3']

for i, data in enumerate(tqdm(dataset, total=len(dataset))):
    #if i >= opt.how_many:
    #    break
    counter = i
    model.set_input(data)
    for j, exposure in enumerate(exposure_list):

        model.test(exposure)
        visuals = model.get_current_visuals()
        visualizer.display_current_results(visuals, counter)
        # PSNR_b = PSNR(visuals['syn_haze_img'], visuals['clear_img'])
        # PSNR_d = PSNR(visuals['s_dehazing_img'], visuals['clear_img'])
        # avgPSNR += PSNR_d
        # avgPSNR_b += PSNR_b
        #pilReala = Image.fromarray(visuals['real_A'])
        #pilFake = Image.fromarray(visuals['fake_B'])
        #pilReal = Image.fromarray(visuals['real_B'])
        #SSIM_b = SSIM(pilReala,pilReal)
        #SSIM_b = SSIM(pilReala).cw_ssim_value(pilReal)
        #SSIM_d = SSIM(pilFake).cw_ssim_value(pilReal)
        #avgSSIM += SSIM_d
        #avgSSIM_b += SSIM_b
        """DANET Evaluation
        img_path = model.get_image_paths()
        img = Image.fromarray(visuals['r_dehazing_img'], 'RGB')
        img.save(os.path.join(opt.results_dir,'syn'+img_path[0].split('/')[-1]))
        
        img = Image.fromarray(visuals['real_haze_img'], 'RGB')
        img.save(os.path.join(opt.results_dir,'ori_'+img_path[0].split('/')[-1]))
        """
        #CYCLEGAN Eval
        if opt.model == 'cyclegan':
            #img_path = model.get_image_paths()

            if j == 0:
                img = Image.fromarray(visuals['real_A'], 'RGB')
                img.save(os.path.join(opt.results_dir,str(counter)+'_Input'+'.png'))

                img = Image.fromarray(visuals['input_B'], 'RGB')
                img.save(os.path.join(opt.results_dir,str(counter)+'_ExpertC'+'.png'))

            img = Image.fromarray(visuals['fake_B'], 'RGB')
            img.save(os.path.join(opt.results_dir,str(counter)+'_'+str(exposure_label[j])+'.png'))

            """
            img = Image.fromarray(visuals['rec_A'], 'RGB')
            img.save(os.path.join(opt.results_dir,str(counter)+'_rec_Syn'+'.png'))
            
            img = Image.fromarray(visuals['real_B'], 'RGB')
            img.save(os.path.join(opt.results_dir,str(counter)+'_Real'+'.png'))

            img = Image.fromarray(visuals['fake_A'], 'RGB')
            img.save(os.path.join(opt.results_dir,str(counter)+'_R2S'+'.png'))
            
            img = Image.fromarray(visuals['rec_B'], 'RGB')
            img.save(os.path.join(opt.results_dir,str(counter)+'_rec_Real'+'.png'))
            """

        else:
            img_path = model.get_image_paths()
            img = Image.fromarray(visuals['r_dehazing_img'], 'RGB')
            img.save(os.path.join(opt.results_dir,'syn'+img_path[0].split('/')[-1]))
            
            img = Image.fromarray(visuals['real_haze_img'], 'RGB')
            img.save(os.path.join(opt.results_dir,'ori_'+img_path[0].split('/')[-1]))

        # print('process image... %s ... Deblurred PSNR ... %f' % (img_path, PSNR_d))
        # print('process image... %s ... Blurred PSNR ... %f ... Deblurred PSNR ... %f'
        #     % (img_path, PSNR_b, PSNR_d))
        #print('process image... %s ... Blurred SSIM ... %f ... Deblurred SSIM ... %f' % (img_path, SSIM_b, SSIM_d))
        # visualizer.save_images(webpage, visuals, img_path)

# avgPSNR /= counter
# avgPSNR_b /= counter
#avgSSIM /= counter
#avgSSIM_b /= counter
# print('Blurred PSNR = %f, Deblurred PSNR = %f' % (avgPSNR_b, avgPSNR))
#print('PSNR = %f, SSIM = %f' %
#                 (avgPSNR, avgSSIM))

# webpage.save()