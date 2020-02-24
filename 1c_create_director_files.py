import os
import numpy as np
imO='/workspace/8data/img/'
maO='/workspace/8data/gt_cat/'
im_ls=os.listdir(imO)
ma_ls=os.listdir(maO)

with open("img8_small.txt", "w") as f1:
    with open("gt_cat8_small.txt", "w") as f2:
        
        for im_name in im_ls:
            keys=im_name.split('.')[0].split('_')[:3]
            mask_name='_'.join(keys)+'_gtFine_color'+'.png' 
            
            if os.path.exists(maO+mask_name):
                #print(im_name,mask_name)
                f1.write('%-1s\n' % (imO+im_name))
                #f2.write('%-1s,%s\n' % (imO+im_name,maO+mask_name))
                f2.write('%-1s\n' % (maO+mask_name))  