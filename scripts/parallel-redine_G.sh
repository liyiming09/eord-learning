#目前最好结果：refine-G-4-effect-masknce=0.1-style500，0.1的masknce，在线去做传统eord，不用生成网络
#尝试1：去掉masknce中pos的负对部分，只约束正对相近
#尝试2：part代表只有底层的两个尺度会用到masknce
#尝试3： masknce目前是针对fake的，改为针对pos试一下
#尝试5：现在的sota是传统online做的，改成encoder做如何？
#尝试9： effect 由pos 和 neg 计算得到
#尝试10： pos的nce计算时，拉近所有对的距离

#尝试12： 使用pos-proto计算neg

#尝试14： 使用nce loss 不weight， 不select
#name="refine-G-12-effect-masknce_rangque=0.1-pos=cos-fromproto-style500-fakeatt"
#name="refine-G-10-effect-masknce_rangque=0.1-pos=nonegk-style500-fakeatt-1" 
name="refine-G-sota-diffaug"
#name="refine-G-9-poseffect-masknce_rangque=0.1-pos=cos-style500-fakeatt" 修改effect的输入图像
#sotaname="refine-G-4-effect-masknce=0.1-pos=cos-style500"
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 3 --master_port 7994 train_refine.py  --name $name --lr 0.0001 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0,1,2  --no_instance --no_skip_connections --batchSize 9 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --netD divcomultiscale --serial_batches --model refineg --niter 40 --niter_decay 100 --netE vae --effect  --masknce  --use_style_loss --lambda_masknce 0.1 --lambda_style 500 --continue_train --use_queue --netF maskboxque --eord --fakeattention  --rand_pos_que --negtype frompos --no_negk --diffaug
#CUDA_VISIBLE_DEVICES="0,4,5,6" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train_refine.py  --name $name --lr 0.0001 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 8 --mix_input_gen --nThreads 4 --bbox --segmentation_mask --netD divcomultiscale --serial_batches --model refineg --niter 0 --niter_decay 40 --netE vae  --masknce --netF maskbox --effect --negtype frombg


CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal --which_epoch 130  --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 130 --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal --which_epoch 130 --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 130 --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal --which_epoch 120  --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 120 --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal --which_epoch 120 --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 120 --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal  --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition  --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal  --vae
CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition  --vae







#CUDA_VISIBLE_DEVICES="1,0,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train_refine.py  --name refine-G-1-nofakeatt-noeffect-estep10 --lr 0.0001 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 8 --mix_input_gen --nThreads 4 --bbox --segmentation_mask --netD divcomultiscale --serial_batches --model refineg --niter 0 --niter_decay 40 --netE vae  --monce --effect_steps_per_G 10 --continue_train

#CUDA_VISIBLE_DEVICES="4,6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal
#CUDA_VISIBLE_DEVICES="4,6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition



#CUDA_VISIBLE_DEVICES="4,6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --removal
#CUDA_VISIBLE_DEVICES="4,6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --addition
python cal_ind.py
python cal_ind.py


#CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch --nproc_per_node 1 --master_port 9999 train.py  --name only-E-1 --lr 0.0002 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0  --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model learneord --effect --no_vgg_loss --cost_type hard `
