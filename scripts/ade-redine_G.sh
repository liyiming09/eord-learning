

#尝试4：在确定ade baseline之后，进行finetune，是用小的腐蚀 结果：小腐蚀有一个点的提升
#尝试4-1：在确定ade baseline之后，进行finetune，是用大的腐蚀 结果：

#尝试： --vae  --partnce两部分的消融实验
#name="refine-G-ade-1-effect-partmasknce=0.1-pos=cos-style500-vae"
#name="refine-G-ade-4-effect-masknce_randque=0.1-pos=cos-style500"
#name="refine-G-ade-6-effect-masknce_randque=0.1-pos=cos-style500-fakeatt-noOT-1"  --no_ot

#8： monce的finetune
#name="refine-G-ade-8-effect-monce-style500-fakeatt"
#CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9914 train_refine.py  --name $name --lr 0.0001 --netG spadeunet2 --dataset_mode Eadebox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 8 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netD divcomultiscale --serial_batches --model refineg --niter 40 --niter_decay 100 --netE vae --effect --monce --use_style_loss --lambda_style 500 --continue_train --eord --netF boxque --fakeattention 


name="refine-G-ade-8-effect-masknce_randque=0.1-pos=nonegk-style500-fakeatt-median-noselect"
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9914 train_refine.py  --name $name --lr 0.0001 --netG spadeunet2 --dataset_mode Eadebox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 8 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netD divcomultiscale --serial_batches --model refineg --niter 40 --niter_decay 100 --netE vae --effect --masknce --negtype frompos --use_style_loss --lambda_masknce 0.1 --lambda_style 500 --continue_train --eord  --use_queue --netF maskboxque --fakeattention  --rand_pos_que  --no_negk --no_select
#--use_queue  --rand_pos_que --negtype frombg


CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal --which_epoch 120
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition --which_epoch 120
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal --which_epoch 120
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition --which_epoch 120
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal --which_epoch 130
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition --which_epoch 130
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal --which_epoch 130
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition --which_epoch 130

CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal 
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal 
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition 

#name="refine-G-4-effect-masknce0.1-style"
#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal --which_epoch 30
#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition --which_epoch 30
#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal --which_epoch 30
#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition --which_epoch 30
#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal 
#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition 
#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal 
#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition 









#CUDA_VISIBLE_DEVICES="1,0,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train_refine.py  --name refine-G-1-nofakeatt-noeffect-estep10 --lr 0.0001 --netG spadeunet2 --dataset_mode Eadebox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 8 --mix_input_gen --nThreads 4 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netD divcomultiscale --serial_batches --model refineg --niter 0 --niter_decay 40 --netE vae --vae   --monce --effect_steps_per_G 10 --continue_train

#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --removal
#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netE vae --vae  --eord --addition



#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --eord --removal
#CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  test.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Eadebox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --eord --addition
python cal_ind.py
python cal_ind.py


#CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch --nproc_per_node 1 --master_port 9999 train.py  --name only-E-1 --lr 0.0002 --netG spadeunet2 --dataset_mode Eadebox --gpu 0  --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --dataroot ../../datasets/ade20k/ --netD divcomultiscale --eord --serial_batches --model learneord --effect --no_vgg_loss --cost_type hard `
