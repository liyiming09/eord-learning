#主要带解决问题：总是陷入训练崩溃
#尝试3： masknce目前是针对fake的，改为针对pos试一下  结果：fake的好一点，但是好的不明显，基本相近，这说明我们的模型对微小干扰的鲁棒性很好  
#尝试1：去掉masknce中pos的负对部分，只约束正对相近[修改pos的nceloss为cos相似度]  结果：fid明显提升，其余基本不变
#尝试3：去掉sota里的vgg，是否会变好？  结果：变差了
#尝试2：part代表只有底层的两个尺度会用到masknce  结果：变差了

#尝试5：现在的sota是传统online做的，改成encoder做如何？  结果： 变好了一点， 需要再跟sota对比一下

#尝试4：在确定ade baseline之后，进行finetune，是用小的腐蚀 结果：小腐蚀有一个点的提升
#尝试4-1：在确定ade baseline之后，进行finetune，是用大的腐蚀 结果：

#尝试6： 加入queue，比较结果 最好
#尝试7： 加入queue，比较rand and pos queue的区别
#尝试8： 加入queue，比较monce的OT部分 QS的挑选部分
#尝试9： effect 由pos 和 neg 计算得到
#尝试10： pos的nce计算时，拉近所有对的距离
#尝试11： 去掉所有gran的nce loss
#尝试12： 使用pos-proto计算neg
#尝试13： sota 修复了norm的问题
name="refine-G-13-effect-masknce_posque=0.1-pos=nonegk-style500-fakeatt"
#sotaname="refine-G-4-effect-masknce=0.1-pos=cos-style500"   refine-G-13-effect-masknce_rangque=0.1-pos=cos-style500-fakeatt-1   --rand_pos_que
CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 3 --master_port 7991 train_refine.py  --name $name --lr 0.0001 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0,1,2  --no_instance --no_skip_connections --batchSize 9 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --netD divcomultiscale --serial_batches --model refineg --niter 40 --niter_decay 100 --netE vae --effect  --masknce --negtype frompos --use_style_loss --lambda_masknce 0.1 --lambda_style 500 --continue_train --use_queue --netF maskboxque --eord --fakeattention --no_negk



CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal --which_epoch 130 --vae
CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 130 --vae

CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal --which_epoch 120 --vae
CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 120 --vae

CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal  --vae
CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition  --vae


#name="refine-G-4-effect-masknce0.1-style"
#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal --which_epoch 30
#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 30
#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal --which_epoch 30
#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 30
#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal 
#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition 
#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal 
#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition 









#CUDA_VISIBLE_DEVICES="1,0,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train_refine.py  --name refine-G-1-nofakeatt-noeffect-estep10 --lr 0.0001 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 8 --mix_input_gen --nThreads 4 --bbox --segmentation_mask --netD divcomultiscale --serial_batches --model refineg --niter 0 --niter_decay 40 --netE vae  --monce --effect_steps_per_G 10 --continue_train

#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal
#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition



#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --removal
#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  test.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --addition
python cal_ind.py
python cal_ind.py


#CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch --nproc_per_node 1 --master_port 9999 train.py  --name only-E-1 --lr 0.0002 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0  --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model learneord --effect --no_vgg_loss --cost_type hard `
