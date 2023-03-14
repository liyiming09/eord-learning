
name="refine-G-ade-7-effect-masknce_randque=0.1-pos=nonegk-style500-fakeatt-median"
#dataset="Ecitybox"
#root="../../datasets/cityscape/"

dataset="Eadebox"
root="../../datasets/ade20k/"

#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 130 --vae


#CUDA_VISIBLE_DEVICES="1,2,3" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9901 test_refine.py --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 120 --vae

CUDA_VISIBLE_DEVICES="0, 1,2,3" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9904 test_refine.py --name $name --dataset_mode $dataset --dataroot $root --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition  --vae


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



#CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch --nproc_per_node 1 --master_port 9999 train.py  --name only-E-1 --lr 0.0002 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0  --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model learneord --effect --no_vgg_loss --cost_type hard `
