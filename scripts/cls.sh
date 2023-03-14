

#CUDA_VISIBLE_DEVICES="4" python  train_cls.py   --batchSize 64  --bbox --segmentation_mask --netC resnet --gpu 0 --model cls --serial_batches --no_instance --name visual-C-2 --dataset_mode Ecitybox --nThreads 8 --niter 30 --niter_decay 30 --continue_train


CUDA_VISIBLE_DEVICES="7" python  test_cls_all.py  --batchSize 1 --nThreads 0 --method RAP --bbox --segmentation_mask --netC resnet --gpu 0 --model cls --serial_batches --no_instance --name visual-C-2 --dataset_mode Ccitybox --addition  --results_dir ./results_set --phase train

#CUDA_VISIBLE_DEVICES="4" python  test_cls.py  --batchSize 1 --nThreads 0 --method LRP --bbox --segmentation_mask --netC resnet --gpu 0 --model cls --serial_batches --no_instance --name visual-C-2 --dataset_mode Ecitybox --addition --how_many 100

#CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train_cls.py   --batchSize 64 --nThreads 0 --bbox --segmentation_mask --netC resnet --gpu 0,1,2,3 --model cls --serial_batches --no_instance --name visual-C-1 --dataset_mode Ecitybox --nThreads 4 --niter 40 --niter_decay 61 --continue_train
#--fakeattention --effect


#CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name refine-G-3-fakeatt-effect-nopretrain --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --removal
#CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --name refine-G-3-fakeatt-effect-nopretrain --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netE vae --eord --addition

#CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --removal
#CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test.py --name refine-G-1-nofakeatt-noeffect-estep10 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --addition



#CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch --nproc_per_node 1 --master_port 9999 train.py  --name only-E-1 --lr 0.0002 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0  --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model learneord --effect --no_vgg_loss --cost_type hard `
