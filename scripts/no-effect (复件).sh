name="erode_sea_base"

#CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train.py  --name $name --lr 0.0002 --netG unet --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 32 --mix_input_gen --nThreads 4 --bbox --segmentation_mask --netD pix2pix  --serial_batches --model refineg  --no_ganFeat_loss --continue_train

#spadeunet2
#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_online.py --results_dir ./results_online/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG unet --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal
#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_online.py --results_dir ./results_online/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG unet --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal


CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_online.py --eord --results_dir ./results_online/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG sesame --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition

CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_online.py --eord --results_dir ./results_online/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG sesame --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition





