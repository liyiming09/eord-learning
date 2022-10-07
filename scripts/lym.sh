
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train.py  --name E-ED-0 --lr 0.0002 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 4 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model learneord  
 
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name E-ED-0 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name E-ED-0 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name E-ED-0 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name E-ED-0 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition



#CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch --nproc_per_node 1 --master_port 9999 train.py  --name E-ED-0 --lr 0.0002 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0  --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model learneord --effect --no_vgg_loss --cost_type hard 
