name="noeffect-diff-city-1"

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9199 train.py  --name $name --lr 0.0002 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 32 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --netD divcomultiscale  --serial_batches --model refineg   --diffaug  --use_style_loss

#spadeunet2
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test_online.py --eord --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test_online.py --eord --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal


CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test_online.py --eord --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition

CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test_online.py --eord  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition





