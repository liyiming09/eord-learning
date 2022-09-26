

CUDA_VISIBLE_DEVICES="2,3,4,5" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train.py  --name no-effect-base-ade-monce --lr 0.0002 --netG spadeunet2 --dataset_mode ade20kbox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 32 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale  --serial_batches --model attentioneffect   --cost_type hard --netF Box --dataroot ../../datasets/ade20k/ --monce --no_vgg_loss

CUDA_VISIBLE_DEVICES="5,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name no-effect-base-ade-monce --dataset_mode ade20kbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal --dataroot ../../datasets/ade20k/
CUDA_VISIBLE_DEVICES="5,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name no-effect-base-ade-monce --dataset_mode ade20kbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition --dataroot ../../datasets/ade20k/
CUDA_VISIBLE_DEVICES="5,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name no-effect-base-ade-monce --dataset_mode ade20kbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal --dataroot ../../datasets/ade20k/
CUDA_VISIBLE_DEVICES="5,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name no-effect-base-ade-monce --dataset_mode ade20kbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition --dataroot ../../datasets/ade20k/
python cal_ind.py
python cal_ind.py



