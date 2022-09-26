

#CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train.py  --name no-effect-bboxmonce-hard-ade --lr 0.0002 --netG spadeunet2 --dataset_mode ade20kbox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 32 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale  --serial_batches --model attentioneffect  --monce --no_vgg_loss --cost_type hard --netF Box --continue_train

#CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test.py --name no-effect-bboxmonce-hard-ade --dataset_mode ade20kbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal
#CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test.py --name no-effect-bboxmonce-hard-ade --dataset_mode ade20kbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test.py --name no-effect-bboxmonce-hard-ade --dataset_mode ade20kbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test.py --name no-effect-bboxmonce-hard-ade --dataset_mode ade20kbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition





