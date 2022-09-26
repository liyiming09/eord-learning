
CUDA_VISIBLE_DEVICES="3,4,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train.py  --name effect-spade-unet-10-1-ade --lr 0.0002 --netG spadeunet2 --dataset_mode cityscapesbox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 4 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model attentioneffect --effect --monce --no_vgg_loss --cost_type hard --fakeattention 
 
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name effect-spade-unet-10-1-ade --dataset_mode cityscapesbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name effect-spade-unet-10-1-ade --dataset_mode cityscapesbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name effect-spade-unet-10-1-ade --dataset_mode cityscapesbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name effect-spade-unet-10-1-ade --dataset_mode cityscapesbox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition




