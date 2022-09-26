python test.py --name erode_sea_eord_divco --dataset_mode cityscapesbox --gpu 0,1 --netG sesame --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition

CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2  test.py --name ddp_sea_eord_divco_fix --dataset_mode cityscapesbox --gpu 0,1 --netG sesame --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal

