name="base-onlyeord-ade-big"
netg="spadeunet2"
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train_refine.py  --name $name --lr 0.0002 --netG $netg --dataset_mode Eadebox --dataroot ../../datasets/ade20k/ --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 16 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --netD divcomultiscale  --serial_batches --model refineg --eord

CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Eadebox --dataroot ../../datasets/ade20k/ --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition --which_epoch 90
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Eadebox --dataroot ../../datasets/ade20k/ --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal --which_epoch 90
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Eadebox --dataroot ../../datasets/ade20k/ --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Eadebox --dataroot ../../datasets/ade20k/ --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal

CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Eadebox --dataroot ../../datasets/ade20k/ --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition --which_epoch 90
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Eadebox --dataroot ../../datasets/ade20k/ --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal --which_epoch 90
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Eadebox --dataroot ../../datasets/ade20k/ --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Eadebox --dataroot ../../datasets/ade20k/ --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal

#CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train_refine.py  --name $name --lr 0.0002 --netG $netg --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 16 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --netD divcomultiscale  --serial_batches --model refineg --eord --continue_train

#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition

#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --addition

#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal

#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_refine.py --eord  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal


#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_online.py --results_dir ./results_online/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal
#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_online.py --results_dir ./results_online/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --removal





python cal_ind.py
python cal_ind.py
