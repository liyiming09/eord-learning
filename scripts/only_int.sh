


CUDA_VISIBLE_DEVICES="5,4,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train_inter.py  --name only-E-5-vae-D --lr 0.002 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 32 --mix_input_gen --nThreads 4 --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model learneord --niter 30 --niter_decay 60 --netE vae

#CUDA_VISIBLE_DEVICES="5,4,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 train_inter.py  --name only-E-5-vae-D --lr 0.0002 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 32 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model learneord --niter 30 --niter_decay 60 --netE vae

CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_inter.py --name only-E-5-vae-D --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --how_many 100 --results_dir ./results_int/ --addition --netE vae
CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test_inter.py --name only-E-5-vae-D --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --how_many 100 --results_dir ./results_int/ --addition --netE vae
#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name only-E-1 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --addition
#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name only-E-1 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --removal
#CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2  test.py --name only-E-1 --dataset_mode Ecitybox --gpu 0,1 --netG spadeunet2 --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --eord --addition
#python cal_ind.py
#python cal_ind.py


#CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch --nproc_per_node 1 --master_port 9999 train.py  --name only-E-1 --lr 0.0002 --netG spadeunet2 --dataset_mode Ecitybox --gpu 0  --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model learneord --effect --no_vgg_loss --cost_type hard `
