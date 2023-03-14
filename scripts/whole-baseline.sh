
name="whole-city-pix"
netg='unetw'
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 7990 train_whole_baseline.py  --name $name --lr 0.0002 --netG $netg --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 32 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --netD sesamemultiscale --serial_batches --model whole --niter 40 --niter_decay 60  --use_style_loss --lambda_style 500 --no_inpaint --no_ganFeat_loss

#name="whole-city-pix2pix"
#CUDA_VISIBLE_DEVICES="4, 7,6,5" python -m torch.distributed.launch --nproc_per_node 4 --master_port 7990 train_whole.py  --name $name --lr 0.0002 --netG pixw --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 32 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --netD divcomultiscale --serial_batches --model whole --niter 40 --niter_decay 60  --use_style_loss --lambda_style 500 --no_inpaint

#name="whole-city-hong"
#CUDA_VISIBLE_DEVICES="4, 7,6,5" python -m torch.distributed.launch --nproc_per_node 4 --master_port 7990 train_whole.py  --name $name --lr 0.0002 --netG hongw --dataset_mode Ecitybox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 32 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --netD divcomultiscale --serial_batches --model whole --niter 40 --niter_decay 60  --use_style_loss --lambda_style 500 --no_inpaint

#name="sesame-ade-$netg"
#CUDA_VISIBLE_DEVICES="4, 7,6,5" python -m torch.distributed.launch --nproc_per_node 4 --master_port 7990 train_whole.py  --name $name --lr 0.0002 --netG $netg --dataset_mode Eadebox --gpu 0,1,2,3  --no_instance --no_skip_connections --batchSize 32 --mix_input_gen --nThreads 8 --bbox --segmentation_mask --netD sesamemultiscale --serial_batches --model whole --niter 40 --niter_decay 60  --use_style_loss --lambda_style 500 --no_inpaint  --dataroot ../../datasets/ade20k/



CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9009 test_whole_baseline.py  --results_dir ./results_whole/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --no_inpaint --bbox --segmentation_mask --netE vae --eord --removal --which_epoch 90  
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9009 test_whole_baseline.py  --results_dir ./results_whole/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --no_inpaint --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 90 
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9009 test_whole_baseline.py  --results_dir ./results_whole/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --no_inpaint --bbox --segmentation_mask --netE vae --eord --removal --which_epoch 90 
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9009 test_whole_baseline.py  --results_dir ./results_whole/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --no_inpaint --bbox --segmentation_mask --netE vae --eord --addition --which_epoch 90
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9009 test_whole_baseline.py  --results_dir ./results_whole/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --no_inpaint --bbox --segmentation_mask --netE vae --eord --removal  
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9009 test_whole_baseline.py  --results_dir ./results_whole/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --no_inpaint --bbox --segmentation_mask --netE vae --eord --addition  
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9009 test_whole_baseline.py  --results_dir ./results_whole/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --no_inpaint --bbox --segmentation_mask --netE vae --eord --removal  
CUDA_VISIBLE_DEVICES="0,1,2" python -m torch.distributed.launch --nproc_per_node 2  --master_port 9009 test_whole_baseline.py  --results_dir ./results_whole/  --name $name --dataset_mode Ecitybox --gpu 0,1 --netG $netg --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --no_inpaint --bbox --segmentation_mask --netE vae --eord --addition  

python cal_ind_whole.py
python cal_ind_whole.py


#CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch --nproc_per_node 1 --master_port 9999 train.py  --name only-E-1 --lr 0.0002 --netG $netg --dataset_mode Ecitybox --gpu 0  --no_instance --no_skip_connections --batchSize 2 --mix_input_gen --nThreads 0 --no_inpaint --bbox --segmentation_mask --netD divcomultiscale --eord --serial_batches --model learneord --effect --no_vgg_loss --cost_type hard `
