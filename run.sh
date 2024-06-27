python single_gpu_pretrain.py --job_dir ./job_dir --batch_size 1 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05
