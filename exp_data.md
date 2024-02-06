nohup python -u train.py --learning_rate 0.001 --model_name senet154 --epoch 200 --batch_size 32 --step_lr 20 --dataset_name food101 >senet154_food101.log 2>&1 &


nohup python -u train.py --learning_rate 0.001 --model_name PRENet --epoch 200 --batch_size 32 --step_lr 20 --dataset_name food101 >PRENet_food101.log 2>&1 &

nohup python main.py --dataset food101 --image_path ./data/food-101/images/ --train_path ./data/food-101/meta/train.txt --test_path ./data/food-101/meta/test.txt --weight_path ./model/resnet50.pth --test --use_checkpoint --checkpoint ./pretrained_model/model.pth >PRENet_food101.log 2>&1 &

nohup python main.py --dataset food101 --image_path ./data/food-101/images/ --train_path ./data/food-101/meta/train.txt --test_path ./data/food-101/meta/test.txt --weight_path ./model/resnet50.pth --use_checkpoint False >PRENet_food101.log 2>&1 &
## KD_train
nohup python -u train_kd.py --learning_rate 0.001 --model_name senet154_pretrained --epoch 200 --batch_size 32 --step_lr 20 --dataset_name food101 >senet154_food101_KD.log 2>&1 &


## KD_train_alpha=0.5
nohup python -u train_kd.py --learning_rate 0.001 --model_name senet154_pretrained --epoch 200 --batch_size 32 --step_lr 20 --dataset_name food101 >senet154_food101_KD_alpha=0.5.log 2>&1 &


## train mobilenetV3_KD
nohup python -u train_kd.py --learning_rate 0.001 --model_name mobilenetV3 --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenetV3_food101_KD.log 2>&1 &

## train mobilenetV3_KD_T=4.0
nohup python -u train_kd.py --learning_rate 0.001 --model_name mobilenetV3 --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenetV3_food101_T=4.log 2>&1 &

## train mobilenet_pretrained_KD_T=3.0
nohup python -u train_kd.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD.log 2>&1 &

## train mobilenetV3_pretrained_KD_T=3.0
nohup python -u train_kd.py --learning_rate 0.001 --model_name mobilenetV3_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenetV3_pretrained_food101_KD.log 2>&1 &

## train mobilenet_pretrained_KD_T=3.0_intermediate_features
nohup python -u train_kd_Intermediate_features.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_intermediate_features.log 2>&1 &

## train mobilenet_KD_contrastive_T=3.0
nohup python -u train_kd_contrastive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive.log 2>&1 &

## train mobilenet_KD_contrastive_T=3.0_tri_beta=0.3
nohup python -u train_kd_contrastive_Triplet.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_Triplet_beta=0.3.log 2>&1 &

## train mobilenet_KD_contrastive_T=3.0_tri_beta=0.4
nohup python -u train_kd_contrastive_Triplet.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_Triplet_beta=0.4.log 2>&1 &

## train mobilenet_KD_contrastive_T=3.0_tri_CA_alpha=0.5_beta=0.3_dynamic
nohup python -u train_kd_contrastive_Triplet_CA_dynamic.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_Triplet_CA_alpha=0.5_beta=0.3_dynamic.log 2>&1 &s

## train mobilenet_KD_contrastive_T=3.0_tri_CA_alpha=0.5_beta=0.5_dynamic
nohup python -u train_kd_contrastive_Triplet_CA_dynamic.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_Triplet_CA_alpha=0.5_beta=0.5_dynamic.log 2>&1 &s

## train mobilenet_KD_contrastive_CA_T=3.0_tri_beta=0.5_alpha=0.5_improved_dynamic loss
nohup python -u train_kd_contrastive_Triplet_CA_improved_dynamic.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_Triplet_beta=0.3_CA_loss_improved_dynamic.log 2>&1 &

## train mobilenet_KD_contrastive_CA_T=3.0_tri_beta=0.3—————>0.8251
nohup python -u train_kd_contrastive_Triplet_CA.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_Triplet_beta=0.3_CA.log 2>&1 &

## train mobilenet_KD_contrastive_CA_T=3.0_tri_beta=0.5_alpha=0—————>0.8251
nohup python -u train_kd_contrastive_Triplet_CA.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_Triplet_beta=0.5_alpha=0.5_CA_loss.log 2>&1 &

## train mobilenet_KD_contrastive_CA_T=3.0_tri_beta=0.5_alpha=0—————>0.8251
nohup python -u train_kd_contrastive_Triplet_CA.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food500 >mobilenet_pretrained_food500_KD_contrastive_Triplet_beta=0.5_alpha=0.5_CA_loss.log 2>&1 &

## train mobilenet_KD_contrastive_CA_T=3.0_tri_beta=0.5_alpha=0.5_improved loss
nohup python -u train_kd_contrastive_Triplet_CA_improved.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_Triplet_beta=0.5_CA_loss_improved.log 2>&1 &

## train mobilenet_KD_contrastive_CA_T=3.0_tri_beta=0.5_pretrained
nohup python -u train_kd_contrastive_Triplet_CA.py --learning_rate 0.001 --model_name mbv2_ca_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mbv2_ca_pretrained_food101_KD_contrastive_Triplet_beta=0.5_CA_pretrained.log 2>&1 &

## train mobilenet_KD_contrastive_T=3.0 --changed infoNCEloss and criterionloss
nohup python -u train_kd_contrastive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_new.log 2>&1 &

## train mobilenet_KD_contrastive_T=3.0 --changed infoNCEloss and criterionloss
nohup python -u train_kd_contrastive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_new.log 2>&1 &

## train mobilenet_KD_contrastive_T=3.0_tri(beta=0.5)--->0.808
nohup python -u train_kd_contrastive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_tri.log 2>&1 &

## train mobilenet_KD_contrastive_T=3.0_tri_beta=0.3--->0.8128
nohup python -u train_kd_contrastive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 400 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_tri_beta=0.3.log 2>&1 &   

## train mobilenet_KD_contrastive_T=3.0_tri_beta=0.1--->0.8089
nohup python -u train_kd_contrastive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 400 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_tri_beta=0.1.log 2>&1 & 

## train mobilenet_KD_contrastive_T=3.0_tri_beta=0.4--->0.8109
nohup python -u train_kd_contrastive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 400 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_tri_beta=0.4.log 2>&1 & 

## train mobilenet_KD_after_contrastive_T=3.0_tri
nohup python -u train_kd_after_contrastive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_after_contrastive_tri.log 2>&1 &

## train mobilenet_KD_contrastive_T=4.0_tri_beta=0.3
nohup python -u train_kd_contrastive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 400 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_KD_contrastive_tri_beta=0.3_T=4.log 2>&1 &  

## train mobilnet_CA
nohup python -u train_CA.py --learning_rate 0.001 --model_name mobilnet --epoch 200 --batch_size 32 --step_lr 20 --dataset_name food101 >mobilnet_CA_food101.log 2>&1 &

## train mobilnet_CA food 500
nohup python -u train_CA.py --learning_rate 0.001 --model_name mobilnet --epoch 200 --batch_size 32 --step_lr 20 --dataset_name food500 >mobilnet_CA_food500.log 2>&1 &

## train mobilenet_CA_KA_T=3.0
nohup python -u train_kd_CA.py --learning_rate 0.001 --model_name mobilenet --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_food101_KD_CA.log 2>&1 &

## train mobilenet_CA_KD_contrastive_T=3.0
nohup python -u train_kd_contrastive_CA.py --learning_rate 0.001 --model_name mobilenet --epoch 400 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_food101_KD_contrastive_CA.log 2>&1 & 

## train mobilenet_KD_contrastive_T=3.0_mult_tri
nohup python -u train_kd_contrastive_mult.py --learning_rate 0.001 --model_name mobilenet --epoch 400 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_food101_KD_contrastive_mult_tri.log 2>&1 & 

## train mobilenet_KD_adaptive--->0.8092
nohup python -u train_kd_adaptive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_kd_adaptive.log 2>&1 & 

## train mobilenet_KD_adaptive--->0.811
nohup python -u train_kd_adaptive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_kd_adaptive_test.log 2>&1 & 

## train mobilenet_KD_adaptive_batch_size--->0.811
nohup python -u train_kd_adaptive.py --learning_rate 0.001 --model_name mobilenet_pretrained --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mobilenet_pretrained_food101_kd_adaptive_test_batch_size.log 2>&1 & 


## train mbv2_ca
nohup python -u train_mbv2.py --learning_rate 0.001 --model_name mbv2_CA --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food101 >mbv2_ca.log 2>&1 &

## train mbv2_ca_food500
nohup python -u train_mbv2.py --learning_rate 0.001 --model_name mbv2_CA --epoch 200 --batch_size 32 --step_lr 30 --dataset_name food500 >mbv2_ca_food500.log 2>&1 &


# train_mobilenetV2_food101
nohup python -u train.py --learning_rate 0.001 --model_name mobilenet --epoch 200 --batch_size 32 --step_lr 20 --dataset_name food101 >mobilenet_food101.log 2>&1 &