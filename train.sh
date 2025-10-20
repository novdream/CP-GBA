
#!/bin/bash
PYTHON_SCRIPT_PATH="/CP-GBA/train.py"

# 模型数据集
dataset="Cora"
# 定义变量范围600/50
prompt_sizes=(5)
num_prompts=(350) #300
homo_boost_thrd=0.5
hidden=128
id=0
trojan_epochs=10 # 100 
val_feq=10 #50
evaluate_mode=1by1
selection_method=cluster_degree
defense_modes=( "none") #"none"
homo_loss_weights=(10.0) # 10.0
norm_weight=0.0001
dis_weight=0.0
layer=3 
seeds=(1) # 10 106 100 1000 10000 123 1234 12345 123456
exp=8
train_lr=5e-4 # 5e-4
# 0.005
new="Y"
position=True
index=0
train=True
num_attach=50
fit_attach_nums=(50) # 50
test_thr=0.1
prune_thr=0.1
Init=True
target_labels=(0)
attack_single=True
attack_nums=(1)
pre_train_model_path='./pre_trained_gnn/Cora.Edgepred_GPPT.GCN.128hidden_dim.pth'

# GRACE
drop_edge_rate_1s=(0.1)
drop_edge_rate_2s=(0.5) 
drop_feature_rate_1s=(0.1) 
drop_feature_rate_2s=(0.5)



# # CCA-SSG
# drop_edge_rate_1s=(0.1)
# drop_edge_rate_2s=(0.1) 
# drop_feature_rate_1s=(0.5) 
# drop_feature_rate_2s=(0.7)
tau=0.4
lr_method='GPL'
down_method='GSL'
prompt_types=('Gprompt') #'GPF' 'All-in-one'
prompt_type='Gprompt' #GPPT All-in-one Gprompt
top_k=30 # 30
gcl_hidden=256 #256-GRACE 512-CCA-SSG
clr_weights=(1.0) #10.0 20.0 30.0 40.0
class_clr_weights=(1.0) #10.0 20.0 30.0 40.0
atk_weights=1.0

dshield_pretrain_epochs=100
dshield_finetune_epochs=400 
dshield_classify_epochs=200 
dshield_neg_epochs=100 
dshield_kappa1=5 
dshield_kappa2=5 
dshield_kappa3=0.1 
dshield_edge_drop_ratio=0.20 
dshield_feature_drop_ratio=0.20 
dshield_tau=0.9 
dshield_balance_factor=0.5 
dshield_classify_rounds=1 
dshield_thresh=2.5
for (( i=1; i<=1; i++ ))  
do  

    for defense_mode in ${defense_modes[@]};do
        for homo_loss_weight in ${homo_loss_weights[@]};do
        # for seed in ${seeds[@]};do
            for fit_attach_num in ${fit_attach_nums[@]};do
            
                for prompt_size in ${prompt_sizes[@]}; do
                    
                    for num_prompt in ${num_prompts[@]}; do
                    
                                # for homo_loss_weight in ${homo_loss_weights[@]};do
                                for seed in ${seeds[@]};do
                                    for attack_num in ${attack_nums[@]};do
                                        for class_clr_weight in ${class_clr_weights[@]};do

                                            for clr_weight in ${clr_weights[@]};do

                                                for target_label in ${target_labels[@]};do
                                                    for prompt_type in ${prompt_types[@]};do
                                                        for drop_edge_rate_1 in ${drop_edge_rate_1s[@]};do
                                                                for drop_edge_rate_2 in ${drop_edge_rate_2s[@]};do
                                                                    for drop_feature_rate_1 in ${drop_feature_rate_1s[@]};do
                                                                        for drop_feature_rate_2 in ${drop_feature_rate_2s[@]};do
                                                                        echo "Running model with prompt_size=$prompt_size, num_prompts=$num_prompt"
                                                                        # 调用模型训练命令，传递变量
                                                                        # python $PYTHON_SCRIPT_PATH --dataset $dataset --prompt_size $prompt_size  --num_prompts $num_prompt --prune_thr $prune_thr --defense_mode $defense_mode
                                                                        python $PYTHON_SCRIPT_PATH --dataset $dataset --prompt_size $prompt_size  --num_prompts $num_prompt \
                                                                                                    --homo_boost_thrd $homo_boost_thrd --hidden $hidden --num_attach $fit_attach_num \
                                                                                                    --norm_weight $norm_weight --layer $layer --trojan_epochs $trojan_epochs\
                                                                                                    --device_id $id --evaluate_mode $evaluate_mode --homo_loss_weight $homo_loss_weight\
                                                                                                    --dis_weight $dis_weight --selection_method $selection_method --seed $seed --exp $exp\
                                                                                                    --prune_thr $prune_thr --defense_mode $defense_mode --index $index --fit_attach_num $fit_attach_num\
                                                                                                    --new $new --position $position --train_lr $train_lr \
                                                                                                    --test_thr $test_thr --val_feq $val_feq \
                                                                                                    --train $train\
                                                                                                    --Init $Init \
                                                                                                    --attack_num $attack_num\
                                                                                                    --drop_edge_rate_1 $drop_edge_rate_1 --drop_edge_rate_2 $drop_edge_rate_2\
                                                                                                    --drop_feature_rate_1 $drop_feature_rate_1 --drop_feature_rate_2 $drop_feature_rate_2\
                                                                                                    --tau $tau\
                                                                                                    --prompt_type $prompt_type\
                                                                                                    --class_clr_weight $class_clr_weight --clr_weight $clr_weight\
                                                                                                    --pre_train_model_path $pre_train_model_path\
                                                                                                    --lr_method $lr_method\
                                                                                                    --top_k $top_k --gcl_hidden $gcl_hidden --down_method $down_method\
                                                                                                    --attack_single $attack_single --target_label $target_label\
                                                                                                    --atk_weights $atk_weights\
                                                                                                    --dshield_pretrain_epochs=$dshield_pretrain_epochs\
                                                                                                    --dshield_finetune_epochs=$dshield_finetune_epochs\
                                                                                                    --dshield_classify_epochs=$dshield_classify_epochs\
                                                                                                    --dshield_neg_epochs=$dshield_neg_epochs\
                                                                                                    --dshield_kappa1=$dshield_kappa1\
                                                                                                    --dshield_kappa2=$dshield_kappa2\
                                                                                                    --dshield_kappa3=$dshield_kappa3\
                                                                                                    --dshield_edge_drop_ratio=$dshield_edge_drop_ratio\
                                                                                                    --dshield_feature_drop_ratio=$dshield_feature_drop_ratio\
                                                                                                    --dshield_tau=$dshield_tau\
                                                                                                    --dshield_balance_factor=$dshield_balance_factor\
                                                                                                    --dshield_classify_rounds=$dshield_classify_rounds\
                                                                                                    --dshield_thresh=$dshield_thresh
                                                                                                    # --new $new --position $position --train_lr $train_lr  --train $train\
                                                                        done
                                                                    done
                                                                done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                        
                                done
                    done
                done
                
            done

        done

    done

done
