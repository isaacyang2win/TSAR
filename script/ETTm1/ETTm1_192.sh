device=0


if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_ETTm1" ]; then
    mkdir ./logs/LongForecasting_ETTm1
fi

if [ ! -d "./logs/LongForecasting_ETTm1/p192" ]; then
    mkdir ./logs/LongForecasting_ETTm1/p192
fi
seq_len=96
pred_len=192
model_name=TSAR

lambda_48=0.03

python -u run.py \
    --is_training 1 \
    --transfer_data ETTm1 \
    --transfer_root_path ./data/ETT-small/ \
    --transfer_data_path ETTm1.csv \
    --data ETTm1 \
    --root_path ./data/ETT-small/ \
    --data_path ETTm1.csv \
    --dropout 0.0 \
    --learning_rate 0.0001 \
    --d_model 512 \
    --d_ff 2048 \
    --token_len 48 \
    --e_layers 6 \
    --gpu $device \
    --label_len 0 \
    --seq_len $seq_len \
    --ar_seq_len $seq_len \
    --pred_len $pred_len \
    --lambda_48 $lambda_48 \
    --train_epochs 10 \
    --ar_pred_len $pred_len >logs/LongForecasting_ETTm1/p192/$model_name'_'ETTm1'_'$seq_len'_'$pred_len.log
