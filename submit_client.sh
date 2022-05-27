spark-submit \
    --master yarn \
    --deploy-mode client \
    segmentation.py \
    --output $1 \
    --dataset $2