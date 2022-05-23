spark-submit \
    --master yarn \
    --deploy-mode cluster \
    segmentation.py \
    --output $1
    --dataset "test"
