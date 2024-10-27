export KINESIS_STREAM_INPUT="stg_insurance_events_r-mlops-practice"
export KINESIS_STREAM_OUTPUT="stg_insurance_predictions_r-mlops-practice"

SHARD_ID=$(aws kinesis put-record  \
        --stream-name ${KINESIS_STREAM_INPUT}   \
        --partition-key 1  --cli-binary-format raw-in-base64-out  \
        --data '{
            "insurance": {
                "smoker": "yes",
                "sex": "female",
                "children": 0,
                "bmi": 26.29,
                "age": 62
            },
            "medical_insurance_id": 256
        }'  \
        --query 'ShardId'
    )

#SHARD_ITERATOR=$(aws kinesis get-shard-iterator --shard-id ${SHARD_ID} --shard-iterator-type TRIM_HORIZON --stream-name ${KINESIS_STREAM_OUTPUT} --query 'ShardIterator')

#aws kinesis get-records --shard-iterator $SHARD_ITERATOR
