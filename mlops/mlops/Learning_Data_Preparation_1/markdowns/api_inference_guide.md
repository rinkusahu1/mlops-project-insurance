curl --location 'http://127.0.0.1:6789/api/pipeline_schedules/15/api_trigger' \
--header 'Authorization: Bearer 469405433a364bd988ad2bf9deef4b9f' \
--header 'Content-Type: application/json' \
--header 'Cookie: lng=en' \
--data '{
    "run": {
        "pipeline_uuid": "predict",
        "block_uuid": "inference",
        "variables": {
            "inputs": [
                {
                    "DOLocationID": "239",
                    "PULocationID": "236",
                    "trip_distance": 1.98
                },
                {
                    "DOLocationID": "170",
                    "PULocationID": "65",
                    "trip_distance": 6.54
                }
            ]
        }
    }
}'

# Note
The Authorization header is using this pipeline’s API trigger’s token value. The token value is set to fire for this project. If you create a new trigger, that token will change. Only use a fixed token for testing or demonstration purposes.