from datetime import datetime

import pandas as pd
import sqlite3


QUERY = """
SELECT
    runs.run_uuid
    , ROW_NUMBER() OVER (ORDER BY runs.start_time) AS run_id
    , tags.value AS model
    , runs.start_time AS start_time
    , CASE WHEN metrics.key = 'val_mse' THEN metrics.value ELSE NULL END AS val_mse
    , CASE WHEN metrics.key = 'val_rmse' THEN metrics.value ELSE NULL END AS val_rmse
FROM runs

INNER JOIN tags ON runs.run_uuid = tags.run_uuid
INNER JOIN metrics ON runs.run_uuid = metrics.run_uuid

WHERE tags.key = 'model'
    AND tags.value IS NOT NULL
    AND tags.value != 'NoneType'
    AND metrics.value IS NOT NULL
	AND (val_mse IS NOT NULL
	OR val_rmse  IS NOT NULL)

ORDER BY runs.start_time ASC
"""


def load_data(*args, **kwargs) -> pd.DataFrame:
    with sqlite3.connect("database/mlflow.db") as conn:
        print("Start loading Data")
        cursor = conn.cursor()

        cursor.execute(QUERY)

        rows = cursor.fetchall()
        processed_rows = []
        for row in rows:
            run_uuid, run_id, model, start_time, val_mse, val_rmse = row
            start_time = datetime.utcfromtimestamp(start_time / 1000)
            start_time_day = start_time.day
            start_time_hour = start_time.hour
            start_time_minute = start_time.minute
            start_time_format_day = start_time.strftime('%Y-%m-%d')
            start_time_format_hour = start_time.strftime('%Y-%m-%d %H:%M')
            start_time_format_minute = start_time.strftime('%H:%MD%d')

            model = model.split('.')[-1].strip("'>")
            data = dict(
                model=model,
                mse=val_mse,
                rmse=val_rmse,
                run_id=run_id,
                run_uuid=run_uuid,
                start_time=start_time,
                start_time_day=start_time_day,
                start_time_format_day=start_time_format_day,
                start_time_format_hour=start_time_format_hour,
                start_time_format_minute=start_time_format_minute,
                start_time_hour=start_time_hour,
                start_time_minute=start_time_minute,
            )
            data[f'mse_{model}'] = val_mse
            data[f'rmse_{model}'] = val_rmse

            processed_rows.append(data)

        return pd.DataFrame(processed_rows)
