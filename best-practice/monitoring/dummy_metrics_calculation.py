import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()



def main():
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
		for i in range(0, 10):
			
			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	main()