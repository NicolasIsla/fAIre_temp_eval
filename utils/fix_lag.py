import os
from datetime import datetime, timedelta

main_path = 'data'

DELTA_LAG = -9

offset = timedelta(hours=DELTA_LAG)

date_time_str_len = 19  # 'YYYY_MM_DDTHH_MM_SS'

for root, dirs, files in os.walk(main_path):
    print(f'Processing folder: {root}')
    print(f'Number of files: {len(files)}')
    print('-'*50)
    for filename in files:
        if filename.endswith('.jpg'):
            print(f'Processing: {filename}')
            date_time_str = filename[-(date_time_str_len+4):-4]
            try:
                original_time = datetime.strptime(date_time_str, '%Y_%m_%dT%H_%M_%S')
                
                corrected_time = original_time + offset
                
                new_date_time_str = corrected_time.strftime('%Y_%m_%dT%H_%M_%S')
                new_filename = filename.replace(date_time_str, new_date_time_str)
                
                os.rename(os.path.join(root, filename), os.path.join(root, new_filename))
                print(f'Renamed: {filename} -> {new_filename}')
            except ValueError as ve:
                print(f"Error processing {filename}: {ve}")

print('Processing complete.')
