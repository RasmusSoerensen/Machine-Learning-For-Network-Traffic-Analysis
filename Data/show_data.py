import os
import pandas as pd
import gc  # For garbage collection

folder = os.path.join("Data", "BigDataset", "IoTScenarios")
file = os.path.join("bro", "conn.log.labeled")

list_2_csv_part_1 = ["CTU-Honeypot-Capture-4-1", "CTU-Honeypot-Capture-5-1", "CTU-Honeypot-Capture-7-1","CTU-IoT-Malware-Capture-1-1", "CTU-IoT-Malware-Capture-3-1", "CTU-IoT-Malware-Capture-7-1", 
                     "CTU-IoT-Malware-Capture-8-1", "CTU-IoT-Malware-Capture-9-1", "CTU-IoT-Malware-Capture-17-1"]
list_2_csv_part_2 = ["CTU-IoT-Malware-Capture-20-1", "CTU-IoT-Malware-Capture-21-1", "CTU-IoT-Malware-Capture-33-1", 
                     "CTU-IoT-Malware-Capture-34-1"]
list_2_csv_part_3 = ["CTU-IoT-Malware-Capture-39-1"]
list_2_csv_part_4 = ["CTU-IoT-Malware-Capture-35-1", "CTU-IoT-Malware-Capture-36-1", "CTU-IoT-Malware-Capture-42-1", 
                     "CTU-IoT-Malware-Capture-44-1", "CTU-IoT-Malware-Capture-48-1", "CTU-IoT-Malware-Capture-49-1", 
                     "CTU-IoT-Malware-Capture-52-1", "CTU-IoT-Malware-Capture-60-1"]
list_2_csv_part_5 = ["CTU-IoT-Malware-Capture-43-1"]

MAX_ROWS = 1000000  # Maximum number of rows to process per dataset

def process_file_in_chunks(data_path, output_file):
    print(f"Processing {data_path} in chunks")
    chunk_size = 15000000  # Adjust based on available memory
    total_rows_processed = 0  # Track how many rows have been processed

    for chunk in pd.read_csv(data_path, sep='\t', comment="#", header=None, chunksize=chunk_size):
        remaining_rows = MAX_ROWS - total_rows_processed
        if remaining_rows <= 0:
            break

        if len(chunk) > remaining_rows:
            chunk = chunk.iloc[:remaining_rows]

        # Append the chunk to output CSV
        chunk.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        total_rows_processed += len(chunk)

        del chunk  # Free memory after processing the chunk
        gc.collect()  # Garbage collection after each chunk

def process_and_save(file_list, output_file):
    with pd.option_context('mode.chained_assignment', None):
        for folder_name in file_list:
            data_path = os.path.join(folder, folder_name, file)
            process_file_in_chunks(data_path, output_file)

# Process and save each part
process_and_save(list_2_csv_part_1, "combined_data_2.csv")
print("Combined data 2 has been saved")

process_and_save(list_2_csv_part_2, "combined_data_3.csv")
print("Combined data 3 has been saved")

process_and_save(list_2_csv_part_3, "combined_data_4.csv")
print("Combined data 4 has been saved")

process_and_save(list_2_csv_part_4, "combined_data_5.csv")
print("Combined data 5 has been saved")

process_and_save(list_2_csv_part_5, "combined_data_6.csv")
print("Combined data 6 has been saved")

