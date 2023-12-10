import os
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
import csv
from multiprocessing import Pool, cpu_count
from pdf_processing import headers, pdf_to_text_with_structure

###################

from itertools import product
from PIL import Image

# Path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your path

# Example usage
pdf_folder_path = "C:/Users/ENeue/OCR Tryout/Batch1" # Update with your path

# Specify your desired file path
output_file_path = "C:/Users/ENeue/OCR Tryout/Batch1"

# Iterate through all PDFs
def process_folder(pdf_folder_path, headers, output_file_path):
    # Check if the output CSV file already exists
    file_exists = os.path.exists(output_file_path)
    
    # Get all PDF files in the folder
    pdf_files = [file for file in os.listdir(pdf_folder_path) if file.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        try:
            df, _ = pdf_to_text_with_structure(pdf_path, headers)
            # Write to CSV
            if file_exists:
                df.to_csv(output_file_path, mode='a', header=False, index=False, sep = "|")
            else:
                df.to_csv(output_file_path, index=False, sep = "|")
                file_exists = True
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

# Call the function
process_folder(pdf_folder_path, headers, output_file_path)

# Iterate through all PDFs, Multiprocessing
def process_folder(pdf_folder_path, headers, output_file_path):
    # Check if the output CSV file already exists
    file_exists = os.path.exists(output_file_path)

    # Get all PDF files in the folder
    pdf_files = [os.path.join(pdf_folder_path, file) for file in os.listdir(pdf_folder_path) if file.endswith('.pdf')]

    # Create a list of tuples, where each tuple is (pdf_file, headers)
    args_list = [(pdf_file, headers) for pdf_file in pdf_files]

    # Use a process pool to parallelize the work, using one core less than available
    num_processes = max(1, cpu_count() - 6)
    with Pool(num_processes) as pool:
        results = pool.starmap(pdf_to_text_with_structure, args_list)

    # Write the results to CSV
    for df, _ in results:
        if df is not None:
            if file_exists:
                df.to_csv(output_file_path, mode='a', header=False, index=False, sep="|")
            else:
                df.to_csv(output_file_path, index=False, sep="|")
                file_exists = True
                
# Call the function
process_folder(pdf_folder_path, headers, output_file_path)