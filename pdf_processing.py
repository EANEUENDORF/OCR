
import os
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
import csv

from itertools import product
from PIL import Image

# Path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your path

headers = ["#","Companies", "Ownership Status", "Description","Deal Type","Deal Type 2","Date","Pre-money Valuation","Raised to Date","Size","Revenue","Employees","Investors","Lead/Sole Investors","HQ Location","Financing Status","Business Status","Primary Industry Code","Verticals","Deal Status","Company Website"]

def pdf_to_text_with_structure(pdf_path, headers):
    """
    Convert pdf file to a DataFrame using OCR and positional data.
    Parameters:
    - pdf_path: str, path to the pdf file.
    - headers: list, list of column headers to structure the data.
    Returns:
    - df: DataFrame, structured data from pdf.
    """
    words_and_coordinates = []  # <- This will store our word-coordinate data
    try:
        # Convert PDF to images
        Image.MAX_IMAGE_PIXELS = 500000000
        images = convert_from_path(pdf_path, 500)  # 500 dpi to improve OCR quality
        
        # Data placeholder
        all_columns_data = {header: [] for header in headers}
        
        # Define a vertical range for header detection
        y_range = 25  # Modify as needed
        
        # Iterate through each image
        for img in images:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            # Append word data to our list
            for i in range(len(data['text'])):
                words_and_coordinates.append({
                    'word': data['text'][i],
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i]
                })

            column_positions = {}
            for header in headers:
                words = header.split()
                if len(words) > 1:
                    # For headers that consist of multiple words
                    word_indices = [[i for i, text in enumerate(data['text']) if text == word and abs(data['top'][i] - 787) <= y_range] for word in words]
                    
                    # DEBUG: Print each word and its matched indices
                    for word, indices in zip(words, word_indices):
                        print(f"Word: {word}, Matched Indices: {indices}")

                    # Find the combination of words with the smallest distance and smallest x value for the first word
                    min_distance = float('inf')
                    min_x = float('inf')
                    chosen_indices = []
                    for combination in product(*word_indices):
                        distance = data['left'][combination[-1]] - data['left'][combination[0]]
                        if 0 < distance < min_distance or (distance == min_distance and data['left'][combination[0]] < min_x):
                            min_distance = distance
                            min_x = data['left'][combination[0]]
                            chosen_indices = combination

                    if chosen_indices:
                        column_positions[header] = data['left'][chosen_indices[0]]
                    else:
                        # Check for partial matches
                        for word_index_list in word_indices:
                            if word_index_list:  # if there's at least one match for any word in the header
                                column_positions[header] = data['left'][word_index_list[0]]
                                break  # use the first matching word's position and exit loop
                else:
                    # For headers that consist of a single word
                    header_indices = [i for i, text in enumerate(data['text']) if text == header and abs(data['top'][i] - 787) <= y_range]
                    
                    # DEBUG: Print the header and its matched indices
                    print(f"Header: {header}, Matched Indices: {header_indices}")

                    if header_indices:
                        column_positions[header] = data['left'][header_indices[0]]

            # Manually insert the "Ownership Status" header
            column_positions["Ownership Status"] = 1600
            # DEBUG: Print column positions
            print(f"Column Positions: {column_positions}")

            # Find the x-coordinate of the first header
            first_header_x_position = min(column_positions.values())
            # Exclude words above the header line and to the left of the first header
            header_y_position = 787
            valid_indices = [
                i for i in range(len(data['text'])) 
                if data['top'][i] >= header_y_position-1 and data['left'][i] >= first_header_x_position-20
            ]
            valid_indices = sorted(valid_indices, key=lambda i: (data['left'][i], data['top'][i]))

            # Step 1: Using the sorted headers by their x-coordinates.
            sorted_headers = sorted(column_positions.items(), key=lambda x: x[1])
            # DEBUG: Print sorted headers
            print(f"Sorted Headers: {sorted_headers}")

            # Step 2: Group words by their y-coordinate to form rows.
            y_range2 = 11
            y_range3 = 22 
            grouped_by_y = {}
            for i in valid_indices:
                y = data['top'][i]
                found_group = False
                for existing_y in grouped_by_y.keys():
                    absolute_difference = abs(existing_y - y)
                    is_within_y_range2 = absolute_difference <= y_range2
                    is_larger_by_y_range3 = 0 <= (y - existing_y) <= y_range3
                    
                    if is_within_y_range2 or is_larger_by_y_range3:
                        grouped_by_y[existing_y].append(i)
                        found_group = True
                        break

                if not found_group:
                    grouped_by_y[y] = [i]

            # # Checking and merging groups that fulfill the conditions
            # keys_list = list(grouped_by_y.keys())
            # for y1 in keys_list:
            #     if y1 not in grouped_by_y:
            #         continue
            #     potential_merge_keys = []
            #     for y2 in keys_list:
            #         if y1 != y2 and y2 in grouped_by_y:
            #             absolute_difference = abs(y1 - y2)
            #             is_within_y_range2 = absolute_difference <= y_range2
            #             is_larger_by_y_range3 = 0 <= (y2 - y1) <= y_range3
            #             if is_within_y_range2 or is_larger_by_y_range3:
            #                 potential_merge_keys.append(y2)

            #     # Find the key with the longest string and merge others to it
            #     if potential_merge_keys:
            #         all_keys = [y1] + potential_merge_keys
            #         longest_key = max(all_keys, key=lambda k: len(grouped_by_y[k]))
            #         for key in all_keys:
            #             if key != longest_key:
            #                 grouped_by_y[longest_key].extend(grouped_by_y[key])
            #                 grouped_by_y[longest_key].sort(key=lambda x: valid_indices.index(x))  # Retaining the sort order of valid_indices
            #                 del grouped_by_y[key]  # Remove the merged key

            # DEBUG 
            for key, indices in grouped_by_y.items():
                words = [data['text'][index] for index in indices]
                print(f"Y Coordinate: {key} -> Words: {', '.join(words)}")
                
            rows_data = []
            # Sort the grouped_by_y dictionary by keys (i.e., y-coordinates)
            sorted_grouped_by_y = dict(sorted(grouped_by_y.items()))
            for y, indices in sorted_grouped_by_y.items():
                row_data = {}
                for index in indices:
                    word = data['text'][index]
                    word_x = data['left'][index]
                    for j, (header, x_pos) in enumerate(sorted_headers):
                        if j == len(sorted_headers) - 1:
                            if word_x >= x_pos:
                                if header in row_data:
                                    row_data[header] += " " + word
                                else:
                                    row_data[header] = word
                        else:
                            if x_pos <= word_x < sorted_headers[j+1][1]:
                                if header in row_data:
                                    row_data[header] += " " + word
                                else:
                                    row_data[header] = word
                rows_data.append(row_data)

            # Append this image's data to the overall data
            for row in rows_data:
                for header in headers:
                    all_columns_data[header].append(row.get(header, None))

            # Handle empty cells
            max_len = max(len(col) for col in all_columns_data.values())
            for header in headers:
                while len(all_columns_data[header]) < max_len:
                    all_columns_data[header].append(None)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_columns_data)
        return df, words_and_coordinates
    except Exception as e:
        print(f"Error in processing {pdf_path}: {str(e)}")
        return
    