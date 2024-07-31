import streamlit as st

import fitz  # PyMuPDF
import easyocr
import cv2
from PIL import Image
import csv
import os
import math
import re
import sys
import pandas as pd
import math


class SUPPORTED_OCR_ENGINES:
    def __init__(self):
        self.PaddleOCR = "PaddleOCR"
        self.SuryaOCR = "SuryaOCR"
        self.EasyOCR = "EasyOCR"

        
class BoundingBoxElement:
    def __init__(self, obj, ocr_name):
        if ocr_name == "PaddleOCR":
            self.bbox = [obj[0][0][0], obj[0][0][1], obj[0][2][0], obj[0][2][1]]
            self.confidence = obj[1][1]
            self.text = obj[1][0]
            self.polygon = obj[0]
        elif ocr_name == "SuryaOCR":
            self.bbox = obj.bbox
            self.confidence = obj.confidence
            self.text = obj.text
            self.polygon = obj.polygon
        elif ocr_name == "EasyOCR":
            self.bbox = [obj[0][0],obj[0][1],obj[0][2],obj[0][3]]
            self.confidence = obj[2]
            self.text = obj[1]
            self.polygon = obj[0]


    def __str__(self):
        return ", ".join([str(self.text), str(self.confidence), str(self.bbox), str(self.polygon)])

    
    def get_bounding_box_height(self):
        return self.bbox[3][1] - self.bbox[1][1]
    

    def get_bounding_box_right_top(self):
        return self.polygon[1][1]

    def get_bounding_box_text(self):
        return self.text


    def get_bounding_box_top(self):
        return self.bbox[1][1]


    def get_bounding_box_left(self):
        return self.bbox[0]
        

class PageLayoutBuilder:
    def __init__(self, result_obj, ocr_name, tolerance=1):
        self.tolerance = tolerance
        self.ocr_name = ocr_name
        if ocr_name == SUPPORTED_OCR_ENGINES().SuryaOCR:
            self.pageImageBox = result_obj[0].image_bbox
            self.boxes_list = self.create_bbElement_list(result_obj[0].text_lines)
            self.average_box_height = self.get_average_box_height(self.boxes_list)
            self.fixed_page_content = self.fix_page_content_v2(self.boxes_list)
        if ocr_name == SUPPORTED_OCR_ENGINES().PaddleOCR:
            self.pageImageBox = [-1, -1, -1, -1]
            self.boxes_list = self.create_bbElement_list(result_obj[0])
            self.average_box_height = self.get_average_box_height(self.boxes_list)
            self.fixed_page_content = self.fix_page_content_v2(self.boxes_list)
        if ocr_name == SUPPORTED_OCR_ENGINES().EasyOCR:
            self.pageImageBox = [-1, -1, -1, -1]
            self.boxes_list = self.create_bbElement_list(result_obj)
            self.average_box_height = self.get_average_box_height(self.boxes_list)
            self.fixed_page_content = self.fix_page_content_v2(self.boxes_list)


    def create_bbElement_list(self, boxes):
        bb_element_list = []
        for box in boxes:
            bb_element_list.append(BoundingBoxElement(box, self.ocr_name))
        return bb_element_list


    def get_average_box_height(self, boxes_list) -> int:
        total_height = []
        for box in boxes_list:
            total_height.append(box.get_bounding_box_height())
            total_height = sorted(total_height)
        return total_height[math.floor(len(boxes_list) / 2)]


    def fix_page_content(self, boxes_list):
        sorted_boxes_list = sorted(boxes_list, key=lambda box_element: box_element.get_bounding_box_top())
        fixed_page_content = []

        box = sorted_boxes_list.pop(0)
        while len(sorted_boxes_list) > 0:
            fixed_line_content = []
            position = box.get_bounding_box_right_top()
            fixed_line_content.append(box)
            while len(sorted_boxes_list) > 0:
                box = sorted_boxes_list.pop(0)
                if box.get_bounding_box_top() < math.floor((self.average_box_height * .50) + position):
                    position = box.get_bounding_box_right_top()
                    fixed_line_content.append(box)
                else:
                    fixed_page_content.append(
                        sorted(fixed_line_content, key=lambda box_element: box_element.get_bounding_box_left()))
                    break

        return fixed_page_content


    def fix_page_content_v2(self, boxes_list):
        sorted_boxes_list = sorted(boxes_list, key=lambda box_element: box_element.get_bounding_box_top())
        fixed_page_content = []

        box = sorted_boxes_list.pop(0)
        position = box.get_bounding_box_right_top()
        height = box.get_bounding_box_height()
        while len(sorted_boxes_list) > 0:
            fixed_line_content = []
            fixed_line_content.append(box)
            while len(sorted_boxes_list) > 0:
                position = box.get_bounding_box_right_top()
                height = box.get_bounding_box_height()
                box = sorted_boxes_list.pop(0)
                if box.get_bounding_box_top() < math.floor(((height / 2) * self.tolerance) + position):
                    fixed_line_content.append(box)
                else:
                    fixed_page_content.append(
                        sorted(fixed_line_content, key=lambda box_element: box_element.get_bounding_box_left()))
                    break

        return fixed_page_content


    def fix_page_content_v3(self, boxes_list):
        sorted_boxes_list = sorted(boxes_list, key=lambda box_element: box_element.get_bounding_box_top())
        fixed_page_content = []

        box = sorted_boxes_list.pop(0)
        position = box.get_bounding_box_right_top()
        height = box.get_bounding_box_height()
        while len(sorted_boxes_list) > 0:
            fixed_line_content = []
            fixed_line_content.append(box)
            while len(sorted_boxes_list) > 0:
                position = box.get_bounding_box_right_top()
                height = box.get_bounding_box_height()
                box = sorted_boxes_list.pop(0)
                if box.get_bounding_box_top() < math.floor(((height / 2) * self.tolerance) + position):
                    fixed_line_content.append(box)
                else:
                    fixed_page_content.append(
                        sorted(fixed_line_content, key=lambda box_element: box_element.get_bounding_box_left()))
                    break

        return fixed_page_content


    def get_page_content(self):
        return self.fixed_page_content


    def get_page_text(self) -> list:
        page_text = []
        
        for lines in self.fixed_page_content:
            line_text = []
            for box in lines:
                line_text.append(box.text)
            
            page_text.append(line_text)
        return page_text


    def get_raw_text(self) -> str:
        raw_text = ""
        for lines in self.fixed_page_content:
            line_text = []
            for box in lines:
                line_text.append(box.text)
            raw_text = raw_text + ' '.join(line_text) + os.linesep
        return raw_text




class PocTadawal:

    def pdf_to_image(self,pdf_path, output_image_path, image_format='PNG'):
        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)
            image_files = []
            pdf_document = fitz.open(pdf_path)
        
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(dpi=600)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            filename_with_path = (f"{output_image_path}/{os.path.basename(pdf_path).split('.')[0]}"
                                    f"_page-{page_number+1}.{image_format}")
            img.save(filename_with_path, format=image_format)
            image_files.append(filename_with_path)

        pdf_document.close()

        return image_files



    def page_in_layout(self,data, tolerance=10, x_diff_threshold=500):
        if not data:
            return []

        boxes_list = []
        for bbox in data:
            coordinates = bbox[0][0]
            text = bbox[1]        
            width = bbox[0][1][0] - bbox[0][0][0]
            boxes_list.append([coordinates[0],coordinates[1], text, width] )
          
        sorted_boxes_list = sorted(boxes_list, key=lambda box_element: box_element[1])
        fixed_page_content = []
        box = sorted_boxes_list.pop(0)
        while len(sorted_boxes_list) > 0:
            fixed_line_content = []
            position = box[1]
            fixed_line_content.append(box)
            while len(sorted_boxes_list) > 0:
                box = sorted_boxes_list.pop(0)
                if box[1] < math.floor((position+(41/2))):
                    position = box[1]
                    fixed_line_content.append(box)
                else:
                    fixed_page_content.append(
                    sorted(fixed_line_content, key=lambda box_element: box_element[0]))
                    break
      
        max_column = 0
        row = 0
        results = []
        for index, box in enumerate(fixed_page_content):
            if len(box) > max_column:
                max_column = len(box)
                row = index
                max_column_size = [0] * max_column
          
        for box in fixed_page_content:
            col = []
            for index, column in enumerate(box):
                max_column_size[index] = max(max_column_size[index],column[3])    
            for i in range(max_column):
                col.append([-1,-1,""])
                results.append(col)
        column_coordinates = fixed_page_content[row]
        margin = 0
        for i in range(len(fixed_page_content)):
            for j in range(len(fixed_page_content[i])):
                placed = False
                divider = 10
                while not placed:
                    divider = divider * 2
                    for k in range(len(column_coordinates)):
                      # margin = max_column_size[k]+ divider
                        margin =  divider
                        if column_coordinates[k][0]-margin < fixed_page_content[i][j][0] < column_coordinates[k][0] + margin *10.5:
                            results[i][k] = fixed_page_content[i][j]
                            placed = True
                            break
                  
        return results


    def fill_nd_list_4(self, n_dimensional_list, index_l, max_len):
        index = 0
        temp = True
        for i in range(len(index_l)):
            for j in range(len(index_l[i])):
                if j == len(index_l[i]) - 1:
                    if index_l[i][j] == max_len:
                        if temp:
                            index = i
                            temp = False
                        for k in range(len(index_l[i]) - 2, -1, -1):
                            n_dimensional_list[i][k] = index_l[i][k][0][1]
        return n_dimensional_list, index


    def fill_nd_list_down(self, n_dimensional_list, index_l, max_entry, index):
        list_max_coord = []
        dist = sys.maxsize
        pos_of_text = 0
        for i in range(len(max_entry)):
            list_max_coord.append([max_entry[i][1][0][0], max_entry[i][1][0][1]])

        for i in range(index + 1, len(index_l)):
            # print(i)
            for k in range(len(index_l[i])):
                if k == len(index_l[i]) - 1:
                    for j in range(len(index_l[i]) - 2, -1, -1):
                        for l in range(len(list_max_coord)):
                            eucli_distance = self.euclidean_distance(index_l[i][j][0][2][0][0], index_l[i][j][0][2][0][1], list_max_coord[l][0], list_max_coord[l][1])
                            if eucli_distance < dist:
                                dist = eucli_distance
                                pos_of_text = l
                            try:
                                n_dimensional_list[i][pos_of_text] = index_l[i][j][0][1]
                            except:
                                continue
                        dist = sys.maxsize    

        return n_dimensional_list


    def create_nd_list(self,max_entry, get_page_text):
        return [["" for _ in range(max_entry)] for _ in range(len(get_page_text))]


    def euclidean_distance(self,x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def fill_nd_list_up(self, n_dimensional_list, index_l, max_entry, index):
        list_max_coord = []
        dist = sys.maxsize
        pos_of_text = 0
        incrementor = 0

        for i in range(len(max_entry)):
            list_max_coord.append([max_entry[i][1][0][0], max_entry[i][1][0][1]])
        print(len(list_max_coord))

        for i in range(index - 1, -1, -1):
            # print(index_l[i])
            for k in range(len(index_l[i])):
                # print(index_l[i][k])
                if k == len(index_l[i]) - 1:
                    for j in range(len(index_l[i]) - 2, -1, -1):
                        for l in range(len(list_max_coord)):
                            eucli_distance = self.euclidean_distance(index_l[i][j][0][2][0][0], index_l[i][j][0][2][0][1], list_max_coord[l][0], list_max_coord[l][1])

                            if eucli_distance < dist:
                                dist = eucli_distance
                                pos_of_text = l

                            try:
                                n_dimensional_list[i][pos_of_text] = index_l[i][j][0][1]
                            except: 
                                continue
                        dist = sys.maxsize
        self.fill_nd_list_down(n_dimensional_list, index_l, max_entry, index)
        return n_dimensional_list


    def page_in_layout1(self, data, tolerance=10, x_diff_threshold=500, tolerance1=1):
        page_layout_builder = PageLayoutBuilder(data, 'EasyOCR')
        fixed_page_content = page_layout_builder.fix_page_content_v3(page_layout_builder.boxes_list)
        get_page_text = page_layout_builder.get_page_text()
        l1 = []

        max_len = 0
        max_column = []
        for i in get_page_text:
            if max_len < len(i):
                max_len = len(i)

        max_entry = []
        for i in get_page_text:
            if max_len == len(i):
                max_entry.append(i)
                break
        l2 = []
        l3 = []
        l4 = []

        for i in range(len(get_page_text)):
            for j in range(len(get_page_text[i])):
                if get_page_text[i][j] == fixed_page_content[i][j].text:
                    l1.append([i, fixed_page_content[i][j].text, fixed_page_content[i][j].bbox])

        index = -1
        line_length = -1

        for i, box in enumerate(fixed_page_content):
            if line_length < len(box):
                line_length = len(box)
                index = i
        l2 = []
        for box in fixed_page_content[index]:
            l2.append([box.text, box.bbox])
        n_dimensional_list = self.create_nd_list(max_len, get_page_text)

        index_to_entries = []
        for entry in l1:
            index = entry[0]
            while len(index_to_entries) <= index:
                index_to_entries.append([])
            index_to_entries[index].append([entry])
        for sublist in index_to_entries:
            sublist.append(len(sublist))

        n_dimensional_list, index = self.fill_nd_list_4(n_dimensional_list, index_to_entries, max_len)
        n_dimensional_list = self.fill_nd_list_up(n_dimensional_list, index_to_entries, l2, index)
        return n_dimensional_list, max_len
    
    
    def write_filtered_data_to_csv(self,data, filename,output_dir):
        if not os.path.exists(output_dir):   
            os.makedirs(output_dir)
        df = pd.DataFrame(data)

      # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        

    def extract_tables_from_pdf1(self,pdf_path,file_path ,output_dir):
        if not os.path.exists(output_dir):   
            os.makedirs(output_dir)
        reader = easyocr.Reader(['en'], gpu=True)
        data1 = []
        page_number = 0
        image = cv2.imread(pdf_path)
        results = reader.readtext(image)
        n_dimensional_list,max_len=self.page_in_layout1(results)
        return n_dimensional_list,max_len
    
    
    def save_text_to_csv(self,text, filename):
        lines = text.split(os.linesep)  # Split the text into lines
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            for line in lines:
                csvwriter.writerow([line]) 
                

    def check_for_column(self, results, image_path, output_dir, output_dir1, output_file):
        if not os.path.exists(output_dir):   
            os.makedirs(output_dir)
        number_counter = 0
        sentence_counter = 0
        date_pattern = re.compile(
            r'(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)|'  # Matches dates like 31-12-2020, 31/12/2020, 31-12-20, etc.
            r'(\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b)|'    # Matches dates like 2020-12-31, 2020/12/31
            r'(\b\d{1,2}\s+\w+\s+\d{4}\b)|'          # Matches dates like 31 December 2020
            r'(\b\w+\s+\d{1,2},\s+\d{4}\b)'          # Matches dates like December 31, 2020
        )

        phone_pattern = re.compile(
            r'(\+?\d{1,2}[-.\s]??\d{1,4}[-.\s]??\d{1,4}[-.\s]??\d{1,9})|'  # Matches +xx-xxxx-xxxx, xxxx-xxxx, xxx-xxxx-xxxx etc.
            r'(\b\d{2,4}[-.\s]\d{3,4}[-.\s]\d{4,6}\b)'                     # Matches formats like xx-xxx-xxxx or xxxx-xxxx
        )

        address_pattern = re.compile(
            r'\b(?:PO Box|P\.O\. Box|Post Office Box)\b|\d{1,5}\s\w+\s\w+|\d{1,5}\s\w+'  # Matches PO Box or addresses with numbers
        )

        page_layout_builder = PageLayoutBuilder(results, 'EasyOCR')
        fixed_page_content_v3 = page_layout_builder.fix_page_content_v3(page_layout_builder.boxes_list)
        page_text = page_layout_builder.get_page_text()
        for i in page_text:
            for j in i:
                if not (any(char.isdigit() for char in j)) and not date_pattern.search(j) and not phone_pattern.search(j) and not address_pattern.search(j):
                    sentence_counter += 1
        for i in page_text:
            for j in i:
                if (any(char.isdigit() for char in j)) and not date_pattern.search(j) and not phone_pattern.search(j) and not address_pattern.search(j):
                    number_counter += 1
        print(os.linesep, sentence_counter, " ,", number_counter, " , ", f"{image_path}")
        if sentence_counter > number_counter:
            raw_text = page_layout_builder.get_raw_text()
            self.save_text_to_csv(raw_text, f'{output_dir}/{output_file}.csv')
        else:
            n_dimensional_list, max_len = self.extract_tables_from_pdf1(image_path, output_file, output_dir1)
            self.write_filtered_data_to_csv(n_dimensional_list, f'{output_dir1}/{output_file}.csv', output_dir1)


st.title('PDF to CSV Processor')
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file is not None:
    base_dir = '/content/drive/MyDrive'
    temp_dir = os.path.join(base_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    pdf_path = os.path.join(temp_dir, pdf_file.name)
    
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    st.success(f"Uploaded {pdf_file.name}")

    output_dir = os.path.join(base_dir, 'text-data')
    output_dir1 = os.path.join(base_dir, 'table-data')
    output_image_path = os.path.join(base_dir, 'image-data')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_image_path, exist_ok=True)

    st.write("Processing PDF...")
    progress_bar = st.progress(0)
    total_steps = 2
    poc_tadawal_instance = PocTadawal()
    image_files=[]
    image_files=poc_tadawal_instance.pdf_to_image(pdf_path, output_image_path)
    progress_bar.progress(1 / total_steps)

    reader = easyocr.Reader(['en'], gpu=True)

    for i, file_path in enumerate(image_files):
        file_csv = os.path.splitext(os.path.basename(file_path))[0]
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray_image)
        poc_tadawal_instance.check_for_column(results, file_path, output_dir, output_dir1, file_csv)
        progress_bar.progress((i + 2) / (len(image_files) + 1))


    st.success("Processing Complete!")

    pdf_base_name = os.path.splitext(pdf_file.name)[0]

    st.write("Generated CSV files:")
    for filename in os.listdir(output_dir):
        if filename.startswith(pdf_base_name):
            st.write(f"Text CSV: {filename}")
            st.download_button('Download', os.path.join(output_dir, filename), filename)

    for filename in os.listdir(output_dir1):
        if filename.startswith(pdf_base_name):
            st.write(f"Table CSV: {filename}")
            st.download_button('Download', os.path.join(output_dir1, filename), filename)