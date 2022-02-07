# -*- coding:utf-8 -*-
import argparse
import os, tqdm
import json
import cv2
import pandas as pd
from datetime import datetime

# Color Code for terminal
CGREEN = '\33[32m'
CCYAN  = '\33[36m'
CRED   = '\33[31m'
CEND   = '\33[0m'

DEBUG = None

# def generate_csv_from_json(image, json_data, logs_dir = "logs/", csv_dir = "csvs_dir",name="unknown.jpg"):
### install json, os,cv2, pandas, datetime
### put json in "test_jsons" folder
### generated csv will be saved in "csvs_dir"

parser = argparse.ArgumentParser("json to csv")

parser.add_argument('json_directory',metavar= 'jsons/',type=str,help="Enter the json directory")
parser.add_argument('csv_directory',metavar= "csvs/",type = str, help='Enter csv directory')
parser.add_argument('--DEBUG', metavar='False', type = bool,default=False, help='Toggle Debugging')
parser.add_argument('--test_image_directory',metavar='test_imgs',type = str,default='test_images/', help='Enter image directory')
parser.add_argument('--test_logs',metavar='test_logs',type = str,default='test_logs/', help='Enter logs directory')


def get_sequences(p,page):

    """[summary]
        getSequences: generate sequence of sorted words from words from same region['Line']
    """
    sequences = []

    image,image_name = None ,None
    if DEBUG:
        image_name = page['id']+".jpg"
        image = cv2.imread(imgs_dir + image_name)
    regions = page['regions'] if 'regions' in page else []

    # Gell all the sequence in a list
    for region in regions:
        if "Line" in region['type']:
            if 'words' in region and region['words']:
                words = region['words']
                sentence = ''
                words = sorted(words,key=lambda x:x['x'])

                for word in words:
                    sentence += word['text']

                if sentence.replace(" ", "") != "":
                    confidence = 0
                    total_word = len(region["orginal_words"])
                    for i in range(total_word):
                        confidence = confidence + region["orginal_words"][i]["confidence"]
                    region['text'] = sentence
                    region['confidence'] = confidence/total_word
                    sequences.append(region)
        if "_Table" in region['type']:
            cells = region['cells'] if 'cells' in region else []
            for cell in cells:
                lines = cell['cell'] if 'cell' in cell else []
                for line in lines:
                    words = line['words'] if 'words' in line else []
                    sentence = ''
                    words = sorted(words,key=lambda x:x['x'])

                    for word in words:
                        sentence += word['text']

                    if sentence.replace(" ", "") != "":
                        confidence = 0
                        total_word = len(line["orginal_words"])
                        for i in range(total_word):
                            confidence = confidence + line["orginal_words"][i]["confidence"]
                        line['text'] = sentence.strip()
                        line['confidence'] = confidence/total_word
                        sequences.append(line)
    return sequences,image,image_name

def getLines(sorted_sequences):


    """[summary]
        generates line from sequences from similar y co-ordinate
    """


    total_sequence = len(sorted_sequences)
    lines = []
    lines_position = []

    while(len(sorted_sequences) != 0):
        line = []
        first_sequence = sorted_sequences[0]
        line.append(first_sequence)

        x0 = first_sequence['x']
        y0 = first_sequence['y']
        w0 = x0 + first_sequence['width']
        h0 = y0 + first_sequence['height']

        [x,y,w,h] = [x0,y0,w0,h0]
        delete_sequences = [0]

        for i in range(1,total_sequence):
            xi = sorted_sequences[i]['x']
            yi = sorted_sequences[i]['y']
            wi = xi + sorted_sequences[i]['width']
            hi = yi + sorted_sequences[i]['height']

            if y0 <= yi and yi <= (y0 + h0)/2:
                if xi < x: x = xi
                if yi < y: y = yi
                if wi > w: w = wi
                if hi > h: h = hi

                line.append(sorted_sequences[i])
                delete_sequences.append(i)

        line = sorted(line,key=lambda x:x['x'])
        lines.append(line)
        lines_position.append([x,y,w,h])
        delete_sequences.sort(reverse=True)
        for j in delete_sequences:
            del sorted_sequences[j]

        total_sequence = len(sorted_sequences)
        
    return lines,lines_position



def get_blocks(lines, lines_position):

    """[summary]
    generates block from lines based on differences of Y co-ordinate value, ":"
    """

    total_lines = len(lines)
    blocks = []
    blocks_position = []

    while(len(lines) != 0):
        block = []
        first_line = lines[0]
        block.append(first_line)

        first_sequence = first_line[0]

        x1 = first_sequence['x']
        y1 = first_sequence['y']
        w1 = x1 + first_sequence['width']
        h1 = y1 + first_sequence['height']

        [x,y,w,h] = lines_position[0]

        delete_lines = [0]
        prev_y = h1 + (h1 - y1)

        if first_line[len(first_line)-1]['text'][-1] == ":":pass

        elif first_sequence['text'][-1] == ":" and len(first_line) > 1:
            for i in range(1,total_lines):
                [xi,yi,wi,hi] = lines_position[i]
                [xp,yp,wp,hp] = lines_position[i-1]

                if w1 < xi and prev_y > yi:
                    if xi < x: x = xi
                    if yi < y: y = yi
                    if wi > w: w = wi
                    if hi > h: h = hi

                    prev_y = hi + (hi - yi)
                    block.append(lines[i])
                    delete_lines.append(i)
                else:
                    break

        elif first_sequence['text'][-1] != ":" and len(first_line) > 1:
            colon_count = 0

            for i in range(1,total_lines):
                [xi,yi,wi,hi] = lines_position[i]
                [xp,yp,wp,hp] = lines_position[i-1]

                if hp +(hp - yp)/3 < yi:
                    colon_count = 2
                if lines[i][0]['text'][-1] == ":":
                    colon_count += 1
                if colon_count == 2:
                    break
                # print(yi, prev_y)
                if hp +(hp - yp) > yi:
                    if xi < x: x = xi
                    if yi < y: y = yi
                    if wi > w: w = wi
                    if hi > h: h = hi

                    prev_y = hi + (hi - yi)/3
                    block.append(lines[i])
                    delete_lines.append(i)
                else:
                    break

        else:
            pass

        blocks.append(block)
        blocks_position.append([x,y,w,h])
        delete_lines.sort(reverse=True)
        for j in delete_lines:
            del lines[j]
            lines_position.pop(j)

        total_lines = len(lines)
                          
    return blocks, blocks_position




def generate_csv(p,blocks,blocks_position,image):


    """generateCSV: generates CSV from blocks co-ordinate, l
    ine co-ordinate and sequence co-ordinate."""
    
    prev_line_no = 1 
    csv_data_page = []
    for b in range(0, len(blocks)):                
        lines = blocks[b]
        total_line = len(lines)
        # print(total_line)
        [x,y,w,h] = blocks_position[b]
        cv2.rectangle(image,(x,y),(w,h),(0,0,255),3)              
        cv2.putText(image, str(prev_line_no), (x-75,int((y+h)/2)), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 1, cv2.LINE_AA, False)
        first_line = lines[0]
        first_sequence = first_line[0]

        prev_h = first_sequence['height']
        prev_w = first_sequence['x'] + first_sequence['width']

        text_list = [first_sequence['text']]
        confidence_list = [first_sequence['confidence']]
        confidence_num_list = [1]

        if total_line == 1:
            for i in range(1,len(first_line)):
                next_sequence = first_line[i]
                x = next_sequence['x']
                w = x + next_sequence['width']
                text = next_sequence['text']
                confidence = next_sequence['confidence']
                height = next_sequence['height']

                if x < prev_w + prev_h * 2:

                    top = len(text_list)-1
                    text_list[top] = text_list[top] + " " + text
                    confidence_list[top] = confidence_list[top] + confidence
                    confidence_num_list[top] = confidence_num_list[top]+1
                    prev_w = w
                    prev_h = height

                else:
                    text_list.append(text)
                    confidence_list.append(confidence)
                    confidence_num_list.append(1)
                    prev_w = w

        if total_line > 1:
            temp_text = ""
            temp_confidence = 0
            temp_confidence_num = 0
            for i in range(0,total_line):
                if i == 0:
                    for j in range(1,len(lines[i])):
                        next_sequence = lines[i][j]

                        x = next_sequence['x']
                        w = x + next_sequence['width']
                        text = next_sequence['text']
                        height = next_sequence['height']

                        confidence = next_sequence['confidence']

                        if x < prev_w + prev_h * 2 and len(text_list) < 2:
                            top = len(text_list)-1
                            text_list[top] = text_list[top] + " " + text
                            confidence_list[top] = confidence_list[top] + confidence
                            confidence_num_list[top] = confidence_num_list[top]+1
                            prev_w = w
                            prev_h = height

                        else:
                            temp_text = temp_text + " " + text
                            temp_confidence = temp_confidence + confidence
                            temp_confidence_num = temp_confidence_num+1
                else:
                    # print(lines[i])
                    for j in range(len(lines[i])):
                        text = lines[i][j]['text']
                        temp_text = temp_text + " " + text
                        temp_confidence = temp_confidence + confidence
                        temp_confidence_num = temp_confidence_num+1
            text_list.append(temp_text)
            confidence_list.append(temp_confidence)
            confidence_num_list.append(temp_confidence_num)

        for i in range(len(text_list)):
            csv_data_page.append([p+1,prev_line_no,round((confidence_list[i]*100)/confidence_num_list[i],1),text_list[i]])

        prev_line_no = prev_line_no + total_line
        
    return csv_data_page, image



def generate_csv_from_json(json_data, imgs_dir='imgs/', logs_dir = "logs/", csv_dir = "csvs_dir/",name="unknown.json"):
	os.makedirs(csv_dir, exist_ok=True)
	os.makedirs(logs_dir, exist_ok=True)
	if "original_output" in json_data and "pages" in json_data["original_output"]:
		csv_data = []
		document_name = json_data['original_output']['document_name']
		pages = json_data["original_output"]['pages']
		for p,page in enumerate(pages):
            
			sequences,image,image_name = get_sequences(p,page)
            
			sorted_sequences = sorted(sequences,key = lambda x: (x['y'], x['x']))
            
			lines,lines_position = getLines(sorted_sequences)
                      
			blocks, blocks_position = get_blocks(lines,lines_position)
            
			csv_data_page,image = generate_csv(p,blocks,blocks_position, image)
            
			csv_data += csv_data_page
            
			if DEBUG: cv2.imwrite(logs_dir + image_name, image)
            
		now = datetime.now()
		csv_file_name = ".".join(document_name.split(".")[:-1])+"_"+now.strftime("%Y%m%d%H%M%S")+".csv"
		csv_file_path = csv_dir + csv_file_name
		pd.DataFrame(csv_data).to_csv(csv_file_path ,sep=',',line_terminator='\r\n', header=False, index=False)



if __name__ == "__main__":


	args = parser.parse_args()

	csvs_dir = args.csv_directory
	json_dir = args.json_directory
	DEBUG = args.DEBUG
	imgs_dir = args.test_image_directory
	logs_dir = args.test_logs

	#python generate_csv_from_jsons.py test_imgs test_jsons test_csvs test_logs

	for root, dirs, files in os.walk(json_dir):
		for file in tqdm.tqdm(files):
			if file.endswith(".json"):
				# if file != "01_21__Master.json": continue
				if DEBUG: print(file)

				json_path = os.path.join(root, file)

				with open(json_path, 'r') as outfile:
					json_data = json.load(outfile)

				generate_csv_from_json(json_data,imgs_dir=imgs_dir, logs_dir=logs_dir, csv_dir=csvs_dir, name=file)
