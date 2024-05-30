import cv2
from paddlenlp.utils.doc_parser import DocParser
import base64
import numpy as np
import matplotlib.pyplot as plt
from paddlenlp import Taskflow
from PIL import Image, ImageDraw, ImageFont

schema = ["商品名称", "规格", "数量", "生产日期", "产地/生产厂家", "单位", "件数", "单价", "金额", "编号", "有效期", "剂型", "批号", "批准文号"]

colors = ['#8B4513', '#01c26d', '#ae899b', '#c3667e', '#c6498e', '#0a9a8e', '#e5e40f', '#dd4415', '#9fa7d8', '#86860a',
          '#394dbf', '#696969', '#ba8a8b', '#fd0a59', '#556B2F', '#1f95e9', '#003ae6', '#99a182', '#35b984', '#7af380',
          '#41463f', '#f6ffd9', '#2b5ce8', '#20369b', '#f6e6e5', '#461342', '#169e0b', '#b44073', '#e4cb0a', '#7f8c68']  #######颜色记得改，有时候不是30个类别

font = ImageFont.truetype("FZYDZHJW.TTF") 

label2color = {label: colors[idx] for idx, label in enumerate(schema)}
print(label2color)

my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best', precison='fp16')

doc_path = "test.jpg"
image = Image.open(doc_path).convert("RGB")  # image打开图片
# print(image.size)
# 旋转图像
image = image.rotate(90)

doc_parser = DocParser()
results = my_ie({"doc": doc_path})


draw = ImageDraw.Draw(image)

for idx in results[0].keys():
    res = results[0][idx][0]
    draw.rectangle(res['bbox'][0], outline=label2color[idx], width=2)
    draw.text((res['bbox'][0][0] + 10, res['bbox'][0][1] - 10), text=idx, fill=label2color[idx], font=font)


image.save("vis.jpg")