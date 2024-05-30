# 建议安装最新版
import paddlenlp, paddleocr
from pprint import pprint
from paddlenlp import Taskflow
from paddlenlp.utils.doc_parser import DocParser

# 实体抽取

# 海关报关单信息抽取
schema = ["商品名称", "规格", "数量", "生产日期", "产地/生产厂家", "单位", "件数", "单价", "金额", "编号", "有效期", "剂型", "批号", "批准文号"]
# ie = Taskflow("information_extraction", schema=schema, model="uie-m-base", ocr_lang='ch')
# pprint(ie({"doc": "dataset/images/益比奥201702034V.jpg"}))

# doc_parser = DocParser(layout_analysis=False, ocr_lang='ch')


# schema = {
#     '项目名称': [
#         '结果',
#         '单位',
#         '参考范围'
#     ]
# }
my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best', precison='fp16')

doc_path = "test.jpg"
pprint(my_ie({"doc": doc_path}))


