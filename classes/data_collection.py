#This script contains the code to collect data from the internet based on a list of websites/urls
import fitz
import re
import requests
import os
import regex
import json
from ftlangdetect import detect
def is_E_language(text:str):
    result = detect(text=text, low_memory=False)
    # rs2 = lang_detect(text)
    if result["lang"] == "en":
        return True
    
    return False


pattern = r"(\n|^)(table of contents|contents|list of figures|list of tables)\n"
ending_pattern = r"(\n|^)(REFERENCES\n|Acknowledgements\n|Appendix\b)"
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        data = list()
        big_text = ""
        page_count = doc.page_count

        count = 0
        for page in doc:
            # blocks = page.get_text("blocks")
            text = page.get_text()
            count += 1
            if regex.search(ending_pattern, text, regex.IGNORECASE) and count > page_count/2:
                break
            if regex.search(pattern, text, regex.IGNORECASE):
                continue
            big_text += " "+ text.strip()
    big_text = big_text.replace("-\n", "")
    data = fixing(big_text.strip())
    # data = fix_blocks(data)
    groups = group_by_section(data)
    groups = group_small_blocks(groups)
    groups = [g for g in groups if is_good_text(g)]
    return groups


def group_small_blocks(blocks):
    i = len(blocks)
    i -= 1
    while i >= 0:
        block = blocks[i]
        splits = block.split("\n")
        first = splits[0]
        if check_header(first):
            flag = True
        else:
            flag = False
        if (len(block) < 512 and not flag) or blocks[i-1][-1] == ":" or len(block) < 256:
            blocks[i-1] =  blocks[i-1]+ "\n" + block
            blocks.pop(i)
            i -= 1
        else:
            i -= 1
    return blocks
def preprocess_text(text):
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    text = text.replace("\\r", " ")
    text = text.replace("- ", "")
    multint = re.compile('[\n]+')
    text = multint.sub('\n', text)

    return text.strip()
def is_good_text(text):
    if text.endswith(":"):
        return True
    if not check_alphabic(text):
        return False
    splits = text.split(" ")
    count = 0
    for s in splits:
        if len(s) == 1:
            count += 1
    if count/len(splits) >= 0.5:
        return False

    return True

def concating(blocks):
    i = len(blocks)
    i -= 1
    while i >= 0:
        block = blocks[i]
        if block.startswith("\u2022"):
            blocks[i-1] =  blocks[i-1]+ "\n" + block
            blocks.pop(i)
        else:
            i -= 1
    return blocks
def fixing(text):
    text = preprocess_text(text)
    split = text.split("\n")
    split = [s.strip() for s in split if s.strip() != "" and not s.strip().isnumeric()]
    temp = [s for s in split if s.startswith("\u2022")]
    blocks = remove_repeated_block_of_text(split)
    blocks = fix_blocks(blocks )
    blocks = concating(blocks)
    # blocks2 = [b if is_good_text(b) else "\n\n" for b in blocks ]
    return blocks
def remove_repeated_block_of_text(blocks):
    marker = [0] * len(blocks)
    for i in range(len(blocks)-1):
        if marker[i] == 1:
            continue
        text = blocks[i]
        checks = [text]
        dupplicate_index = [i]
        for j in range(i+1, len(blocks)):
            if marker[j] == 1:
                continue
            text2 = blocks[j]
            if text == text2:
                checks.append(text2)
                dupplicate_index.append(j)
        if len(checks) > 5 :
            print()
            marker[i] = 1
            for k in dupplicate_index:
                marker[k] = 1
    new_blocks = [blocks[i] for i in range(len(blocks)) if marker[i] == 0]
    return new_blocks
        

def fix_blocks(splits):
    sents = list()
    if len(splits) == 1:
        sents.append( splits[0])
    i = 0
    while i < len(splits)-1:
        sent = splits[i]
        # if i == len(splits)-1:
        #     sents.append(sent)
        #     break
        if (sent.startswith("Figure") or sent.startswith("Table")) and sent[-1] not in [".", "!", "?", ":" ]:
            i += 1
            continue #remove captions
        if check_header(sent):
            sents.append(sent)
            i += 1
            continue
        # if not sent[0].isalpha() and sent[-1] not in [".", "!", "?", ":" ]:
        #     sents.append(sent) # list item
        #     i += 1
        #     continue
        if sent.startswith("\u2022"):
            flag = True
        else:
            flag = False
        for j in range(i+1,len(splits)):
            i = j
            sent2 = splits[j]
            if sent2.startswith("\u2022"):
                sents.append(sent)
                break
            is_header= check_header(sent2)
            if sent.endswith("\\") or sent.endswith("vs.") or sent.endswith("i.e.") or sent.endswith("e.g.") or sent.endswith(",") or sent.endswith("as") or (sent2[0].islower() and not is_header  ) or (sent[-1] not in [".", "!", "?", ":" ]):
                if flag and sent2[0].isupper() and not sent.endswith("\\") and not sent.endswith("vs.") and not sent.endswith("e.g.") and not sent.endswith("i.e.") and not sent.endswith("as") and not sent.endswith(","):
                    sents.append(sent)
                    break
                sent += " " + sent2
            else:
                sents.append(sent)
                break
            if j == len(splits)-1 and sent not in sents:
                sents.append(sent)
                break
            
    return sents 

def check_header(text):
    #check if this line is a header 1. 2.2...
    pattern = r"\b[0-9]{1,3}\.([0-9]{1,3}\.?)?\s"
    match = regex.search(pattern, text, regex.IGNORECASE)
    if match:
        if match.start(0) == 0:
            return True

    splits = text.split(" ")
    splits = [s.strip() for s in splits if s.strip() != ""]
    count = 0
    for s in splits:
        if s[0].isupper():
            count += 1
    if (count/len(splits) >= 0.5 and len(splits) > 2) or text.isupper():
        return True
    
    return False
def group_by_section(data):
    
    marks = list()
    marks.append(0)
    for i in range(1, len(data)):
        text = data[i]
        if check_header(text):
            marks.append(i)
    marks.append(len(data))   
    new_data = list()
    i = 0
    while i < len(marks):
        start = marks[i]
        end = marks[i+1] if i+1 < len(marks) else len(data)
        # if end - start <=1:
        #     i += 1
        #     continue
        text = "\n".join(data[start:end])
        if text.strip() != "":
            new_data.append(text)
        i = i+1
    return new_data

def remove_citations(text):
    pattern = "\[[0-9]{1,5}\]"
    text = regex.sub(pattern, " ", text)
    text = text.replace("  ", " ")
    return text.strip()


def download_pdf(urls, saved_dir):
    pass


def download_html(urls, saved_dir):
    pass

def convert_pdf_to_text(pdf_path, saved_file):
    texts = extract_text_from_pdf(pdf_path)
    data = []
    for i in range(len(texts)):
        text = texts[i]
        data.append({"id": i, "text": text})
    
    with open(saved_file, "w") as f:
        json.dump(data, f, indent=4)
from urllib.request import urlopen,urlretrieve
from inscriptis import get_annotated_text, ParserConfig
annotation_rules = {
           'h1': [ 'h1'],
           'h2': [ 'h2'],
            'h3': [ 'h3'],
           'h4': [ 'h4'],

           'b': ['emphasis', 'bold'],
	   'i': ['emphasis', 'italic'],
	   'div#class=toc': ['table-of-contents'],
	   '#class=FactBox': ['fact-box'],
           'table': ['table'],
           'figure': ['figure'],
           'figcaption': ['figcaption'],
           'img': ['image'],
}
from urllib.request import Request, urlopen
remove_captions = r"(\n|^)(Figure|Table)\s[0-9]{1,3}(.|:)\s"
def check_alphabic(text):
    if len(text) == 0 or text =="":
        return False
    text= preprocess_text(text)
    if not is_E_language(text):
        return False # not english
    if regex.search(remove_captions, text, regex.IGNORECASE):
        return False #remove captions
    count = 0
    split_words = text.split(" ")
    for word in split_words:
        if word[0].isupper() or not (word.isalnum()):
            count += 1
    if count/len(split_words) >= 0.5: #most of word is upper case..., this is a tittle
        return False
    total = len(text)
    if total == 0:
        return False
    count = 0
    for i in range(0, total):
        if text[i].isalpha():
            count += 1
    rate = count/total
    if rate >= 0.5 and len(text) > 64: #most of the text is alphabic
        return True

    return False
def check_text(text):
    count = 0
    split = text.split("\n")
    total = len(split)
    for s in split:
        if check_alphabic(s):
            count += 1
    if count/total >= 0.5:
        return True
    return False
def check_section(text):
    return_text = ""
    text = text.strip().replace("\n\n", "\n")
    splits = text.split("\n")
    print()
    splits = [s.strip() for s in splits if check_alphabic(s)]
    if len(splits) >= 1:
        return_text = "\n".join(splits)
    return return_text.strip()
def convert_html_to_text(url):
    req = Request(
    url=url, 
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
)
    html = urlopen(req)
    if html.getcode() != 200:
        return None
    rss = html.read().decode('utf-8')
    output = get_annotated_text(rss, ParserConfig(annotation_rules=annotation_rules))
    text =output["text"]
    annotations = output["label"]
    text2 = list(text)
    headings = []
    for anno in annotations:
        start =anno[0]
        end = anno[1]
        label = anno[2]
        if label in ['h1','h2','h3']:
            headings.append((text[start:end],start,end))
        if label in ["table", "figure", "figcaption",'table-of-contents', 'img']: #delete table, figure, figcaption
            for i in range(start, end):
                text2[i]= " "
    text2 = "".join(text2)

    sections = []
    if len(headings) == 0:
        sections.append(text2)
    else:
        for i in range(len(headings)-1):
            start = headings[i][2]
            j =i+1
            end = headings[j][1]
            if end > start:
                sections.append(text2[start:end])
        sections.append(text2[headings[-1][2]:])

    return_section = []
    for section in sections:
        checked_section = check_section(section)
        if checked_section != "":
            return_section.append(checked_section)
    return return_section


def download_and_analyze_pdf(pdf_url, pdf_id, local_dir:str="", analyzed_dir:str=""):
    file_name = str(pdf_id)+".pdf"
    analyzed_file = str(pdf_id)+".json"
    local_file = os.path.join(local_dir, file_name)
    analyzed_file = os.path.join(analyzed_dir, analyzed_file)
    urlretrieve(pdf_url, local_file)
    convert_pdf_to_text(local_file, analyzed_file)

def download_and_analyze_html(url,html_id, local_dir:str="", analyzed_dir:str=""):
    file_name = str(html_id)+".html"
    analyzed_file = str(html_id)+".json"
    local_file = os.path.join(local_dir, file_name)
    analyzed_file = os.path.join(analyzed_dir, analyzed_file)
    # page = requests.get(url)
    # if page.status_code != 200:
    #     return None
    # with open(local_file, 'wb+') as f:
    #     f.write(page.content)

    texts = convert_html_to_text(url)
    data = []
    if texts is None:
        return None
    for i in range(len(texts)):
        text = texts[i]
        data.append({"id": i, "text": text})
    
    with open(analyzed_file, "w") as f:
        json.dump(data, f, indent=4)
from mitre_attack import MitreAttack
class DataCollector():
    def __init__(self, urls:list=[], saved_dir:str="", fromMITTRE:bool = False):
        self.urls = urls
        self.saved_dir = saved_dir
        self.anaylyzed_dir = os.path.join(saved_dir, "output")
        if len(self.urls) == 0 and fromMITTRE:
            self.url_gathering_fromMITTRE()
            with open(os.path.join(self.saved_dir, "urls.json"), "w") as f:
                json.dump(self.urls, f, indent=4)

    def collect(self):
        pdf_collect = os.path.join(self.saved_dir, "raw_pdf")
        html_collect = os.path.join(self.saved_dir, "raw_html")
        for i in range(0,len(self.urls)):
            url = self.urls[i]
            if url["type"] == "pdf":
                try:
                    download_and_analyze_pdf(url["url"], url["id"], pdf_collect, self.anaylyzed_dir)
                except:
                    print("error")
            else:
                try:
                    download_and_analyze_html(url["url"], url["id"], html_collect, self.anaylyzed_dir)
                except:
                    print("error")

    def url_gathering_fromMITTRE(self):
        links = MitreAttack.get_all_url()
        urls = []
        for i in range(len(links)):
            link = links[i]
            if link.endswith(".pdf") or link.endswith(".PDF"):
                urls.append({"id": i, "url": link, "type": "pdf"})
            else:
                urls.append({"id": i, "url": link, "type": "html"})
                
        self.urls = urls
