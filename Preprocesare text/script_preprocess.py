import re

def goodLetter(s):# bool vf daca caracterul e cifra litera semn de punctuatie
	if not (s in '%-,.;?!# "' or (s >='a' and s<='z') or (s>='0' and s<='9')):
		return False
	return True

def split(line):
	return [char for char in line] 

def listostring(list):
	s=""
	for i in list:
		s+=i
	return s

def correctedLine(line):#returns a line with good letters and converted to lower
	line=line.lower()
	characters=split(line)
	i=0
	while i in range(len(characters)):
		if not goodLetter(characters[i]):
			characters=characters[:i]+characters[i+1:]
		else:
			i+=1
	return listostring(characters)

def clean_text_regex(text):
	new_text=text
	rgx_m='<[^>]*>|@[a-zA-Z]+|[H,h]ttp[s]?:\/\/[\/]?[a-zA-Z\. ?=0-9\-_]*[\/]?([a-zA-Z\.?=0-9\-]*[\/]?)*|([a-zA-Z]*.[a-zA-Z]*\/)[\/]?[a-zA-Z\. ?=0-9\-_]*[\/]?([a-zA-Z\.?=0-9\-]*[\/]?)*|\d+:\d+'
	new_text=re.sub(rgx_m, '', new_text)
	return new_text

fr = open("file.txt", encoding="utf8")
lines=fr.readlines()
fr.close()
fo= open("preprocessed.txt", "w")
for line in lines:
	fo.write(correctedLine(clean_text_regex(line)))
	fo.write("\n")
fo.close()
