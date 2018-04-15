"""
code to clean gutenburg text, run:

    python src/process_data.py data/shelley.txt
"""
import sys, os
import re

def single_space(s):
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s


def poetry_clean(text):
    # clean numbers
    text = re.sub(r'\_?[0-9]+\.?', '', text)
    text = text.replace(':', ' ')
    return text
    

def process_text(target):
    text = []
    with open(target, mode='r') as f:
        in_quote = False
        for line in f:
            if 'NOTES ON THE TEXT' in line:
                break
            if line.startswith('[') or line.startswith('('):
                in_quote = True
            if ']' in line or ')' in line:
                in_quote = False
                continue
            if not in_quote:
                line = line.rstrip()
                line = poetry_clean(line)
                line = single_space(line)
                text.append(line)
    return text


if len(sys.argv) != 2:
    sys.exit('supply target file')
target = sys.argv[1]
print('Cleaning file:', target)

text = process_text(target)
outfile = target.replace('.txt', '_clean.txt')

with open(outfile, mode='w') as f:
    for line in text:
        f.write(line+'\n')

print('Done')
