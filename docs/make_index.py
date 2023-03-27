## Read template.html, replace {{ file }} with the contents of infills/file.html, 
## and write the result to index.html
import os
import re
import sys

def read_file(path):
    with open(path, 'r') as f:
        return f.read()
    
def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def main():
    template = read_file('template.html')
    infills = {}
    for filename in os.listdir('infills'):
        infills[filename] = read_file(os.path.join('infills', filename))
    index = template
    for filename, content in infills.items():
        index = index.replace('{{ ' + filename + ' }}', content)
    write_file('index.html', index)

if __name__ == '__main__':
    main()