import os, sys, json, re
import codecs

JOERN_PATH = "./joern-cli/joern"
JOERN_PARSER_PATH = "./joern-cli/joern-parse"
JOERN_EXPORT_PATH = "./joern-cli/joern-export"

def gencpg(filesPath):
    if not os.path.exists(filesPath+'/cpg_a.txt'):
        cmd = 'cd ./joern; '+JOERN_PATH+' --script ../locateFunc.sc --param inputFile=../'+filesPath+'/a/ --param outFile=../'+filesPath+'/cpg_a.txt'
        os.system(cmd)

    if not os.path.exists(filesPath+'/cpg_b.txt'):
        cmd = 'cd ./joern; '+JOERN_PATH+' --script ../locateFunc.sc --param inputFile=../'+filesPath+'/b/ --param outFile=../'+filesPath+'/cpg_b.txt'
        os.system(cmd)



def make_new_hunkinfo(filesPath):
    os.system('diff -brN -U 0 -p ' + filesPath + 'a/ ' + filesPath + 'b/ >> ' + filesPath + 'new_diff.txt')
    f = open(filesPath + 'new_diff.txt')

    new_hunkInfo = []  # (filename, lineStart, lineEnd)
    filename1 = ''
    filename2 = ''
    while 1:
        line = f.readline()
        if not line:    break
        # print(line)
        if line[:len('diff -brN')] == 'diff -brN':
            filename1 = line.split()[-2]
            filename2 = line.split()[-1]
            filename1 = filename1[filename1.find('/a/'):]
            filename2 = filename2[filename2.find('/b/'):]
        elif line.count('@') == 4:
            i = line.find('@@ ')
            j = line.find(' @@')
            info = line[i + 3:j]

            i = info.find(' ')
            info_a = info[:i]
            info_b = info[i + 1:]

            i = info_a.find(',')
            try:
                if i < 0:
                    new_hunkInfo.append((filename1, int(info_a[1:]), int(info_a[1:])))
                else:
                    if int(info_a[i + 1:]) > 0:
                        new_hunkInfo.append((filename1, int(info_a[1:i]), int(info_a[1:i]) + int(info_a[i + 1:]) - 1))
                    else:
                        new_hunkInfo.append((filename1, int(info_a[1:i]), 0))


                i = info_b.find(',')

                if i < 0:
                    new_hunkInfo.append((filename2, int(info_b[1:]), int(info_b[1:])))
                else:
                    if int(info_b[i + 1:]) > 0:
                        new_hunkInfo.append((filename2, int(info_b[1:i]), int(info_b[1:i]) + int(info_b[i + 1:]) - 1))
                    else:
                        new_hunkInfo.append((filename2, int(info_b[1:i]), 0))
            except ValueError:
                continue
    f.close()

    return new_hunkInfo


def locate_and_align(filesPath):
    funcInfo = []  # (filename, lineStart, lineEnd)

    if not os.path.exists(filesPath + 'cpg_a.txt') or not os.path.exists(filesPath + 'cpg_b.txt'):
        return []
    f = open(filesPath + 'cpg_a.txt')
    results = f.read()
    f.close()
    resultList = json.loads(results)

    for item in resultList:
        if '_1' in item.keys() and '_2' in item.keys() \
                and '_3' in item.keys() and '_4' in item.keys():
            i = item['_2'].rfind('/a/')
            if i == -1: i = 0
            funcInfo.append(('/a/' + item['_2'][i:], item['_3'], item['_4']))

    f = open(filesPath + 'cpg_b.txt')
    results = f.read()
    f.close()
    resultList = json.loads(results)

    for item in resultList:
        if '_1' in item.keys() and '_2' in item.keys() \
                and '_3' in item.keys() and '_4' in item.keys():
            i = item['_2'].rfind('/b/')
            if i == -1: i = 0
            funcInfo.append(('/b/' + item['_2'][i:], item['_3'], item['_4']))

    os.system('rm ' + filesPath + 'diff.txt')
    os.system('diff -brN -U 0 -p ' + filesPath + 'a/ ' + filesPath + 'b/ >> ' + filesPath + 'diff.txt')

    f = open(filesPath + 'diff.txt')

    hunkInfo = []  # (filename, lineStart, lineEnd)
    filename1 = ''
    filename2 = ''
    while 1:
        try:
            line = f.readline()
        except:
            return []
        if not line:    break
        # print(line)
        if line[:len('diff -brN')] == 'diff -brN':
            filename1 = line.split()[-2]
            filename2 = line.split()[-1]
            filename1 = filename1[filename1.find('/a/'):]
            filename2 = filename2[filename2.find('/b/'):]
        elif line.count('@') == 4:
            i = line.find('@@ ')
            j = line.find(' @@')
            info = line[i + 3:j]

            i = info.find(' ')
            info_a = info[:i]
            info_b = info[i + 1:]

            i = info_a.find(',')
            try:
                if i < 0:
                    hunkInfo.append((filename1, int(info_a[1:]), int(info_a[1:])))
                else:
                    if int(info_a[i + 1:]) > 0:
                        hunkInfo.append((filename1, int(info_a[1:i]), int(info_a[1:i]) + int(info_a[i + 1:]) - 1))
                    else:
                        hunkInfo.append((filename1, int(info_a[1:i]), 0))

                i = info_b.find(',')
                if i < 0:
                    hunkInfo.append((filename2, int(info_b[1:]), int(info_b[1:])))
                else:
                    if int(info_b[i + 1:]) > 0:
                        hunkInfo.append((filename2, int(info_b[1:i]), int(info_b[1:i]) + int(info_b[i + 1:]) - 1))
                    else:
                        hunkInfo.append((filename2, int(info_b[1:i]), 0))
            except ValueError:
                continue
    f.close()

    # print(hunkInfo)

    locate_flag = [0 for i in range(len(funcInfo))]
    align_flag = [0 for i in range(len(hunkInfo))]
    for i in range(len(funcInfo)):
        for j in range(len(hunkInfo)):
            if funcInfo[i][0] == hunkInfo[j][0]:
                if (hunkInfo[j][2] != 0 and max(funcInfo[i][1], hunkInfo[j][1]) <= min(funcInfo[i][2], hunkInfo[j][2])) \
                        or (
                        hunkInfo[j][2] == 0 and funcInfo[i][1] <= hunkInfo[j][1] and hunkInfo[j][1] <= funcInfo[i][2]):
                    locate_flag[i] = 1
                    align_flag[j] = 1

    filename = []
    fileContent = []

    files = os.listdir(filesPath + 'a/')
    for file in files:
        if file in ['.DS_store', '.DS_Store']:    continue
        f = open(filesPath + 'a/' + file)
        _f =  codecs.open(filesPath + 'a/' + file, 'r', encoding='utf-8', errors='ignore')
        filename.append('/a/' + file)

        lines = ['']
        while 1:
            try:
                line = f.readline()
            except UnicodeDecodeError:
                line = _f.readline()
            if not line:    break
            lines.append(line)
        _f.close()

        fileContent.append(lines)

    files = os.listdir(filesPath + 'b/')
    for file in files:
        if file in ['.DS_store', '.DS_Store']:    continue
        f = open(filesPath + 'b/' + file)
        _f = codecs.open(filesPath + 'b/' + file, 'r', encoding='utf-8', errors='ignore')
        filename.append('/b/' + file)

        lines = ['']
        while 1:
            try:
                line = f.readline()
            except UnicodeDecodeError:
                line = _f.readline()
            if not line:    break
            lines.append(line)
        f.close()
        _f.close()

        fileContent.append(lines)

    for i in range(len(locate_flag)): #change other functions to blank
        if locate_flag[i] == 0:
            try:
                idx = filename.index(funcInfo[i][0])
            except:
                return []
            for j in range(funcInfo[i][1], funcInfo[i][2] + 1):
                if j > len(fileContent[idx]) - 1:
                    return []
                fileContent[idx][j] = '\n'

    for i in range(0, len(hunkInfo), 2):
        #	if align_flag[i] == 1 or align_flag[i+1] == 1:
        #		print(hunkInfo[i], hunkInfo[i+1])
        try:
            if hunkInfo[i][2] == 0:
                if hunkInfo[i][0] not in filename:
                    filename.append(hunkInfo[i][0])
                    suffix = '\n' * (hunkInfo[i + 1][2] - hunkInfo[i + 1][1] + 1)
                    fileContent.append(suffix)
                else:
                    idx = filename.index(hunkInfo[i][0])
                    suffix = '\n' * (hunkInfo[i + 1][2] - hunkInfo[i + 1][1] + 1)
                    fileContent[idx][hunkInfo[i][1]] = fileContent[idx][hunkInfo[i][1]] + suffix
            elif hunkInfo[i + 1][2] == 0:
                if hunkInfo[i + 1][0] not in filename:
                    filename.append(hunkInfo[i + 1][0])
                    suffix = '\n' * (hunkInfo[i][2] - hunkInfo[i][1] + 1)
                    fileContent.append(suffix)
                else:
                    idx = filename.index(hunkInfo[i + 1][0])
                    idx_a = filename.index(hunkInfo[i][0])
                    #print(fileContent[idx_a][hunkInfo[i][2]])
                    if fileContent[idx_a][hunkInfo[i][2]].strip(' ') == '\n':
                        suffix = '\n' * (hunkInfo[i][2]-1 - hunkInfo[i][1] + 1)
                    else:
                        suffix = '\n' * (hunkInfo[i][2] - hunkInfo[i][1] + 1)
                    fileContent[idx][hunkInfo[i + 1][1]] = fileContent[idx][hunkInfo[i + 1][1]] + suffix
            else:
                gap = (hunkInfo[i][2] - hunkInfo[i][1]) - (hunkInfo[i + 1][2] - hunkInfo[i + 1][1])
                # print(gap)
                if gap < 0:
                    idx = filename.index(hunkInfo[i][0])
                    suffix = '\n' * (-gap)
                    fileContent[idx][hunkInfo[i][2]] = fileContent[idx][hunkInfo[i][2]] + suffix
                elif gap > 0:
                    idx = filename.index(hunkInfo[i + 1][0])
                    suffix = '\n' * (gap)
                    fileContent[idx][hunkInfo[i + 1][2]] = fileContent[idx][hunkInfo[i + 1][2]] + suffix
        except:
            continue

    for i in range(len(filename)):
        os.system('rm ' + filesPath[:-1] + filename[i])
        f = open(filesPath[:-1] + filename[i], 'a+')
        for line in fileContent[i]:
            f.write(line)
        f.close()

    new_hunkInfo = make_new_hunkinfo(filesPath=filesPath)


    return new_hunkInfo

def importCPG(path):
    nodesByFunc = {}
    edgesByFunc = {}
    files = os.listdir(path)
    for f in files: #dot files
        if not f.endswith('.dot'): continue
        content = open(path+'/'+f, errors='ignore').read()
        if '<SUB>' not in content:	continue
        lines = content[:-1].split(']\n ')
        digraph_content = lines[0].split(' {  \n')
        funcName = digraph_content[0][len('digraph')+2:-1]
        if funcName.startswith("&lt;"): continue
        #funcName = lines[0][len('digraph'):-1]
        if len(funcName) == 0:	continue

        nodescontent = digraph_content[1]
        nodes = nodescontent.split('\n')
        edges = lines[1].split('\n')
        node_pattern = r'"(\d+)"\s*\[label\s*=\s*<\((.*?)\)<SUB>(\d+)</SUB>>\s*\]'
        edge_pattern = r'"(\d+)"\s*->\s*"(\d+)"\s*\[\s*label\s*=\s*"(\w+):\s*(.*)"\s*\]'
        node_dict, edge_list = {}, []
        for n in nodes:
            match = re.search(node_pattern, n)
            if match:
                id = match.group(1)
                label = match.group(2)
                line_no = match.group(3)
            else:
                continue
            node_dict[id] = {"label": label, "line_no": line_no}

        for e in edges:
            e = e.strip()
            match = re.search(edge_pattern, e)

            if match:
                id_from = match.group(1)
                id_to = match.group(2)
                type_ = match.group(3)
                #extra = match.group(4)  # 这将捕获冒号后的任何内容，如果存在的话

                edge_list.append((id_from, id_to, type_))

        nodesByFunc[funcName] = node_dict
        edgesByFunc[funcName] = edge_list

    return nodesByFunc, edgesByFunc