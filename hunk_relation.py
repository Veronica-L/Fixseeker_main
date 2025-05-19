import os.path
import git
from git import Repo
import subprocess
import shutil
import json
from gen_cpg import gencpg, locate_and_align, JOERN_PATH, JOERN_PARSER_PATH, JOERN_EXPORT_PATH, importCPG
import networkx as nx
import Levenshtein as lev
from utils import normalize_code, abstract_code

def commit_checkout(commit, target_directory):
    os.chdir(target_directory)
    subprocess.run(["git", "checkout",commit])
    os.chdir("../../")

def compute_similarity(hunk1, hunk2):
    normalized_code1 = normalize_code(hunk1)
    abstracted_code1 = abstract_code(hunk2)

    normalized_code2 = normalize_code(hunk1)
    abstracted_code2 = abstract_code(hunk2)

    similarity = lev.ratio(abstracted_code1, abstracted_code2)
    return similarity

def extract_lines(path, filename, start_no, end_no):
    extracted_lines = ''
    try:

        if filename.startswith('/a/'):
            mark = '-'
        else:
            mark = '+'

        with open(path, 'r', encoding='utf-8') as file:
            all_lines = file.readlines()

            start_no = max(1, start_no)
            end_no = min(len(all_lines), end_no)

            for i in range(start_no - 1, end_no):
                if all_lines[i].strip(' ') == '\n': continue
                extracted_lines += mark + ' '+ all_lines[i]
            #extracted_lines = all_lines[start_no - 1:end_no]

            return extracted_lines
    except FileNotFoundError:
        print(f"File '{path}' cannot be found.")
        return []
    except IOError:
        print(f"Read file '{path}' error。")
        return []

def merger_hunks(hunks_list, merge_file):
    #Now only merge the delete file
    merge_lineno = 0
    merge_before_after_lineno_map = list() #[(before_lineno, after_lilneno)]
    merge_content = ''
    for hunks in hunks_list:
        for code_tp in hunks:
            line_number, code = code_tp[0], code_tp[1]
            if code.startswith('-'):
                code = code.strip('-')
                merge_content += code + '\n'
                merge_lineno += 1
                merge_before_after_lineno_map.append((line_number, merge_lineno))

    with open(merge_file, 'w') as output_f:
        output_f.write(merge_content)

    return merge_before_after_lineno_map


def if_source_file(filename):
    filebasename = os.path.basename(filename)
    if filebasename.endswith('.c') or filebasename.endswith('.cpp') or filebasename.endswith('.java') or filebasename.endswith('.py'):
        return True
    return False

def get_ab_file(owner, repo, commit_ID):
    if not os.path.exists('./diff_files/' + commit_ID + ".diff"):
        diff_url = f"https://github.com/{owner}/{repo}/commit/{commit_ID}.diff"
        try:
            subprocess.run(["wget", "-P", './diff_files/', diff_url])
        except:
            return False
    if not os.path.exists('./diff_files/' + commit_ID + ".diff"):
        return False
    f = open('./diff_files/' + commit_ID + ".diff")
    try:
        content = f.read()
    except UnicodeDecodeError:
        return False
    f.close()

    if not os.path.exists('./ab_file/' + commit_ID):
        os.system('mkdir ./ab_file/' + commit_ID)

    i = content.find("diff --git")
    new_content = content[:i]
    content = content[i:]

    a = []
    b = []

    # deal with the code difference one by one
    while 'diff --git ' in content:

        i = content.find(' a/')
        j = content.find(' b/')
        k = content.find('\n')

        file_a = content[i + 3:]
        i = file_a.find(' ')
        file_a = file_a[:i]
        file_b = content[j + 3:k]

        if not os.path.exists('./ab_file/' + commit_ID + '/a'):
            os.system('mkdir ./ab_file/' + commit_ID + '/a')
        if not os.path.exists('./ab_file/' + commit_ID + '/b'):
            os.system('mkdir ./ab_file/' + commit_ID + '/b')

        # retrive and download the pre- and post-patch files
        if file_a not in a:
            a.append(file_a)
        if file_b not in b:
            b.append(file_b)

        i = content.find('\ndiff --git ')
        if i > 0:
            content = content[i + 1:]
        else:
            if not os.path.exists('./source/' + repo):
                os.system('cd ./source; git clone https://github.com/' + owner + '/' + repo + '.git')

            # post-patch files (version b)
            os.system('cd ./source/' + repo + '; git reset --hard ' + commit_ID)
            for f_b in b:
                if 'test' in f_b or 'Test' in f_b or not if_source_file(f_b):
                    continue
                os.system('cp ./source/' + repo + '/' + f_b + ' ./ab_file/' + commit_ID + '/b/')

            # pre-patch files (version a)
            out = os.popen('cd ./source/' + repo + '; git rev-list --parents -n 1 ' + commit_ID).read()
            commit_a = out[out.find(' ') + 1:].rstrip()
            os.system('cd ./source/' + repo + '; git reset --hard ' + commit_a)
            for f_a in a:
                if 'test' in f_a or 'Test' in f_a or not if_source_file(f_a):
                    continue
                os.system('cp ./source/' + repo + '/' + f_a + ' ./ab_file/' + commit_ID + '/a/')

            if len(os.listdir(f'./ab_file/{commit_ID}/a')) == 0 or len(os.listdir(f'./ab_file/{commit_ID}/b')) == 0:
                return False
            else:
                return True
            break

def get_all_edge_types(graph, node1, node2):
    if graph.has_edge(node1, node2):
        return [graph[node1][node2][i]['type'] for i in graph[node1][node2]]
    return []

def are_lines_reachable(G, line_numbers):
    path = []
    line_to_nodes = {}
    for node, data in G.nodes(data=True):
        if 'line_no' not in data: continue
        line_no = data['line_no']
        if line_no not in line_to_nodes:
            line_to_nodes[line_no] = []
        line_to_nodes[line_no].append(node)

    for i in range(len(line_numbers)):
        for j in range(i + 1, len(line_numbers)):
            line1, line2 = line_numbers[i], line_numbers[j]
            if line1 not in line_to_nodes or line2 not in line_to_nodes:
                continue
            for node1 in line_to_nodes[line1]:
                for node2 in line_to_nodes[line2]:
                    if nx.has_path(G, node1, node2) or nx.has_path(G, node2, node1):
                        types = get_all_edge_types(G, node1, node2)

                        for type in types:
                            if (line1, line2, type) not in path:
                                path.append((line1, line2, type))

    return path

def call_realtion(storage_files_dir, hunkInfo, a_or_b):
    try:
        with open(storage_files_dir+f'/call_{a_or_b}.json', 'r', encoding='utf-8') as f:
            call_data = json.load(f)
    except FileNotFoundError:
        return []
    method_tp_list = list()
    ids = [m["id"] for m in call_data["methodList"]]
    for method in call_data["methodList"]:
        if method["signature"] == "": continue
        id, filename, funcname, lineNumber, lineNumberEnd, caller, callee = method["id"], method["filename"], method["name"], method["lineNumber"], method["lineNumberEnd"], method["caller"], method["callee"]
        for caller_id in caller:
            if caller_id in ids and caller_id != id and (caller_id, id) not in method_tp_list:
                method_tp_list.append((caller_id, id))
        for callee_id in callee:
            if callee_id in ids and callee_id != id and (id, callee_id) not in method_tp_list:
                method_tp_list.append((id, callee_id))

    hunk_call_list = list()
    for tp in method_tp_list:
        id_from, id_to = tp[0], tp[1]
        calls = [[c for c in call_data["methodList"] if c["id"] == id_from][0], [c for c in call_data["methodList"] if c["id"] == id_to][0]]
        #callee = [c for c in call_data["methodList"] if c["id"] == id_to][0]
        method = calls[0]
        method_callee = calls[1]
        filename, funcname, lineNumber, lineNumberEnd = method["filename"], method[
        "name"], method["lineNumber"], method["lineNumberEnd"]
        # identify which hunk call
        caller_hunks, callee_hunks = [], []
        for index, hunk_tp in enumerate(hunkInfo):
            hunk_filename = os.path.basename(hunk_tp[0])
            #caller
            if hunk_tp[2] == 0: continue
            if not hunk_tp[0].startswith(f'/{a_or_b}/'):
                continue

            hunk = hunk_tp
            if (filename == hunk_filename and lineNumber <= hunk[1] and lineNumberEnd >= hunk[2]) or \
                (filename == hunk_filename and lineNumber >= hunk[1] and lineNumberEnd <= hunk[2]):
                with open(storage_files_dir+hunk_tp[0], 'r') as f:
                    lines = f.readlines()
                for lineno, code in enumerate(lines):
                    if lineno+1 in range(hunk[1], hunk[2]+1) and method_callee["name"] in code:
                        if index not in caller_hunks:
                            caller_hunks.append(index)

                            for index2, hunk_tp2 in enumerate(hunkInfo):
                                #find callee
                                hunk_filename2 = os.path.basename(hunk_tp2[0])
                                if hunk_tp2[2] == 0: continue
                                if not hunk_tp2[0].startswith(f'/{a_or_b}/'):
                                    continue
                                hunk2 = hunk_tp2
                                if method_callee["filename"] == hunk_filename2 and method_callee["lineNumber"] <= hunk2[1] and method_callee["lineNumberEnd"] >= hunk2[2]:
                                    if index2 not in callee_hunks:
                                        callee_hunks.append(index2)
                                elif method_callee["filename"] == hunk_filename2 and method_callee["lineNumber"] >= hunk2[1] and method_callee["lineNumberEnd"] <= hunk2[2]:
                                    if index2 not in callee_hunks:
                                        callee_hunks.append(index2)
        if len(caller_hunks) > 0 and len(callee_hunks) > 0:
            hunk_call_list.append({"caller_hunks": caller_hunks, "callee_hunks": callee_hunks})

    return hunk_call_list

def generateGraph(storage_files_dir, hunkInfo):
    AnodesByFunc, AedgesByFunc = importCPG(storage_files_dir + '/outA/')
    BnodesByFunc, BedgesByFunc = importCPG(storage_files_dir + '/outB/')

    #internal hunk relation
    paths = {}
    for funcname in AnodesByFunc:
        nodes = AnodesByFunc[funcname]
        edges = AedgesByFunc[funcname]
        G = nx.MultiDiGraph()
        lineno_list = list()
        for tp in hunkInfo: #get hunk lines
            if tp[0].startswith("/b/"):
                for idx in range(tp[1], tp[2] +1):
                    lineno_list.append(idx)
        for node in nodes.keys():
            G.add_node(int(node), line_no=int(nodes[node]['line_no']))
        for edge in edges:
            G.add_edge(int(edge[0]), int(edge[1]), type=edge[2])
        path = are_lines_reachable(G, lineno_list) #(node1, node2, type)
        if len(path) > 0:
            paths["a"] = path


    for funcname in BnodesByFunc:
        nodes = BnodesByFunc[funcname]
        edges = BedgesByFunc[funcname]
        G = nx.MultiDiGraph()
        lineno_list = list()
        for tp in hunkInfo: #get hunk lines
            if tp[0].startswith("/b/"):
                for idx in range(tp[1], tp[2] +1):
                    lineno_list.append(idx)
        for node in nodes.keys():
            G.add_node(int(node), line_no=int(nodes[node]['line_no']))
        for edge in edges:
            G.add_edge(int(edge[0]), int(edge[1]), type=edge[2])
        path = are_lines_reachable(G, lineno_list)
        if len(path) > 0:
            paths["b"] = path


    return paths, AnodesByFunc, BnodesByFunc

def external_relation(AnodesByFunc, BnodesByFunc, storage_files_dir, hunkInfo):
    # external hunk relation: call relation
    Afuncs = AnodesByFunc.keys()
    AFuncStr = ''
    for func in Afuncs:
        AFuncStr += func + ','
    AFuncStr.strip(',')

    Bfuncs = BnodesByFunc.keys()
    BFuncStr = ''
    for func in Bfuncs:
        BFuncStr += func + ','
    BFuncStr.strip(',')

    if not os.path.exists(storage_files_dir + '/call_a.json'):
        cmd = 'cd ./joern; ' + JOERN_PATH + ' --script ../CallFunc.sc --param cpgFile=../' + storage_files_dir + '/a/ --param  FuncStr=' + AFuncStr + ' --param outpath=/storage/yirancheng/Hunk/' + storage_files_dir + '/call_a.json'
        os.system(cmd)
    if not os.path.exists(storage_files_dir + '/call_b.json'):
        cmd = 'cd ./joern; ' + JOERN_PATH + ' --script ../CallFunc.sc --param cpgFile=../' + storage_files_dir + '/b/ --param  FuncStr=' + BFuncStr + ' --param outpath=/storage/yirancheng/Hunk/' + storage_files_dir + '/call_b.json'
        os.system(cmd)
    hunk_call_list_a = call_realtion(storage_files_dir, hunkInfo, 'a')
    hunk_call_list_b = call_realtion(storage_files_dir, hunkInfo, 'b')
    print("A hunk call:")
    for hunk_call_a in hunk_call_list_a:
        print("Group:")
        caller_hunks, callee_hunks = hunk_call_a["caller_hunks"], hunk_call_a["callee_hunks"]
        for hunk_index in caller_hunks:
            print(hunkInfo[hunk_index])
        for hunk_index in callee_hunks:
            print(hunkInfo[hunk_index])
    print("B hunk call:")
    for hunk_call_b in hunk_call_list_b:
        print("Group:")
        caller_hunks, callee_hunks = hunk_call_b["caller_hunks"], hunk_call_b["callee_hunks"]
        for hunk_index in caller_hunks:
            print(hunkInfo[hunk_index])
        for hunk_index in callee_hunks:
            print(hunkInfo[hunk_index])
    #print(hunk_call_list_a, hunk_call_list_b)
    return hunk_call_list_a, hunk_call_list_b

def similar_relation(hunkInfo):
    return

def line_hunk_map(hunkInfo, paths, a_or_b):
    a_flow_tps = []
    if len(paths) > 0:
        if a_or_b in paths.keys():
            for flow_tp in paths[a_or_b]:
                lineno_from, lineno_to, type = flow_tp[0], flow_tp[1], flow_tp[2]
                index_from = index_to = 0
                for index, hunk in enumerate(hunkInfo):
                    if hunk[0].startswith(f'/{a_or_b}/') and lineno_from >= hunk[1] and lineno_from <= hunk[2]:
                        index_from = index
                    if hunk[0].startswith(f'/{a_or_b}/') and lineno_to >= hunk[1] and lineno_to <= hunk[2]:
                        index_to = index
                if index_from == index_to: continue
                if (index_from, index_to, type) not in a_flow_tps:
                    a_flow_tps.append((index_from, index_to, type))
    return a_flow_tps

def extract_relation(owner, repo, hash):
    #for commit in url_to_hunk.keys():
        commit_dict = {}
        #owner, repo, hash = commit.split('/')[0], commit.split('/')[1], commit.split('/')[3]

        repo_url = f"https://github.com/{owner}/{repo}.git"
        repo_directory = f"./source/{repo}"

        #file_dict = url_to_hunk[commit]
        #if not os.path.exists(repo_directory):
            #git.Repo.clone_from(repo_url, repo_directory, file_dict.keys())

        if not get_ab_file(owner, repo, hash):
            print('wrong commit')
            return [], []
        storage_files_dir = f'./ab_file/{hash}'
        gencpg(storage_files_dir)
        hunkInfo = locate_and_align(storage_files_dir+'/')
        if hunkInfo == []:
            return [], []
        if os.path.isfile(storage_files_dir + '/cpg_a.txt') and os.path.isfile(storage_files_dir + '/cpg_b.txt'):
            os.system('cd ./joern; '+JOERN_PARSER_PATH+' .'+storage_files_dir+'/a; '+JOERN_EXPORT_PATH+' --repr cpg14 --out .'+storage_files_dir+'/outA')
            os.system('cd ./joern; '+JOERN_PARSER_PATH+' .'+storage_files_dir+'/b; '+JOERN_EXPORT_PATH+' --repr cpg14 --out .'+storage_files_dir+'/outB')
            if not os.path.exists(storage_files_dir + '/outA') or not os.path.exists(storage_files_dir + '/outB'):
                return [], []
            lenA = os.listdir(storage_files_dir + '/outA')
            lenB = os.listdir(storage_files_dir + '/outB')
            if len(lenA) + len(lenB) > 0:
                paths, AnodesByFunc, BnodesByFunc = generateGraph(storage_files_dir, hunkInfo)
                hunk_call_list_a, hunk_call_list_b = external_relation(AnodesByFunc, BnodesByFunc, storage_files_dir, hunkInfo)
                a_flow_tps = line_hunk_map(hunkInfo, paths, 'a')
                b_flow_tps = line_hunk_map(hunkInfo, paths, 'b')
                print(a_flow_tps, b_flow_tps)

                hunks, relations = [], []
                for i in range(0, len(hunkInfo), 2):
                    deleted_hunk = hunkInfo[i]
                    added_hunk = hunkInfo[i+1]
                    a_filename, a_start_no, a_end_no = deleted_hunk[0], deleted_hunk[1], deleted_hunk[2]
                    b_filename, b_start_no, b_end_no = added_hunk[0], added_hunk[1], added_hunk[2]
                    a_hunk_content = extract_lines(f"{storage_files_dir}{a_filename}", a_filename, start_no=a_start_no, end_no=a_end_no)
                    b_hunk_content = extract_lines(f"{storage_files_dir}{b_filename}", b_filename, start_no=b_start_no, end_no=b_end_no)
                    hunk_content = a_hunk_content + b_hunk_content
                    hunks.append(hunk_content)

                for hunk_call_a in hunk_call_list_a:
                    caller_hunks, callee_hunks = hunk_call_a["caller_hunks"], hunk_call_a["callee_hunks"]
                    for hunk_index in caller_hunks:
                        for hunk_index2 in callee_hunks:
                            if (int(hunk_index/2), int(hunk_index2/2), 'CALL') not in relations:
                                relations.append((int(hunk_index/2), int(hunk_index2/2), 'CALL'))
                for hunk_call_b in hunk_call_list_b:
                    caller_hunks, callee_hunks = hunk_call_b["caller_hunks"], hunk_call_b["callee_hunks"]
                    for hunk_index in caller_hunks:
                        for hunk_index2 in callee_hunks:
                            if (int(hunk_index/2), int(hunk_index2/2), 'CALL') not in relations:
                                relations.append((int(hunk_index/2), int(hunk_index2/2), 'CALL'))

                for flow_tp in a_flow_tps:
                    if flow_tp[2] == "CFG" or flow_tp[2] == "DDG":
                        if (int(flow_tp[0]/2), int(flow_tp[1]/2), flow_tp[2]) not in relations and (int(flow_tp[1]/2), int(flow_tp[0]/2), flow_tp[2]) not in relations:
                            if flow_tp[0] < flow_tp[1]:
                                relations.append((int(flow_tp[0]/2), int(flow_tp[1]/2), flow_tp[2]))
                            else:
                                relations.append((int(flow_tp[1]/2), int(flow_tp[0]/2), flow_tp[2]))

                for flow_tp in b_flow_tps:
                    if flow_tp[2] == "CFG" or flow_tp[2] == "DDG":
                        if (int(flow_tp[0]/2), int(flow_tp[1]/2), flow_tp[2]) not in relations and (int(flow_tp[1]/2), int(flow_tp[0]/2), flow_tp[2]) not in relations:
                            if flow_tp[0] < flow_tp[1]:
                                relations.append((int(flow_tp[0]/2), int(flow_tp[1]/2), flow_tp[2]))
                            else:
                                relations.append((int(flow_tp[1]/2), int(flow_tp[0]/2), flow_tp[2]))

                for i in range(len(hunks)):
                    for j in range(i+1, len(hunks)):
                        if compute_similarity(hunks[i], hunks[j]) >= 0.8 and i!=j:
                            relations.append(i, j, 'SIM')
                            relations.append(j, i, 'SIM')

                return hunks, relations

        return [], []



