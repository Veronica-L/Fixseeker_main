from tree_sitter import Language, Parser

# 注意C++对应cpp，C#对应c_sharp（！这里短横线变成了下划线）
# 看仓库名称
CPP_LANGUAGE = Language('treesitter/build/my-languages.so', 'cpp')
#CS_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
JAVA_LANGUAGE = Language('treesitter/build/my-languages.so', 'java')


# 遍历语法树，提取所有函数定义及其内部调用的函数
def get_function_calls(node):
    calls = []
    if node.type == 'call_expression':
        for child in node.children:
            if child.type == 'identifier':
                calls.append((child.text.decode('utf-8'), child.start_point[0]+1))
    for child in node.children:
        calls.extend(get_function_calls(child))
    return calls

# 遍历语法树，提取所有函数定义
def get_function_definitions(node, lines, function_list):
    if node.type == 'method_declaration':
        function_dict = {}
        function_name = None
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        function_dict = {}

        for child in node.children:
            if child.type == 'function_declarator':
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        function_name = grandchild.text.decode('utf-8')
                        break
        if function_name:
            calls = get_function_calls(node)

            function_dict['name'] = function_name
            function_dict['calls'] = calls
            function_dict['lines'] = {i + 1: lines[i].strip('\n') for i in range(start_line, end_line + 1)}
            '''
            print('Calls:')
            for call in calls:
                print(f'  {call}')
            print('Lines:')
            for line_number, line_code in function_dict['lines'].items():
                print(f'  {line_number}: {line_code}')
            '''

            function_list.append(function_dict)
    for child in node.children:
        get_function_definitions(child, lines, function_list)

    return function_list


#if __name__ == '__main__':
def parse(file_path):
    # 举一个CPP例子
    cpp_parser = Parser()
    cpp_parser.set_language(JAVA_LANGUAGE)

    # 读取 C 文件内容
    #file_path = 'core/test.c'
    with open(file_path, 'r') as file:
        code = file.read()

    # 没报错就是成功
    tree = cpp_parser.parse(bytes(code, "utf8"))
    # 注意，root_node 才是可遍历的树节点
    root_node = tree.root_node

    # 读取文件的行内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    function_list = get_function_definitions(root_node, lines, [])
    return function_list
