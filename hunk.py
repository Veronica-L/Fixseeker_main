import re

def retrieve_commit_content(diff):
    header_pattern = "@@ -{},{} \+{},{} @@.*\n"
    integer_pattern = "(?P<{}>[0-9]+)"
    header_pattern = header_pattern.format(integer_pattern.format("a_start"), integer_pattern.format('a_lines'),
                              integer_pattern.format("b_start"), integer_pattern.format("b_lines"))
    if re.finditer(header_pattern, diff):
        matched_segments = list(re.finditer(header_pattern, diff))
    ll = len(matched_segments)

    hunks_list = []
    for idx in range(ll):
        seg = matched_segments[idx]
        matched_dict = seg.groupdict()
        a_start, a_lines, b_start, b_lines = matched_dict['a_start'], matched_dict['a_lines'], matched_dict['b_start'], matched_dict['b_lines']
        diff_lines = (int(b_start) + int(b_lines)) - (int(a_start) + int(a_lines))
        if idx == ll - 1:
            code_end_pos = len(diff)
        else:
            code_end_pos = matched_segments[idx + 1].span()[0]
        code_start_pos = seg.span()[1]
        source_code = diff[code_start_pos:code_end_pos]

        vul_index, patch_index = 0, 0


        internal_hunk_list = []
        for line in source_code.split('\n'):
            if line.startswith(('+', '-')):
                #hunk = hunk + line + '\n'
                if line.startswith('-'):
                    hunk_tp = (int(a_start) + vul_index, line.strip())
                    internal_hunk_list.append(hunk_tp)
                    vul_index += 1
                elif line.startswith('+'):
                    hunk_tp = (int(b_start) + patch_index, line.strip())
                    internal_hunk_list.append(hunk_tp)
                    patch_index += 1
            else:
                if len(internal_hunk_list) > 0: # finish a hunk
                    #hunk = hunk[:-1]
                    hunks_list.append(internal_hunk_list)
                    internal_hunk_list = []

                vul_index += 1
                patch_index += 1

    return hunks_list

def hunk_empty(hunk):
    if hunk.strip() == '':
        return True

    for line in hunk.split('\n'):
        if line[1:].strip() != '':
            return False

    return True

def get_hunk_from_diff(diff):
    hunk_list = []
    hunk = ''
    for line in diff.split('\n'):
        if line.startswith(('+', '-')):
            hunk = hunk + line + '\n'
        else:
            if not hunk_empty(hunk):    # finish a hunk
                hunk = hunk[:-1]
                hunk_list.append(hunk)
                hunk = ''

    if not hunk_empty(hunk):
        hunk_list.append(hunk)

    return hunk_list
