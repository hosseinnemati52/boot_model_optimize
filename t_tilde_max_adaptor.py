
import numpy as np
import time

def indices_finder(string, begin_char, end_char):
    
    start_index = -1 + len(begin_char)
    
    while 1:
        start_index += 1
        
        if string[start_index-len(begin_char):start_index] == begin_char:
            
            end_index = string.index(end_char, start_index)
            
            try:
                value = float(string[start_index:end_index])
                break
            except:
                continue
        
    
    return start_index, end_index

def modify_func(task_info, file_path, new_val):
    
    modified_lines = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line[:len(task_info['line_word'])] == task_info['line_word']:
                
                # start_index = line.index(task_info['begin_char']) + len(task_info['begin_char'])
                # end_index = line.index(task_info['end_char'])
                
                start_index, end_index = indices_finder(line, task_info['begin_char'], task_info['end_char'])
                
                modified_line = line[:start_index]+str(task_info['factor']*new_val)+line[end_index:]
                modified_lines.append(modified_line)
                
            else:
                modified_lines.append(line)
    
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)
       
    return 1

with open("compos.txt", "r") as compos:
    compos_key = compos.readline()[:-1]
    compos.close()

if compos_key != 'WT':
    
    while 1:
        try:
            t_tilde_eval_file = np.loadtxt("../WT/t_tilde_eval.txt", delimiter=',')
        except FileNotFoundError:
            time.sleep(60)
            # print("ding")
            continue
        break
    
    t_tilde_eval_avg = t_tilde_eval_file[0]
    t_tilde_eval_err = t_tilde_eval_file[1]
    
    t_tilde_max = t_tilde_eval_avg - (t_tilde_eval_avg%2) + 2
    
    modif_task = dict()
    modif_task['nondim_quantity'] = 't_tilde_max'
    modif_task['line_word'] = 'maxTime'
    modif_task['update_switch'] = 1
    modif_task['begin_char'] = '='
    modif_task['end_char'] = '\n'
    modif_task['factor'] = +1
    
    modify_func(modif_task, "source/params.csv" , t_tilde_max)
    
