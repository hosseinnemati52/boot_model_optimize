
import numpy as np
import time
import os
import re

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

def parse_array(value):
    # Remove surrounding brackets and extra spaces
    value = value.strip('[]')
    # Handle multi-line arrays (convert ';' to '],[')
    if ';' in value:
        value = value.replace(';', ',')
    # Add surrounding brackets to make it a proper list format
    value = f'[{value}]'
    
    try:
        # Convert to a NumPy array
        return np.array(eval(value))
    except (SyntaxError, NameError) as e:
        print(f"Error parsing array: {e}")
        return None

def read_custom_csv(filename):
    # Initialize dictionary to hold variables
    variables = {}
    
    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            # Skip comments or empty lines
            if line.strip() == '' or line.strip().startswith('##'):
                continue
            
            # Split the line into key and value
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Process based on the type of the value
            if value.startswith('[') and value.endswith(']'):
                # This is a list or array
                variables[key] = parse_array(value)
                
            elif re.match(r'^[\d.]+$', value):
                # This is a number (int or float)
                variables[key] = float(value) if '.' in value else int(value)
            
            elif re.match(r'^[\w]+$', value):
                # This is a string or keyword
                variables[key] = value
            
    # Extract variables from the dictionary
    N_UpperLim = variables.get('N_UpperLim', None)
    NTypes = variables.get('NTypes', None)
    typeR0 = variables.get('typeR0', None)
    typeR2PI = variables.get('typeR2PI', None)
    typeTypeEpsilon = variables.get('typeTypeEpsilon', None)
    
    typeGamma = variables.get('typeGamma', None)
    typeTypeGammaCC = variables.get('typeTypeGammaCC', None)
    typeTypeF_rep_max = variables.get('typeTypeF_rep_max', None)
    typeTypeF_abs_max = variables.get('typeTypeF_abs_max', None)
    R_eq_coef = variables.get('R_eq_coef', None)
    R_cut_coef_force = variables.get('R_cut_coef_force', None)
    
    typeFm = variables.get('typeFm', None)
    typeDr = variables.get('typeDr', None)
    
    G1Border = variables.get('G1Border', None)
    
    typeOmega = variables.get('typeOmega', None)
    typeBarrierW = variables.get('typeBarrierW', None)
    typeSigmaPhi = variables.get('typeSigmaPhi', None)
    typeSigmaFit = variables.get('typeSigmaFit', None)
    barrierPeakCoef = variables.get('barrierPeakCoef', None)
    typeFit0 = variables.get('typeFit0', None)
    Fit_Th_Wall = variables.get('Fit_Th_Wall', None)
    Fit_Th_G0 = variables.get('Fit_Th_G0', None)
    Fit_Th_Diff = variables.get('Fit_Th_Diff', None)
    Fit_Th_Apop = variables.get('Fit_Th_Apop', None)
    
    maxTime = variables.get('maxTime', None)
    dt = variables.get('dt', None)
    dt_sample = variables.get('dt_sample', None)
    samplesPerWrite = variables.get('samplesPerWrite', None)
    printingTimeInterval = variables.get('printingTimeInterval', None)
    
    R_cut_coef_game = variables.get('R_cut_coef_game', None)
    tau = variables.get('tau', None)
    typeTypePayOff_mat_real_C = variables.get('typeTypePayOff_mat_real_C', None)
    typeTypePayOff_mat_real_F1 = variables.get('typeTypePayOff_mat_real_F1', None)
    typeTypePayOff_mat_real_F2 = variables.get('typeTypePayOff_mat_real_F2', None)
    typeTypePayOff_mat_imag_C = variables.get('typeTypePayOff_mat_imag_C', None)
    typeTypePayOff_mat_imag_F1 = variables.get('typeTypePayOff_mat_imag_F1', None)
    typeTypePayOff_mat_imag_F2 = variables.get('typeTypePayOff_mat_imag_F2', None)
    
    # typeOmega0 = variables.get('typeOmega0', None)
    # typeOmegaLim = variables.get('typeOmegaLim', None)
    newBornFitKey = variables.get('newBornFitKey', None)
    
    shrinkageRate = variables.get('shrinkageRate', None)
    
    initConfig = variables.get('initConfig', None)
    
    return {
        'N_UpperLim': N_UpperLim,
        'NTypes': NTypes,
        'typeR0': typeR0,
        'typeR2PI': typeR2PI,
        'typeTypeEpsilon': typeTypeEpsilon,
        
        'typeGamma': typeGamma,
        'typeTypeGammaCC': typeTypeGammaCC,
        'typeTypeF_rep_max': typeTypeF_rep_max,
        'typeTypeF_abs_max': typeTypeF_abs_max,
        'R_eq_coef': R_eq_coef,
        'R_cut_coef_force': R_cut_coef_force,
        
        'typeFm': typeFm,
        'typeDr': typeDr,
        
        'G1Border' : G1Border,
        
        'typeOmega' : typeOmega,
        'typeBarrierW' : typeBarrierW,
        'typeSigmaPhi' : typeSigmaPhi,
        'typeSigmaFit' : typeSigmaFit,
        'barrierPeakCoef' : barrierPeakCoef,
        'typeFit0' : typeFit0,
        'Fit_Th_Wall' : Fit_Th_Wall,
        'Fit_Th_G0' : Fit_Th_G0,
        'Fit_Th_Diff' : Fit_Th_Diff,
        'Fit_Th_Apop' : Fit_Th_Apop,
        
        'maxTime': maxTime,
        'dt': dt,
        'dt_sample': dt_sample,
        'samplesPerWrite': samplesPerWrite,
        'printingTimeInterval': printingTimeInterval,
                
        'R_cut_coef_game' : R_cut_coef_game,
        'tau' :tau,
        'typeTypePayOff_mat_real_C' :typeTypePayOff_mat_real_C,
        'typeTypePayOff_mat_real_F1' :typeTypePayOff_mat_real_F1,
        'typeTypePayOff_mat_real_F2' :typeTypePayOff_mat_real_F2,
        'typeTypePayOff_mat_imag_C' :typeTypePayOff_mat_imag_C,
        'typeTypePayOff_mat_imag_F1' :typeTypePayOff_mat_imag_F1,
        'typeTypePayOff_mat_imag_F2' :typeTypePayOff_mat_imag_F2,
        
        'newBornFitKey' : newBornFitKey,
        
        'shrinkageRate' : shrinkageRate,
        
        'initConfig': initConfig
    }

with open("compos.txt", "r") as compos:
    compos_key = compos.readline()[:-1]
    compos.close()

if compos_key == 'WT':
    
    if os.path.exists("t_sufficiency.txt"):
        
        filename = "t_sufficiency.txt"
        with open(filename, "r") as file:
            string = file.readline()
        file.close()
        
        if string[0:len("fail")] == "fail":
            
            params = read_custom_csv("source/params.csv")
            
            modif_task = dict()
            modif_task['nondim_quantity'] = 't_tilde_max'
            modif_task['line_word'] = 'maxTime'
            modif_task['update_switch'] = 1
            modif_task['begin_char'] = '='
            modif_task['end_char'] = '\n'
            modif_task['factor'] = +1
            
            modify_func(modif_task, "source/params.csv" , 1.2*params['maxTime']+ 0.0000001)
        

    filename = "t_sufficiency.txt"
    with open(filename, "w") as file:
        file.write("null")
    file.close()
    
    
if compos_key != 'WT':
    
    while 1:
        try:
            tau_file = np.loadtxt("../WT/tau.txt", delimiter=',')
            t_max_exp = np.loadtxt("../WT/t_max_exp.txt", delimiter=',')[0]
#            t_tilde_eval_file = np.loadtxt("../WT/t_tilde_eval.txt", delimiter=',')
        except FileNotFoundError:
            time.sleep(60)
            # print("ding")
            continue
        break
    
    tau_avg = tau_file[0]
    tau_err = tau_file[1]
    
    t_tilde_max = t_max_exp/tau_avg
    t_tilde_max = t_tilde_max - (t_tilde_max%2) + 2 + 0.0000001
    
    modif_task = dict()
    modif_task['nondim_quantity'] = 't_tilde_max'
    modif_task['line_word'] = 'maxTime'
    modif_task['update_switch'] = 1
    modif_task['begin_char'] = '='
    modif_task['end_char'] = '\n'
    modif_task['factor'] = +1
    
    modify_func(modif_task, "source/params.csv" , t_tilde_max)
    
