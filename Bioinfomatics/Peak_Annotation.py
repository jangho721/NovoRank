#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd

mgf = sys.argv[1]
txt = sys.argv[2]
ion_type = sys.argv[3]
tolerance = float(sys.argv[4])

# N-term / C-term modification 처리 확인하기

data = open(mgf)
data = data.readlines()

def list_split(data):
    L = []
    append = L.append
    m = 0
    for i in range(len(data)):
        if data[i] == 'END IONS\n':
            append(data[m:i+1])
            m = i+1
    return L

data_1 = list_split(data)

dic = {}
for i in data_1:
    key_scan = i[1].split(".")[1]
    if key_scan not in dic:
        dic[key_scan] = i

test_file = pd.read_csv(txt, delimiter = '\t')

# Should I get the PrecursorNM ?

H_2_O = 18.01057
NH_3 = 17.02655
proton = 1.00782
residue_mass = {"A": 71.03712, "R": 156.10112, "N": 114.04293, "D": 115.02695, "C": 103.00919, "E": 129.04260, "Q": 128.05858,
        "G": 57.02147, "H": 137.05891, "I": 113.08407, "L": 113.08407, "K": 128.09497, "M": 131.04049, "F": 147.06842,
        "P": 97.05277, "S": 87.03203, "T": 101.04768, "W": 186.07932, "Y": 163.06333, "V": 99.06842}

# b_ion = sum of residues mass + proton * charge(1,2) / charge(1,2)

def b_ion(amino_acid, strip_seq):
    icon = '+'
    count = 0
    text = []

    for charge in [1,2]:
        sum_residue = 0
        cnt = 1

        append = text.append

        for i in amino_acid:

            if cnt == len(strip_seq):
                break

            if i.isalpha() == True:
                sum_residue += residue_mass[i]
                b_ion = round((sum_residue + proton * charge ) / charge, 5)
                # neutral loss ion
                neutral_loss_HtoO = round((sum_residue - H_2_O + proton * charge ) / charge, 5)
                neutral_loss_NH_3 = round((sum_residue - NH_3 + proton * charge ) / charge, 5)

                append(['b'+str(cnt)+icon*charge, b_ion])
                # neutral loss ion
                append(['b'+str(cnt)+'-H2O'+icon*charge, neutral_loss_HtoO])
                append(['b'+str(cnt)+'-NH3'+icon*charge, neutral_loss_NH_3])

                count += 3
                cnt += 1

            else:
                cnt -= 1
                if cnt == 0: # N-term modification
                    sum_residue += float(i.split('+')[1])

                else:
                    if '-' in i:
                        sum_residue -= float(i.split('-')[1])
                        b_ion = round((sum_residue + proton * charge ) / charge, 5)
                        # neutral loss ion
                        neutral_loss_HtoO = round((sum_residue - H_2_O + proton * charge ) / charge, 5)
                        neutral_loss_NH_3 = round((sum_residue - NH_3 + proton * charge ) / charge, 5)

                        text[count-3] = ['b'+str(cnt)+icon*charge, b_ion]
                        # neutral loss ion
                        text[count-2] = ['b'+str(cnt)+'-H2O'+icon*charge, neutral_loss_HtoO]
                        text[count-1] = ['b'+str(cnt)+'-NH3'+icon*charge, neutral_loss_NH_3]

                    else:
                        sum_residue += float(i.split('+')[1])
                        b_ion = round((sum_residue + proton * charge ) / charge, 5)
                        # neutral loss ion
                        neutral_loss_HtoO = round((sum_residue - H_2_O + proton * charge ) / charge, 5)
                        neutral_loss_NH_3 = round((sum_residue - NH_3 + proton * charge ) / charge, 5)

                        text[count-3] = ['b'+str(cnt)+icon*charge, b_ion]
                        # neutral loss ion
                        text[count-2] = ['b'+str(cnt)+'-H2O'+icon*charge, neutral_loss_HtoO]
                        text[count-1] = ['b'+str(cnt)+'-NH3'+icon*charge, neutral_loss_NH_3]
                cnt += 1

    # text
    table = pd.DataFrame(text, columns = ['ion','m/z'])
    table = table.sort_values(by=['m/z'])
    
    return table

# y_ion = sum of residues mass + proton * charge(1,2) + water mass / charge(1,2)

def y_ion(amino_acid_reverse, strip_seq):
    icon = '+'
    text = []

    for charge in [1,2]:
        sum_residue = 0
        cnt = 1

        append = text.append

        for i in amino_acid_reverse:

            if cnt == len(strip_seq):
                break

            if i.isalpha() == True:
                sum_residue += residue_mass[i]
                y_ion = round((sum_residue + H_2_O + proton * charge ) / charge, 5)
                neutral_loss_HtoO = round((sum_residue + H_2_O - H_2_O + proton * charge ) / charge, 5)
                neutral_loss_NH_3 = round((sum_residue + H_2_O - NH_3 + proton * charge ) / charge, 5)

                append(['y'+str(cnt)+icon*charge, y_ion])
                # neutral loss ion
                append(['y'+str(cnt)+'-H2O'+icon*charge, neutral_loss_HtoO])
                append(['y'+str(cnt)+'-NH3'+icon*charge, neutral_loss_NH_3])
                cnt += 1

            else:
                if '-' in i:
                    sum_residue -= float(i.split('-')[1])
                else:
                    sum_residue += float(i.split('+')[1])
    
    # text
    table = pd.DataFrame(text, columns = ['ion','m/z'])
    table = table.sort_values(by=['m/z'])
    
    return table

def sequence(file_name, scan, c, pep):
    
    amino_acid = []
    temp = pep
    title = file_name+'.'+str(scan)+'.'+str(scan)+'.'+str(c)
    number =''
    cnt = 0

    for i in temp:
        if i.isalpha() == True:
            if cnt == 0:
                amino_acid.append(i)
            else:
                amino_acid.append(number)
                amino_acid.append(i)
                number =''
                cnt = 0
        else:
            number += i
            cnt = 1

    strip_seq = list(filter(str.isalpha,temp))
    amino_acid_reverse = list(reversed(amino_acid))
    
    return amino_acid, strip_seq, amino_acid_reverse, temp, title

test = test_file[['SpecFile', 'ScanNum','Charge','Peptide']].values
ion = ion_type
fragment_tolerance = tolerance # Da

if ion == 'b':
    T = open('Theoretical_bion.txt', 'w')
    A = open('Annotated_peak_bion.txt', 'w')
    
    for file_name, scan, c, pep in test:
        amino_acid, strip_seq, amino_acid_reverse, temp, title = sequence(file_name, scan, c, pep)
        table = b_ion(amino_acid, strip_seq)
        
        T.write('BEGIN\nPeptide= '+temp+'\n'+'TITLE= '+title+'\n')
        loop = table.values
        for i in loop:
            T.write(i[0]+'\t'+str(i[1])+'\n')
        T.write('END\n\n')
        
        A.write('BEGIN\nPeptide= '+temp+'\n'+'TITLE= '+title+'\n')
        experiment_peak = dic[str(scan)][5:len(dic[str(scan)])-1]
        for i in table.values: # loop
            for j in experiment_peak:
                if float(j.split()[0]) >= i[1]-fragment_tolerance and float(j.split()[0]) <= i[1]+fragment_tolerance:
                    A.write(i[0]+'\t'+j.split()[0]+'\t'+j.split()[1]+'\n')
        A.write('END\n\n')

    T.close()
    A.close()
    
elif ion == 'y':
    T = open('Theoretical_yion.txt', 'w')
    A = open('Annotated_peak_yion.txt', 'w')
    
    for file_name, scan, c, pep in test:
        amino_acid, strip_seq, amino_acid_reverse, temp, title = sequence(file_name, scan, c, pep)
        table = y_ion(amino_acid_reverse, strip_seq)
        
        T.write('BEGIN\nPeptide= '+temp+'\n'+'TITLE= '+title+'\n')
        loop = table.values
        for i in loop:
            T.write(i[0]+'\t'+str(i[1])+'\n')
        T.write('END\n\n')
        
        A.write('BEGIN\nPeptide= '+temp+'\n'+'TITLE= '+title+'\n')
        experiment_peak = dic[str(scan)][5:len(dic[str(scan)])-1]
        for i in table.values: # loop
            for j in experiment_peak:
                if float(j.split()[0]) >= i[1]-fragment_tolerance and float(j.split()[0]) <= i[1]+fragment_tolerance:
                    A.write(i[0]+'\t'+j.split()[0]+'\t'+j.split()[1]+'\n')
        A.write('END\n\n')

    T.close()
    A.close()

else:
    print('Only use b-ion and y-ion')
    print('Type in lowercase')