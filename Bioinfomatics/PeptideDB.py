# Making DB
# 펩타이드 최소 길이 : 6
# */* 도 만들기
# 중복제거
# cond_fully = fully_DB 주소만 복사, 진짜 복사는 [:] 또는 copy 함수 사용

def write(L):
    global count
    f.write(str(count)+'\t'+str(L[0])+'\t'+str(L[1])+'\t'+str(L[2])+'\t'+str(L[3])+'\t'+str(L[4])+'\t'+str(L[5])+ '\n')
    count += 1

# U, O : Proteinogenic AA
# B(Asx)는 N(Asn) or D(Asp)
# Z(Glx)는 E(Glu) or Q(Gln)
# X Xaa

def cal_mass(sequence):
    mass = {"A": 71.03711378804, "R": 156.10111102874, "N": 114.04292744752, "D": 115.02694303224, "C": 103.00918447804, "E": 129.04259309652, "Q": 128.0585775118,
            "G": 57.02146372376, "H": 137.0589118628, "I": 113.08406398088, "L": 113.08406398088, "K": 128.09496301826, "M": 131.0404846066, "F": 147.0684139166,
            "P": 97.05276385232, "S": 87.03202841014, "T": 101.04767847442, "W": 186.07931295398, "Y": 163.0633285387, "V": 99.0684139166, "U": 150.95364, "O": 237.14773,
            'B': 114.04292744752, 'Z': 129.04259309652}
    temp = 0
    for i in sequence:
        temp += mass[i]
    water = 18.01528
    return temp + water

def MakeDB(pro ,Enzyme_rule, Missed_cleav=None, enzyme_spec=None):

    if (str(type(Enzyme_rule)) == "<class 'list'>") != True:
        print('The input value has a list shape.')
    else:
        rule = {}
        for i in Enzyme_rule:
            temp = i.split('/')
            if len(temp[0]) > 1:
                for j in temp[0]:
                    rule[j] = temp[1]
            else:
                rule[temp[0]] = temp[1]

    enzyme_site_list = list("".join(list(rule.keys())))
    point = []
    for target in enzyme_site_list:
        index = -1
        while True:
            index = pro.find(target, index + 1)
            if index == -1:
                break
            point.append(index)
    point.sort()

    if enzyme_spec == 2:
        result, F, s_idx, fs_idx = fully(pro, rule, point) # miss_fully, c_idx, Miss =
        if Missed_cleav > 0:
            Missed_cleavage_fully(Missed_cleav, F, fs_idx, pro)

    elif enzyme_spec == 1:
        result, F, s_idx, fs_idx = fully(pro, rule, point)
        semi(result, s_idx, pro)
        if Missed_cleav > 0:
            miss_fully, c_idx, Miss = Missed_cleavage_fully(Missed_cleav, F, fs_idx, pro)
            Missed_cleavage_semi(miss_fully, rule, c_idx, Miss, pro)

    elif enzyme_spec == 0:
        result, F, s_idx, fs_idx = fully(pro, rule, point)
        semi(result, s_idx, pro)
        none(result, s_idx, pro)

        if Missed_cleav > 0:
            miss_fully, c_idx, Miss = Missed_cleavage_fully(Missed_cleav, F, fs_idx, pro)
            Missed_cleavage_semi(miss_fully, rule, c_idx, Miss, pro)
            Missed_cleavage_none(miss_fully, rule, c_idx, Miss, pro)
        pass
    else:
        print('Enzyme Spec(NTT) : fully = 2, semi = 1, none = 0')

def fully(pro, rule, point):
    fully_DB = []
    cond_fully = []
    start = 0
    start_ind = []
    s_idx = []
    for i in point:
        if rule[pro[i]] == "C":
            end = i + 1
            temp = pro[start:end]
            if len(temp) > 5:
                cond_fully.append(temp)
                Mass = cal_mass(temp)
                if start == 0:
                    write([start+1, end, 2, 0, '(-)' + temp + '(' + pro[end] + ')', Mass])
                elif end == len(pro):
                    write([start+1, end, 2, 0, '(' + pro[start-1] + ')' + temp + '(-)', Mass])
                else:
                    write([start+1, end, 2, 0, '(' + pro[start - 1] + ')' + temp + '(' + pro[end] + ')', Mass])
                start_ind.append(start)
            fully_DB.append(temp)
            s_idx.append(start)
        else:
            end = i
            temp = pro[start:end]
            if len(temp) > 5:
                cond_fully.append(temp)
                Mass = cal_mass(temp)
                if start == 0:
                    write([start + 1, end, 2, 0, '(-)' + temp + '(' + pro[end] + ')', Mass])
                elif end == len(pro):
                    write([start + 1, end, 2, 0, '(' + pro[start - 1] + ')' + temp + '(-)', Mass])
                else:
                    write([start + 1, end, 2, 0, '(' + pro[start - 1] + ')' + temp + '(' + pro[end] + ')', Mass])
                start_ind.append(start)
            fully_DB.append(temp)
            s_idx.append(start)
        start = end

    return cond_fully, fully_DB, start_ind, s_idx

def semi(cond_F, idx, pro):

    for i, s in zip(cond_F, idx):
        num = 1
        while True:
            temp_1 = i[0:len(i)-num]
            num += 1
            if len(temp_1) < 6:
                break
            Mass = cal_mass(temp_1)
            end = s+len(temp_1)-1
            if s == 0:
                write([s + 1, end+1, 1, 0, '(-)' + temp_1 + '(' + pro[end+1] + ')', Mass])
            else:
                write([s + 1, end+1, 1, 0, '(' + pro[s - 1] + ')' + temp_1 + '(' + pro[end + 1] + ')', Mass])

        num = 1
        while True:
            temp_2 = i[num:len(i)]
            num += 1
            s += 1
            end = s + len(temp_2) - 1
            if len(temp_2) < 6:
                break
            Mass = cal_mass(temp_2)
            if end == len(pro)-1:
                write([s + 1, end, 1, 0, '(' + pro[s -1] + ')' + temp_2 + '(-)', Mass])
            else:
                write([s + 1, end, 1, 0, '(' + pro[s - 1] + ')' + temp_2 + '(' + pro[end] + ')', Mass])

def none(cond_F, idx, pro):

    for i, s in zip(cond_F, idx):
        for j in range(len(i)):
            start = j+1
            s += 1
            if len(i[start:len(i)-1]) < 6:
                break
            for k in range(len(i)):
                end = len(i)-(k+1)
                temp_1 = i[start : end]
                end_idx = s + len(temp_1) - 1
                if len(temp_1) < 6:
                    break
                Mass = cal_mass(temp_1)
                write([s + 1, end_idx, 0, 0, '(' + pro[s - 1] + ')' + temp_1 + '(' + pro[end_idx+1] + ')', Mass])

def Missed_cleavage_fully(num, result, idx, pro):
    DB = []
    DB_idx = []
    DB_M = []
    window = 1
    for i in range(num):
        start = 0
        window += 1
        while True:
            if len(idx) <= num:
                break
            elif len(idx) == num + 1:
                start = 0
                end = len(result)
                temp = "".join(result[start:end])
                end_end = len(temp)
                DB.append(temp)
                Mass = cal_mass(temp)
                write([start+1 + 1, end_end, i + 1, num, '(-)' + temp + '(-)', Mass])
                DB_idx.append(start)
                DB_M.append(i + 1)
                break
            else:
                s_s = idx[start]
                end = start + window
                end_idx = end - 1
                end_end = idx[end_idx] + len(result[end_idx])
                if result[start] == '' or result[end_idx] == '':
                    pass
                else:
                    temp = "".join(result[start:end])
                    if len(temp) > 5:
                        DB.append(temp)
                        Mass = cal_mass(temp)
                        if start == 0:
                            write([s_s + 1, end_end, 2, i + 1, '(-)' + temp + '(' + pro[end_end] + ')', Mass])
                        elif end_end == len(pro):
                            write([s_s + 1, end_end, 2, i + 1, '(' + pro[s_s-1] + ')' + temp + '(-)', Mass])
                        else:
                            write([s_s + 1, end_end, 2, i + 1, '(' + pro[s_s-1] + ')' + temp + '(' + pro[end_end] + ')', Mass])
                        DB_idx.append(s_s)
                        DB_M.append(i + 1)
                start += 1
                if end == len(result):
                    break

    return DB, DB_idx, DB_M

def Missed_cleavage_semi(result, rule, c_idx, M, pro):

    for i, t, m_m in zip(result, c_idx, M):
        num = 1
        while True:
            temp_1 = i[0:len(i)-num]
            num += 1
            if len(temp_1) < 6:
                break
            last_AA = temp_1[len(temp_1) - 1]
            if last_AA in list(rule.keys()):
                if rule[last_AA] == 'C':
                    break
                else:
                    Mass = cal_mass(temp_1)
                    end = t + len(temp_1) -1
                    if t == 0:
                        write([t + 1, end + 1, 1, m_m, '(-)' + temp_1 + '(' + pro[end + 1] + ')', Mass])
                    else:
                        write([t + 1, end + 1, 1, m_m, '(' + pro[t] + ')' + temp_1 + '(' + pro[end + 1] + ')', Mass])
                    break
            Mass = cal_mass(temp_1)
            end = t + len(temp_1) - 1
            if t == 0:
                write([t + 1, end + 1, 1, m_m, '(-)' + temp_1 + '(' + pro[end + 1] + ')', Mass])
            else:
                write([t + 1, end + 1, 1, m_m, '(' + pro[t - 1] + ')' + temp_1 + '(' + pro[end + 1] + ')', Mass])

        num = 1
        while True:
            temp_2 = i[num:len(i)]
            t += 1
            end = t + len(temp_2) -1
            if len(temp_2) < 6:
                break
            start_AA = temp_2[0]
            num += 1
            if start_AA in list(rule.keys()):
                if rule[start_AA] == 'C':
                    Mass = cal_mass(temp_2)
                    if end == len(pro) - 1:
                        write([t + 1, end + 1, 1, m_m, '(' + pro[t - 1] + ')' + temp_2 + '(-)', Mass])
                    else:
                        write([t + 1, end + 1, 1, m_m, '(' + pro[t - 1] + ')' + temp_2 + '(' + pro[end + 1] + ')', Mass])
                    break
                else:
                    break
            Mass = cal_mass(temp_2)
            if end == len(pro) - 1:
                write([t + 1, end + 1, 1, m_m, '(' + pro[t - 1] + ')' + temp_2 + '(-)', Mass])
            else:
                write([t + 1, end + 1, 1, m_m, '(' + pro[t - 1] + ')' + temp_2 + '(' + pro[end + 1] + ')', Mass])

def Missed_cleavage_none(result, rule, c_idx, M, pro):

    for i, tt, m_m in zip(result, c_idx, M):
        for j in range(len(i)):
            start = j + 1
            tt += 1
            if len(i[start:len(i) - 1]) < 6:
                break
            start_AA = i[start]
            if start_AA in list(rule.keys()):
                if rule[start_AA] == 'C':
                    for k in range(len(i)):
                        end = len(i) - (k + 1)
                        temp_1 = i[start: end]
                        end_idx = tt + len(temp_1) - 1
                        if len(temp_1) < 6:
                            break
                        Mass = cal_mass(temp_1)
                        write([tt + 1, end_idx + 1, 0, m_m, '(' + pro[tt - 1] + ')' + temp_1 + '(' + pro[end_idx + 1] + ')',Mass])
                    break
                else:
                    break

            for t in range(len(i)):
                end = len(i) - (t + 1)
                temp_1 = i[start: end]
                end_idx = tt + len(temp_1) - 1
                if len(temp_1) < 6:
                    break
                last_AA = temp_1[len(temp_1)-1]
                if last_AA in list(rule.keys()):
                    if rule[last_AA] == 'C':
                        break
                    else:
                        Mass = cal_mass(temp_1)
                        write([tt + 1, end_idx, 0, m_m, '(' + pro[tt - 1] + ')' + temp_1 + '(' + pro[end_idx + 1] + ')',Mass])
                        break
                Mass = cal_mass(temp_1)
                write([tt + 1, end_idx, 0, m_m, '(' + pro[tt - 1] + ')' + temp_1 + '(' + pro[end_idx + 1] + ')', Mass])

# *, **

import sys

input = sys.argv[1]
option_1 = sys.argv[2]
option_2 = int(sys.argv[3])
option_3 = int(sys.argv[4])
output = sys.argv[5]
option_1 = option_1.split(",")

fasta_file = open(input)
sequence = fasta_file.readlines()
pre_proc = "".join(sequence)
pre_proc = pre_proc.split('>')
del pre_proc[0]

f = open(output,'w')

for i in pre_proc:
    count = 1
    a = i.partition('\n')
    pro = a[2].replace("\n", '')
    f.write('\n'+'Protein' + ' >' + a[0] + "\n")
    f.write('Peptide No.' + '\t' + 'Start' + '\t' + 'End' + '\t' + 'NTT' + '\t' + 'Missed Cleavages' + '\t' + 'Sequence' + '\t' + 'Mass'+ "\n")
    MakeDB(pro, option_1, Missed_cleav=option_2, enzyme_spec=option_3)