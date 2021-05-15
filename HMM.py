# Sylvan Avery Dekker
# University of Massachusetts, Amherst
# 2 April 2021


# the A and B matrices as dictionaries
A = {'N':{'N':0.54, 'V':0.23, 'R':0.08, '#':0.15}, 'V':{'N':0.62, 'V':0.17, 'R':0.11, '#':0.10},
     'R':{'N':0.17, 'V':0.68, 'R':0.10, '#':0.05}, '#':{'N':0.7, 'V':0.2, 'R':0.1, '#':0}}

B = {'N':{'time':0.98, 'flies':0.015, 'quickly':0.005, '#':0},
     'V':{'time':0.33, 'flies':0.64, 'quickly':0.03, '#':0},
     'R':{'time':0.01, 'flies':0.01, 'quickly':0.98, '#':0},
     '#':{'time':0, 'flies':0, 'quickly':0, '#':1.0}}

# Define a new emission matrix B2 for Question 4
B2 = {'N':{'swat':0, 'flies':0.98, 'quickly':0.02, '#':0},
      'V':{'swat':0.64, 'flies':0.33, 'quickly':0.03, '#':0},
      'R':{'swat':0, 'flies':0.015, 'quickly':0.985, '#':0},
      '#':{'swat':0, 'flies':0, 'quickly':0, '#':1.0}}
# two data structures you may find useful for mapping between tags and their (arbitrary) indices
tagnum = {"N":0,"V":1,"R":2,"#":3}    #gives index for a given tag
numtag = ['N','V','R','#']   #gives tag for a given index

def print_table(table, words, alg_type, ef='%.4f', colwidth=12):
    tags = A.keys()
    if alg_type == 1:
        print('-'*8 + '-'*(12*(len(words))) + '\n' + '+ '*11 + ' FORWARD ' + ' +'*6*(len(words)-2))
    else:
        print('-'*8 + '-'*(12*(len(words))) + '\n' + '/ '*11 + ' VITERBI ' + ' /'*6*(len(words)-2))
    print('-'*8 + '-'*(12*(len(words))))
    print(('| ').rjust(9), end='')
    for w in words:
        print(str(' ' + w + ' '*(9-len(w)) + '|').ljust(colwidth), end='')
    print('\n' + '-'*7 + ('|' + '-'*11)*len(words) + '|')
    for n in range(len(tags)):
        print(str(' '*(4) + numtag[n]) + '  |'.ljust(5), end='')
        for t in range(len(words)):
            out = str(table[t][n])
            if type(table[t][n]) == tuple:
                form=ef + ",%s"
                out = form % (table[t][n][0], table[t][n][1])
            elif type(table[t][n]) == float:
                out = str(ef % table[t][n] + '   |')
            print(out.ljust(colwidth), end='')
        print()
    print('' * 68)
    print('')


# ███████╗ ██████╗ ██████╗ ██╗    ██╗ █████╗ ██████╗ ██████╗
# ██╔════╝██╔═══██╗██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔══██╗
# █████╗  ██║   ██║██████╔╝██║ █╗ ██║███████║██████╔╝██║  ██║
# ██╔══╝  ██║   ██║██╔══██╗██║███╗██║██╔══██║██╔══██╗██║  ██║
# ██║     ╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║██║  ██║██████╔╝
# ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝


def forward(test_seq, A, B):
    test_seq = list(test_seq)
    test_seq.append('#')
    test_seq.insert(0, '#')
    T = len(test_seq)  # Number of observed words (timesteps)
    N = len(A)  # Number of possible states
    fwd_prob = 0.0
    fwd_matrix = [{}]

    for j in range(N):  # Initialization (all probs @ timestep 1 == 0)
        fwd_matrix[0][j] = 0.0

    fwd_matrix[0][tagnum['#']] = 1.0  # More initialization (set prob of '#' @ timestep 1 == 1)

    # Recursion
    for t in range(1, T):  # For each timestep 2 to final
        fwd_matrix.append({})  # Add empty dictionary to hold probabilities @ timestep 't'
        for j in range(N):  # For each possible POS ('to' state)
            sum_i = 0.0
            for i in range(N):  # For each possible POS ('from' state)
                # Prob of POS 'j' @ timestep 't'
                # == prob of POS 'i' @ timestep 't-1'
                # * transition prob of POS 'j' given POS 'i'
                # * emission prob of observed word @ timestep 't' given POS 'j'
                sum_i += fwd_matrix[t-1][i] * A[numtag[i]][numtag[j]] * B[numtag[j]][test_seq[t]]
            fwd_matrix[t][j] = sum_i  # Record probability of POS 'j' at timestep 't'

    for j in range(N):  # Calculate forward prob (sum of probs @ final timestep)
        fwd_prob += fwd_matrix[T-1][j]

    print_table(fwd_matrix, test_seq, 1)

    return fwd_prob


#  ██╗   ██╗██╗████████╗███████╗██████╗ ██████╗ ██╗
#  ██║   ██║██║╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██║
#  ██║   ██║██║   ██║   █████╗  ██████╔╝██████╔╝██║
#  ╚██╗ ██╔╝██║   ██║   ██╔══╝  ██╔══██╗██╔══██╗██║
#   ╚████╔╝ ██║   ██║   ███████╗██║  ██║██████╔╝██║
#    ╚═══╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝


def viterbi(test_seq, A, B):  # A = transmission probs, B = emission probs
    test_seq = list(test_seq)
    test_seq.append('#')
    test_seq.insert(0, '#')
    T = len(test_seq)  # Number of observed words (timesteps)
    N = len(A)  # Number of possible states
    v_matrix = [{}]  # For deltas
    back_pointers = {}  # For psis

    for j in range(N):  # Initialization (all probs @ timestep 1 == 0)
        v_matrix[0][j] = 0.0
        back_pointers[j] = [j]  # Backpointers @ POS 'j' == 'j'

    v_matrix[0][tagnum['#']] = 1.0  # More initialization (set prob of '#' @ timestep 1 == 1)

    for t in range(1, T):  # For each timestep 2 to final
        v_matrix.append({})  # Add empty dictionary to hold probabilities @ timestep 't'
        nu_back_points = {} # Create temp dictionary to hold backpointers @ timestep 't'
        for j in range(N):  # For each possible POS ('to' state)
            max_v = 0.0
            back_pointer = 0.0
            for i in range(N):  # For each possible POS ('from' state)
                # Prob of POS 'j' @ timestep 't'
                # == prob of POS 'i' @ timestep 't-1'
                # * transition prob of POS 'j' given POS 'i'
                # * emission prob of observed word @ timestep 't' given POS 'j'
                v = v_matrix[t-1][i] * A[numtag[i]][numtag[j]] * B[numtag[j]][test_seq[t]]
                # If current prob of POS 'j' @ timestep 't'
                # > previous prob of POS 'j' @ timestep 't',
                # replace previous prob with new prob
                if v > max_v:
                    max_v = v
                    back_pointer = i  # Record backpointer corresponding w/ max prob of POS 'j' @ timestep 't'
            v_matrix[t][j] = max_v  # Record max prob of POS 'j' @ timestep 't'
            # The list paired w/ key 'j' (in 'nu_back_points') is replaced w/ the list paired w/ key
            # 'back_pointer' (in 'back_pointers') appended w/ 'j'
            nu_back_points[j] = back_pointers[back_pointer] + [j]
        back_pointers = nu_back_points  # 'back_pointers' is replaced with 'nu_back_points' b/c old paths do not matter

    max_p = 0.0
    argmax = None
    for i in range(N):  # Find word w/ greatest probability (@ penultimate timestep)
        if v_matrix[T-1][i] > max_p:
            max_p = v_matrix[T-1][i]
            argmax = i

    tags = []
    for i in range(len(back_pointers[argmax])):  # Converts best path to corresponding POS tags
        tags.append(numtag[back_pointers[argmax][i]])

    print_table(v_matrix, test_seq, 2)

    return (max_p, tags)


### MAIN CODE GOES HERE ###
def main():

    seq = ('time','flies','quickly')
    print("Forward probability of '" + str("{} "*len(seq)).format(*seq).rstrip() + "': {0}\n".format(forward(seq, A, B)))
    print("Most likely tags for '" + str("{} "*len(seq)).format(*seq).rstrip() + "': {0[1]}\n\nProbability: {0[0]}\n".format(viterbi(seq, A, B)))
    seq2 = ('swat','flies','quickly')
    print("Most likely tags for '" + str("{} "*len(seq2)).format(*seq2).rstrip() + "': {0[1]}\n\nProbability: {0[0]}\n".format(viterbi(seq2, A, B2)))


if __name__ == "__main__":
    main()