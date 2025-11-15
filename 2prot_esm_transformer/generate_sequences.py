import random

alphabet = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", 
               "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

N = 200
k = 2
L = 50

with open('2prot_sequences.txt', 'w') as file:
    for i in range(N):
        seqs = []
        for _ in range(k):
            seq = ''
            for _ in range(L):
                residue = random.choice(alphabet)
                seq += residue
            seqs.append(seq)
        file.write(f'{i}\t{'\t'.join(seqs)}\n')
file.close()
    
