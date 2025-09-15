
"""
modified Prion Aggregation Prediction Algorithm (mPAPA) - a program for predicting the prion propensity of a protein

FoldIndex formula:
2.785(H) - |R| - 1.151
H = sum of the hydrophobicities across the window							
R = net charge (where D/E=-1; K/R=+1; all others = neutral (including H))							
"""


amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

propensities = {'A' :-0.396490246, 'C' : 0.415164505, 'D' : -1.276997939, 'E' : -0.605023827, 'F' : 0.838732498,
                'G' : -0.039220713, 'H': -0.278573356, 'I' : 0.813697862, 'K': -1.576748587, 'L' : -0.040005335,
                'M' : 0.673729095, 'N' : 0.080295334, 'P' : -1.197447496, 'Q' : 0.069168387, 'R' : -0.405858577,
                'S' : 0.133912418, 'T' : -0.11457038, 'V' : 0.813697862, 'W' : 0.666735081, 'Y' : 0.77865336}
																			
hydrophobicities = {'A' : 0.7, 'C' : 0.777777778, 'D' : 0.111111111, 'E' : 0.111111111, 'F' : 0.811111111,
                  'G' : 0.455555556, 'H' : 0.144444444, 'I' : 1, 'K' : 0.066666667, 'L': 0.922222222,
                  'M' : 0.711111111, 'N' : 0.111111111, 'P' : 0.322222222, 'Q' : 0.111111111, 'R' : 0,
                  'S' : 0.411111111, 'T' : 0.422222222, 'V' : 0.966666667, 'W' : 0.4, 'Y' : 0.355555556}

charges = {'A' : 0, 'C' : 0, 'D' : -1, 'E' : -1, 'F' : 0,
          'G' : 0, 'H' : 0, 'I' : 0, 'K' : 1, 'L': 0,
          'M' : 0, 'N' : 0, 'P' : 0, 'Q' : 0, 'R' : 1,
          'S' : 0, 'T' : 0, 'V' : 0, 'W' : 0, 'Y' : 0}


def window_scores(args, sequence, aa_dict, ignore_consecutive_prolines = False) :

    return [window_score(args, sequence, i, aa_dict, ignore_consecutive_prolines) for i in range(len(sequence))]

def window_score(args, sequence, position, aa_dict, ignore_consecutive_prolines = False) :
    """
    Calculates average mPAPA score for each individual window.
    """
    
    start,end = get_window(args, sequence, position)
    score = 0.0
    for i in range(start, end) :
        if sequence[i] not in amino_acids :
            continue
        if (not ignore_consecutive_prolines) :
            score += aa_dict[sequence[i]]
        else :
            if (sequence[i] != 'P') :
                score += aa_dict[sequence[i]]
            elif ((i > 0 and sequence[i-1] == 'P') or (i > 1 and sequence[i-2] == 'P')) :
                pass
            else :
                score += aa_dict[sequence[i]]

    return score / (end - start)

def get_window(args, sequence, position) :
    """New window definitions for new end-scoring method in mPAPA.

    Window start and end were shifted to start at the beginning of the protein.
    The change (compared to originalPAPA) also moves the windows such that scores are now assigned to the last residue of the first 41aa window.
    Within the protein, this will be the last residue of the first 41aa window.
    For ends, this will be the last residue of the first theoretical 41aa window, which is residue_1 of the protein (start would have started at residue -40, but max() 0 if start would otherwise be negative).
   """ 
   
    start = max(position - (args.window_size-1), 0)
    end = min(position+1, len(sequence)+1)  
    
    return start,end
                                   
def fold_index(args, sequence) :
    """
    New method that calculates a single FoldIndex score for each window in mPAPA 
    (i.e. does NOT calculate the average FoldIndex over consecutive windows, as implemented
    in the previous version of PAPA)
    
    This has also been verified to perform calculations properly for the ends of proteins,
    in a manner congruent with the finalized method for end-scoring.
    """
    
    sequence = sequence[:-args.window_size+1]
    fold_index_list = []
    for position in range(len(sequence)):
        start = max(position - (args.window_size-1), 0)
        end = min(position+args.window_size, len(sequence)+1)
        hydrophobicity = 0.0
        charge = 0.0

        window_seq = sequence[start:end]
        for res in window_seq:
            if res not in hydrophobicities:
                continue
            hydrophobicity += hydrophobicities[res]
            charge += charges[res]
        fold_index = 2.785*(hydrophobicity/len(window_seq)) - abs(charge/len(window_seq)) - 1.151
        fold_index_list.append(fold_index)
        
    return fold_index_list

def super_window_scores(args, sequence, window_scores, fold_index_scores = None) :
    """
    Calculates the average score across n=window_size consecutive windows (i.e. the final mPAPA score for each position).
    
    Used the original range in PAPA, but this only works because a string of X's is 
    first added to the sequence (performed in the run() function) in order
    to create a scores list with length == sequence length.
    
    Otherwise the scoring would stop 40 residues from the end, which is 
    similar to the original PAPA method. However, with the extra X's, the 
    length of final scores == sequence length.
    """
    scores = []
    for i in range(len(sequence) - args.window_size+1) :
        if (fold_index_scores is not None and fold_index_scores[i] > 0) :
            scores.append(None)
        else :
            score = 0.0
            weights = 0.0
            for j in range(i, i + args.window_size) :
                start,end = get_window(args, sequence, j)
                score += (end - start) * window_scores[j]
                
                #Correction that excludes X's (added to sequence in run() function) from the score weighting
                non_X_seq = sequence[start:end].replace('X','')
                weights += len(non_X_seq)

            scores.append(score / weights)

    return scores

def classify(args, sequence, ignore_fold_index = False) :

    fold_index_list = fold_index(args, sequence)

    window_propensities = window_scores(args, sequence, propensities, True)
    if ignore_fold_index :
        scores = super_window_scores(args, sequence, window_propensities)
    else :
        scores = super_window_scores(args, sequence, window_propensities, fold_index_list)
        
    max_score = -1.0
    max_position = -1
    for score in scores:
        if score != None and score > max_score:
            max_score = score
            max_position = scores.index(max_score)

    return max_score, max_position+1, scores, fold_index_list
    

class MalformedInput :
    "Exception raised when the input file does not look like a fasta file."
    pass

class FastaRecord :
    "a fasta record."
    def __init__(self, header, sequence):
        self.header = header
        self.sequence = sequence

def _fasta_itr(file_handle) :
    "Provide an iteration through the fasta records in file."

    h = file_handle.readline()[:-1]
    if h[0] != '>':
        raise MalformedInput()
    h = h[1:]

    seq = []
    for line in file_handle:
        line = line[:-1] # remove newline
        if line[0] == '>':
            yield FastaRecord(h,''.join(seq))
            h = line[1:]
            seq = []
            continue
        seq.append(line)
    yield FastaRecord(h,''.join(seq))

class fasta_itr (object) :
    "An iterator through a sequence of fasta records."
    def __init__(self, src) :
        self.__itr = _fasta_itr(src)

    def __iter__(self) :
        return self

    def __next__(self) :
        return self.__itr.__next__()

#===================================================================================================================#

#Function Author: Sean Cascarina
#Added: 6/27/2018

def merge_overlapping(args, hit_positions):
    """A function that accepts a list of positions that are above a certain mPAPA threshold and
        effectively merges positions that would result in overlapping mPAPA-positive windows.
        
        The return is a new list of 2-tuples, where the first value is the start of the mPAPA-positive window, 
        and the second value is the end of the mPAPA-positive window
        e.g. if the original position list = [1, 2, 3, 4, 5, 8, 55, 56, 57, 90, 91, 115, 116, 117, 234, 236, 237],
        then this function would return the list [(0,157) , (194,279)]
        
        The position_extension_combinations list is an intermediate list that is used to calculate
        the start and end positions of the full mPAPA-positive windows.
        This is a list of 2-tuples, where the first value is the position, and the second value
        is the number of extension residues to add to the position.
        The position_extension_combinations list
        e.g. for a hypothetical 2-tuple (136,5), the full mPAPA window would be calculated as amino acids 96-182.
        The calculation would be (136-40) + 81 + 5
                                     ^      ^    ^
                                     |      |    |
                           start position   |   extension residues
                                      papa window size
                                      
        """
    
    j = 0
    position_extension_combinations = []
    
    #Runs until hit_positions is empty. Every time a new merged window is generated, that window is removed from hit_positions.
    #This eventually results in an empty hit_positions list.
    while hit_positions:
        start = hit_positions[0]
        
        #adds one to the index j while there are at least two items in the list and the two values at j and j+1 are within 81 amino acids of each other, indicating overlapping PAPA windows
        #once these conditions aren't met, the loop exits and appends a tuple to position_extension_combinations with the start position as the first value and the number of extension residues as the second value (extension residues are the number of amino acids to add to the start position to get to the position of the last score with an overlapping window)
        while j < (len(hit_positions) - 1) and ( (hit_positions[j+1] - (args.window_size - 1) ) <= ( hit_positions[j] + (args.window_size - 1) ) ):
            j += 1
        
        # Gets the ending position for the current window and outputs the 2-tuple.
        end = hit_positions[j]
        extension_residues = end - start
        position_extension_combinations.append( (start , extension_residues) )

        #modifies hit_positions to delete all positions that were just merged, then re-sets j=0 to start iterating at the first position in the new list
        hit_positions = hit_positions[j+1:]
        j = 0
        
    return position_extension_combinations

#================================================================================================================#    

#Function Author: Sean Cascarina
#Added: 6/27/2018
    
def get_high_scoring_indices(args, sequence, highest_score_position, papa_scores):

    hit_indices = []
    threshold = args.threshold
    
    #get window indices for all prlds above threshold
    filtered_nones = [float(x) for x in papa_scores if x != None ]
    if len(filtered_nones) > 0 and float(max( filtered_nones )) >= threshold:
        index = 0
        for score in papa_scores:
            if score != None and float(score) >= threshold:
                hit_indices.append(index)
            index += 1
        position_extension_combinations = merge_overlapping(args, hit_indices)
        
        #use position/extensions combinations to calculate the actual indices of the start and end of PAPA positive windows
        hit_indices = [ (( x - (args.window_size - 1) ) , (x+(args.window_size-1)+y)) if x>=args.window_size else (0 , (x+(args.window_size-1)+y)) for (x,y) in position_extension_combinations ] #corrects for positions that are near the N-terminus to prevent negative x indices
        hit_indices = [ (x,y) if y <= len(sequence[:-args.window_size]) else ( x, len(sequence[:-args.window_size]) ) for (x,y) in hit_indices ] #corrects for positions that are near the C-terminus to prevent y indices that extend beyond the length of the protein. NOTE: the sequences in orf_trans.dat still contain the '*' stop codon.

    #otherwise, output the window indices for the highest scoring 81aa region only
    else:
        position_extension_combinations = [ (int(highest_score_position), 0) ]
        
        #use position/extensions combinations to calculate the actual indices of the start and end of PAPA positive windows
        hit_indices = [ (( x - (args.window_size - 1) ) , (x+(args.window_size-1)+y)) if x>=args.window_size else (0 , (x+(args.window_size-1)+y)) for (x,y) in position_extension_combinations ] #corrects for positions that are near the N-terminus to prevent negative x indices
        hit_indices = [ (x,y) if y <= len(sequence[:-args.window_size]) else ( x, len(sequence[:-args.window_size]) ) for (x,y) in hit_indices ] #corrects for positions that are near the C-terminus to prevent y indices that extend beyond the length of the protein. NOTE: the sequences in orf_trans.dat still contain the '*' stop codon.

    return hit_indices
    
#=========================================================================================================================#

#Function Author: Sean Cascarina
#Added: 6/27/2018

def get_merged_window_seqs(sequence, hit_indices):
    """A function that accepts the name of the gene of interest and a list of the indices of all mPAPA-positive regions
        and returns the amino acid sequences of those regions"""
        
    seqs = []
    for pos in hit_indices:
        start, end = pos
        seq = sequence[ start : end+1 ]
        seqs.append(seq)

    return seqs


def txt_itr(file_handle):
    """Iterate through a plain text file with one sequence per line."""
    for idx, line in enumerate(file_handle):
        seq = line.strip().upper()
        if not seq:
            continue
        header = f"seq_{idx+1}"
        yield FastaRecord(header, seq)
    
#=========================================================================================================================#

def run(args):
    if args.outfile is None:
        outfile = sys.stdout
    else:
        outfile = open(args.outfile, 'w')
        
    seq_iterator = txt_itr(open(args.fasta_input))

    # if args.input_type == 'fasta':
    #     seq_iterator = fasta_itr(open(args.fasta_input))
    # elif args.input_type == 'txt':
    #     seq_iterator = txt_itr(open(args.fasta_input))
    # else:
    #     raise ValueError("Unknown input type. Must be 'fasta' or 'txt'.")

    high_scores = []
    for fasta_record in seq_iterator:
        sequence_id = fasta_record.header
        sequence = fasta_record.sequence.upper()
        
        # Process sequence like before
        sequence = sequence.replace('*', '').replace('X', '?')
        sequence += ('X'*(args.window_size-1))

        if len(sequence) <= args.window_size:
            outfile.write(f"{sequence_id}\tprotein length below window size\n")
            continue

        high_score, pos, scores, fold_indexes = classify(args, sequence, args.ignore_fold_index)

        # if high_score == -1.0:
        #     outfile.write(f"{sequence_id}\t{high_score}\n")
        #     continue
        high_scores.append(high_score)

    return high_scores


# def run(args) :

#     if args.outfile is None :
#         outfile = sys.stdout
#     else :
#         outfile = open(args.outfile, 'w')

#     # if args.verbose :
#     #     outfile.write('FASTA Header\tHighest Score\tPosition of Highest Score\tPAPA Scores\tFoldIndex Scores\n')
#     # else:
#     #     outfile.write('FASTA Header\tHighest Score\tPosition of Highest Score\tPredicted Prion Domain\tAmino Acid Boundaries of Predicted Prion Domain\n')
    
#     high_scores = []
#     for fasta_record in fasta_itr(open(args.fasta_input)) :
#         sequence_id = fasta_record.header
#         sequence = fasta_record.sequence.upper()
        
#         #Modify sequence to exclude stop codons ('*') and to add a string of X's to the end in order to score the C-terminus, which allows for generating final scores with length == sequence length
#         #This also replaces any pre-existing X's (which are present in some proteomes with poor annotation and/or confidence in protein sequence).
#         sequence = sequence.replace('*', '').replace('X', '?')
#         sequence += ('X'*(args.window_size-1))

#         if len(sequence) <= args.window_size :
#             outfile.write(str(sequence_id) + '\t' + 'protein length below window size\n')
#             continue
            
#         high_score,pos,scores,fold_indexes = classify(args, sequence, args.ignore_fold_index)
        
#         if high_score == -1.0:
#             outfile.write(sequence_id + '\t' + str(high_score) + '\n')
#             continue   
#         high_scores.append(high_score)
            
#         #get the window indices for all PAPA-positive windows and the corresponding sequences
#         window_indices = get_high_scoring_indices(args, sequence, pos, scores)
#         hit_sequences = get_merged_window_seqs(sequence, window_indices)
        
#         #shorten papa_score values by rounding floats to 5 places past the decimal, and by shortening 'None' to 'N'
#         #improves formatting for scores corresponding to extremely long proteins when results are opened in Excel
#         papa_scores = [ round(float(x), 5) if x != None else 'N' for x in scores ]
        
#         #START OUTPUT===========================

#         # #output main scores
#         # if args.verbose :
#         #     scores_token = '[' + ' '.join([str(s) for s in scores ]) + ']'
#         #     fold_index_token = '[' + ' '.join([str(f) for f in fold_indexes ]) + ']'
#         #     outfile.write(str(sequence_id) + '\t' + str(high_score) + '\t' + str(pos) +'\t' +
#         #                   scores_token + '\t' + fold_index_token + '\n')
#         # else :
        
#         #     outfile.write(sequence_id + '\t' + str(high_score) + '\t' + str(pos) + '\t')
            
#         #     #output sequences of highest-scoring domain(s)
#         #     if len(hit_sequences) > 1:
#         #         seqs_token = ['_;_'.join(seq for seq in hit_sequences)]
#         #         outfile.write(str(seqs_token)[2:-2] + '\t')
#         #     else:
#         #         outfile.write(str(hit_sequences)[2:-2] + '\t')
                
#         #     #output window indices of highest-scoring domain(s)
#         #     if len(window_indices) > 1:
#         #         windows_token = ['_;_'.join(str( (window[0]+1, window[1]+1) ) for window in window_indices)]
#         #         outfile.write(str(windows_token)[2:-2] + '\n')
#         #     else:
#         #         windows_token = [(window_indices[0][0]+1 , window_indices[0][1]+1)]
#         #         outfile.write(str(windows_token)[1:-1] + '\n')
#     return high_scores
    
def test() :
    scores = {'Ure2' : 0.1031, 'Sup35': 0.0997, 'Rnq1' : 0.1413, 'New1' : 0.1398, 'Nsp1' : 0.0239, 'Puf2' : 0.0768, 'Pub1' : 0.1551}
    sequences = {}
    for fasta_record in fasta_itr(open('sequences.fasta')) :
        sequences[fasta_record.header] = fasta_record.sequence
    for id in sequences :
        score,pos,scores,fold_index = classify(sequences[id])
        print(id, len(sequences[id]), pos, score, scores[id], score / scores[id])
    print('scores ignoring foldIndex')
    for id in sequences :
        score,pos,scores,fold_index = classify(sequences[id], True)
        print(id, len(sequences[id]), pos, score, scores[id], score / scores[id])

def parse_arguments(arguments) :
    import argparse
    parser = argparse.ArgumentParser(description='Predict whether a given protein is prion forming', prog = 'mpapa')
    parser.add_argument('fasta_input', help = 'the input file (in fasta format)')
    parser.add_argument('-o', '--outfile', type = str,
                        help = """the output file. In non-verbose mode the output is a tab-delimited file the columns are:
                        sequence id, maximum score, position
                        maximum score is the largest window score, and position is the position where it occurs.
                        In verbose mode the output also includes the scores for the whole protein, and the fold index scores.
                        If not output file is given, results are printed to the screen.
                        """)
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help = """in verbose mode the output includes the scores for every position as well as the per-position fold index scores""")
    parser.add_argument('--threshold', type=float, default = 0.05,
                        help = """Set the mPAPA sccore threshold (default=0.05) for merging high-scoring windows.""")
    parser.add_argument('--ignore_fold_index', action='store_true',
                        help = """Whether to ignore the fold index values for the protein (default: False)
                        Set it to True to ignore fold index.
                        """)
    parser.add_argument('--window_size', type = int, default = 41,
                        help = """The window size used by mPAPA (default = 41).
                        Proteins that are shorter than the window size are not classified.
                        """)
    args = parser.parse_args(arguments)

    return args

if __name__ == '__main__' :

    import sys
    args = parse_arguments(sys.argv[1:])
    high_scores = run(args)
    print(high_scores)
    #test()