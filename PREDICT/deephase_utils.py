import numpy as np
import pandas as pd
import pickle
import math
import tempfile
import subprocess

SEED = 42
np.random.seed(SEED)
pd.set_option("display.max_colwidth", False)
pd.set_option('display.expand_frame_repr', False)

from sklearn.ensemble import RandomForestClassifier
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP


AA_array = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

kd = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5,
      "C": 2.5,  "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
      "I": 4.5,
      "L": 3.8, "K": -3.9,
      "M": 1.9,
      "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3,
      "V": 4.2} 


def hydrophobicity(seq):
    sequence = ProteinAnalysis(seq)
    HB = 0
    for k in range(0, len(AA_array)):
        HB = HB + sequence.count_amino_acids()[AA_array[k]] * kd[AA_array[k]]        
    
    return HB


def Shannon_entropy(seq):
    sequence =  ProteinAnalysis(seq)
    entropy = 0
    for k in range(0, len(AA_array)):
        if sequence.get_amino_acids_percent()[AA_array[k]] == 0:
            entropy = entropy + 0
        else:
            entropy = entropy - math.log2(sequence.get_amino_acids_percent()[AA_array[k]]) * sequence.get_amino_acids_percent()[AA_array[k]]        
    return entropy


def extract_LCR(seq):
    tmp_LCR = tempfile.NamedTemporaryFile()  
    with open(tmp_LCR.name, 'w') as f_LCR:
         f_LCR.write('>1\n' + str(seq))
    tmp_LCR.seek(0)
    
    out = subprocess.Popen(['segmasker', '-in', str(tmp_LCR.name)], 
           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout_LCR, stderr_LCR = out.communicate() 
    stdout_LCR = stdout_LCR.split()[1:]
    
    LCR_start_values = []; LCR_end_values = []    
    for i in range(0, int(len(stdout_LCR)/3)):
        LCR_start_values.append(int(str(stdout_LCR[3*i],'utf-8')))
        LCR_end_values.append(int(str(stdout_LCR[3*i + 2],'utf-8')))
    LCR_residues = []
    for i in range(0, len(LCR_start_values)):
        LCR_residues.extend(list(np.linspace(LCR_start_values[i], LCR_end_values[i], (LCR_end_values[i] - LCR_start_values[i] + 1) )))
    LCR_residues = sorted(list(set(LCR_residues)))
    LCR_sequence = ''
    for i in range(0, len(LCR_residues)):
        LCR_sequence = LCR_sequence + seq[int(LCR_residues[i]-1)]
        
    return len(LCR_residues), LCR_sequence


def extract_IDR(seq):
    tmp_IDR = tempfile.NamedTemporaryFile()  
    with open(tmp_IDR.name, 'w') as f_IDR:
         f_IDR.write('>1\n' + str(seq))
    tmp_IDR.seek(0)
    
    out = subprocess.Popen(['python', 'tools/iupred2a.py', str(tmp_IDR.name), 'long'], 
           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout_IDR, stderr_IDR = out.communicate()
    stdout_IDR = stdout_IDR.split()[40:]
    
    IDR_prob = []
    for i in range(0, int(len(stdout_IDR)/3)):
        IDR_prob.append(float(str(stdout_IDR[3*i + 2], 'utf-8')))
       
    TH1 = 0.5
    TH2 = 20
    AAs   = pd.Series(list(map(lambda i:i, seq)))    
    IDR_residues = []
    current = 0
    for t in range(0, len(IDR_prob)):
        if IDR_prob[t] > TH1:
            current = current + 1
            if t == len(IDR_prob) - 1:
                if current > TH2:
                    IDR_residues.extend(range(t - current , t + 1))
        else:
            if current > TH2:
                IDR_residues.extend(range(t - current , t + 1))
                current = 0
            else:
                current = 0
    
    return len(IDR_residues)


def get_AA_count(seq, AA):
    if type(seq) == float:
        count = 0
    else:
        sequence = ProteinAnalysis(seq)
        count = sequence.count_amino_acids()[str(AA)]
    return count

#Types of amino acids
Hydrophobic_AAs = ['A', 'I', 'L', 'M', 'F', 'V']
Polar_AAs = ['S', 'Q', 'N', 'G', 'C', 'T', 'P']
Cation_AAs = ['K', 'R', 'H']
Anion_AAs = ['D', 'E']
Arom_AAs = ['W', 'Y', 'F']



from gensim.models import word2vec

def split_ngrams(seq, n):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    """
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams

def generate_corpusfile(fasta_fname, n, corpus_fname):
    '''
    Args:
        fasta_fname: corpus file name
        n: the number of chunks to split. In other words, "n" for "n-gram"
        corpus_fname: corpus_fnameput corpus file path
    Description:
        Protvec uses word2vec inside, and it requires to load corpus file
        to generate corpus.
    '''
    f = open(corpus_fname, "w")
    fasta = Fasta(fasta_fname)
    for record_id in tqdm(fasta.keys(), desc='corpus generation progress'):
        r = fasta[record_id]
        seq = str(r)
        ngram_patterns = split_ngrams(seq, n)
        for ngram_pattern in ngram_patterns:
            f.write(" ".join(ngram_pattern) + "\n")
    f.close()
    
'''
Binary representation of amino acid residue and amino acid sequence
e.g.
    'A' => [0, 0, 0, 0, 0]
    'AGGP' => [[0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1]]
'''

AMINO_ACID_BINARY_TABLE = {
    'A': [0, 0, 0, 0, 0],
    'C': [0, 0, 0, 0, 1],
    'D': [0, 0, 0, 1, 0],
    'E': [0, 0, 0, 1, 1],
    'F': [0, 0, 1, 0, 0],
    'G': [0, 0, 1, 0, 1],
    'H': [0, 0, 1, 1, 0],
    'I': [0, 0, 1, 1, 1],
    'K': [0, 1, 0, 0, 0],
    'L': [0, 1, 0, 0, 1],
    'M': [0, 1, 0, 1, 0],
    'N': [0, 1, 0, 1, 1],
    'P': [0, 1, 1, 0, 0],
    'Q': [0, 1, 1, 0, 1],
    'R': [0, 1, 1, 1, 1],
    'S': [1, 0, 0, 0, 0],
    'T': [1, 0, 0, 0, 1],
    'V': [1, 0, 0, 1, 0],
    'W': [1, 0, 0, 1, 1],
    'Y': [1, 0, 1, 0, 0]
}

def convert_amino_to_binary(amino):
    '''
    Convert amino acid to 1-dimentional 5 length binary array
    "A" => [0, 0, 0, 0, 0]
    '''
    if not AMINO_ACID_BINARY_TABLE.has_key(amino):
        return None
    return AMINO_ACID_BINARY_TABLE[amino]


def convert_amino_acid_sequence_to_vector(sequence):
    '''
    "AGGP" => [[0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1]]
    '''
    binary_vector = [convert_amino_to_binary(amino) for amino in sequence]
    if None in binary_vector:
        return None
    return binary_vector

def normalize(x):
    return x / np.sqrt(np.dot(x, x))

class ProtVec(word2vec.Word2Vec):

    def __init__(self, fasta_fname=None, corpus=None, n=3, size=100, corpus_fname="corpus.txt",  sg=1, window=25, min_count=1, workers=20):
        """
        Either fname or corpus is required.
        fasta_fname: fasta file for corpus
        corpus: corpus object implemented by gensim
        n: n of n-gram
        corpus_fname: corpus file path
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
        """

        self.n = n
        self.size = size
        self.fasta_fname = fasta_fname

        if corpus is None and fasta_fname is None:
            raise Exception("Either fasta_fname or corpus is needed!")

        if fasta_fname is not None:
            print('Generate Corpus file from fasta file...')
            generate_corpusfile(fasta_fname, n, corpus_fname)
            corpus = word2vec.Text8Corpus(corpus_fname)

        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)

    def to_vecs(self, seq):
        """
        convert sequence to three n-length vectors
        e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
        """
        ngram_patterns = split_ngrams(seq, self.n)

        protvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self.wv[ngram])
                except:
                    raise Exception("Model has never trained this n-gram: " + ngram)
            protvecs.append(sum(ngram_vecs))
        return protvecs
    
    
    def get_vector(self, seq):
        """
        sum and normalize the three n-length vectors returned by self.to_vecs
        """
        #return normalize(sum(self.to_vecs(seq)))
        return sum(self.to_vecs(seq))

    
def load_protvec(model_fname):
    return word2vec.Word2Vec.load(model_fname)

pv = load_protvec('tools/Embeddings/swissprot_size200_window25.model')



def create_features(df):
    df['Sequence_length'] = df['sequence_final'].str.len()
    df['LCR_frac'] = [extract_LCR(seq)[0] for seq in df['sequence_final']] / df['Sequence_length']
    df['LCR_sequence'] = [extract_LCR(seq)[1] for seq in df['sequence_final']]
    df['LCR_length'] = df['LCR_sequence'].str.len()
    df['Hydrophobicity'] = [hydrophobicity(seq) for seq in df['sequence_final']]
    df['Shannon_entropy'] = [Shannon_entropy(seq) for seq in df['sequence_final']]
    df['IDR_frac'] = [extract_IDR(seq) for seq in df['sequence_final']] / df['Sequence_length']
    df['pI'] = [IP(seq).pi() for seq in df['sequence_final']]
    #df['IDR_frac'] = np.where(df['Sequence_length'] < 25, 1, df['IDR_frac'])
 
    #Types of amino acids
    Hydrophobic_AAs = ['A', 'I', 'L', 'M', 'F', 'V']
    Polar_AAs = ['S', 'Q', 'N', 'G', 'C', 'T', 'P']
    Cation_AAs = ['K', 'R', 'H']
    Anion_AAs = ['D', 'E']
    Arom_AAs = ['W', 'Y', 'F']

    # Fractions of the 20 different AAs in the full sequence
    for k in range(0, len(AA_array)):
        df['AA_' + str(AA_array[k])] = [get_AA_count(seq, str(AA_array[k])) for seq in df['sequence_final']]

    df['Polar'] = 0; df['Cation'] = 0; df['Anion'] = 0; df['Arom'] = 0; df['HB'] = 0;
    for i in range(0, len(Hydrophobic_AAs)):
        df['HB']  = df['HB']  + df['AA_' + str(Hydrophobic_AAs[i])]
    df['HB_frac'] = df['HB']  / df["Sequence_length"] 
    for i in range(0, len(Polar_AAs)):
        df['Polar']  = df['Polar']  + df['AA_' + str(Polar_AAs[i])]
    df['Polar_frac'] = df['Polar']  / df["Sequence_length"] 
    for i in range(0, len(Arom_AAs)):
        df['Arom']  = df['Arom']  + df['AA_' + str(Arom_AAs[i])]
    df['Arom_frac'] = df['Arom']  / df["Sequence_length"] 
    for i in range(0, len(Cation_AAs)):
        df['Cation']  = df['Cation']  + df['AA_' + str(Cation_AAs[i])]
    df['Cation_frac'] = df['Cation']  / df["Sequence_length"] 
    for i in range(0, len(Anion_AAs)):
        df['Anion']  = df['Anion']  + df['AA_' + str(Anion_AAs[i])]
    df['Anion_frac'] = df['Anion']  / df["Sequence_length"] 

    df['HB_frac'] = df['HB_frac'].replace(np.nan, 0)
    df['Polar_frac'] = df['Polar_frac'].replace(np.nan, 0)
    df['Arom_frac'] = df['Arom_frac'].replace(np.nan, 0)
    df['Cation_frac'] = df['Cation_frac'].replace(np.nan, 0)
    df['Anion_frac'] = df['Anion_frac'].replace(np.nan, 0)
    
    # Fractions of the 20 different AAs in the LCR:
    for k in range(0, len(AA_array)):
        df['AA_LCR_' + str(AA_array[k])] = [get_AA_count(seq, str(AA_array[k])) for seq in df['LCR_sequence']]

    df['Polar_LCR'] = 0; df['Cation_LCR'] = 0; df['Anion_LCR'] = 0; df['Arom_LCR'] = 0; df['HB_LCR'] = 0;

    for i in range(0, len(Hydrophobic_AAs)):
        df['HB_LCR']  = df['HB_LCR']  + df['AA_LCR_' + str(Hydrophobic_AAs[i])]
    df['HB_LCR_frac'] = df['HB_LCR']  / df["LCR_length"] 
    for i in range(0, len(Polar_AAs)):
        df['Polar_LCR']  = df['Polar_LCR']  + df['AA_LCR_' + str(Polar_AAs[i])]
    df['Polar_LCR_frac'] = df['Polar_LCR']  / df["LCR_length"] 
    for i in range(0, len(Arom_AAs)):
        df['Arom_LCR']  = df['Arom_LCR']  + df['AA_LCR_' + str(Arom_AAs[i])]
    df['Arom_LCR_frac'] = df['Arom_LCR']  / df["LCR_length"] 
    for i in range(0, len(Cation_AAs)):
        df['Cation_LCR']  = df['Cation_LCR']  + df['AA_LCR_' + str(Cation_AAs[i])]
    df['Cation_LCR_frac'] = df['Cation_LCR']  / df["LCR_length"] 
    for i in range(0, len(Anion_AAs)):
        df['Anion_LCR']  = df['Anion_LCR']  + df['AA_LCR_' + str(Anion_AAs[i])]
    df['Anion_LCR_frac'] = df['Anion_LCR']  / df["LCR_length"] 

    df['HB_LCR_frac'] = df['HB_LCR_frac'].replace(np.nan, 0)
    df['Polar_LCR_frac'] = df['Polar_LCR_frac'].replace(np.nan, 0)
    df['Arom_LCR_frac'] = df['Arom_LCR_frac'].replace(np.nan, 0)
    df['Cation_LCR_frac'] = df['Cation_LCR_frac'].replace(np.nan, 0)
    df['Anion_LCR_frac'] = df['Anion_LCR_frac'].replace(np.nan, 0)
    
    Hydrophobic_AAs = ['A', 'I', 'L', 'M', 'F', 'V']
    NonHydrophobic_AAs = ['R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'K', 'P', 'S', 'T', 'W', 'Y', 'U']
    window = 5
    
    df = df.drop(columns = ['LCR_sequence'])
    df = df.reindex(columns = df.columns.tolist() + [x for x in range(0, 200)])
    df[[x for x in range(0, 200)]] = df.apply(lambda x: pd.Series(pv.get_vector(x['sequence_final'])), axis=1)
    
    return df

    
def predict_multiclass(model_name, df_predict_on):
    model = pickle.load(open('tools/Models/' + str(model_name) + '.sav', 'rb'))
    trial = df_predict_on.drop(['sequence_final'], axis=1).to_numpy()
    df_predictions =  df_predict_on.copy()
    df_predictions['prediction'] = model.predict_proba(trial)[:,0] + 0.5* model.predict_proba(trial)[:,1]
    df_predictions = df_predictions.rename(columns={"prediction": "prediction_" + str(model_name)})
    
    return df_predictions
    
    
def DeePhase(df_of_sequences):
    
    data_interm = create_features(df_of_sequences)
    w2v_cols = data_interm.columns[[str(col).isdigit() for col in data_interm.columns]]
    w2v_cols = [int(x) for x in w2v_cols]
    data_w2v = data_interm[w2v_cols]
    info_cols = ['sequence_final']
    data_w2v = pd.concat([data_w2v, data_interm[info_cols]], axis = 1)
    data_w2v.columns = [str(x) for x in data_w2v]
    data_w2v = data_w2v.reindex(sorted(data_w2v.columns), axis=1)

    phys_feature_cols_temp = list(np.setdiff1d(list(data_interm.columns), info_cols))
    phys_feature_cols = list(np.setdiff1d(phys_feature_cols_temp, w2v_cols))
    data_phys = data_interm[phys_feature_cols]
    data_info = data_interm[info_cols]
    data_phys = pd.concat([data_phys, data_interm[info_cols]], axis = 1)
    data_phys_sel = data_phys[{'Hydrophobicity', 'Shannon_entropy', 'LCR_frac', 'IDR_frac',
             #'Polar_frac',
             'Arom_frac', 'Cation_frac', 'sequence_final'
             }]
    data_phys_sel = data_phys_sel.reindex(sorted(data_phys_sel.columns), axis=1)

    data_interm['phys_multi'] = predict_multiclass('phys_multi', data_phys_sel)['prediction_phys_multi']
    data_interm['w2v_multi'] = predict_multiclass('w2v_multi', data_w2v)['prediction_w2v_multi']
    data_interm['DeePhase'] = 0.5*(data_interm['phys_multi']  + data_interm['w2v_multi'] )

    return  str('DeePhase score: ') + str(round(data_interm['DeePhase'].item(), 3))