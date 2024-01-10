import json,pickle
from rdkit import Chem
import numpy as np
from collections import OrderedDict
import os


# from DeepDTA data 
all_prots = []
#datasets = ['pdb','bindingdb']
datasets = ['kiba','davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold_origin = json.load(open(fpath + "folds/train_fold_setting1.txt"))  # len=5
    train_fold = [ee for e in train_fold_origin for ee in e ] # davis len=25046
    train_fold_origin = [e for e in train_fold_origin]  # 5-fold
    test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))  # davis len=5010
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)  # davis len=68
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)  # davis len=442
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')  # davis len=68
    drugs = []
    prots = []
    for d in ligands.keys():
         # lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        lg = ligands[d]
        drugs.append(lg)  # loading drugs
    for t in proteins.keys():
        prots.append(proteins[t])  # loading proteins
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    if dataset == 'bindingdb':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)  # affinity shape=(68 drug,442 prot)


    #five-fold
    opts = ['train','test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False) 
        if opt == 'train':
            rows,cols = rows[train_fold], cols[train_fold] 
            for fold in range(5): #五折
                train_rows, train_cols = np.where(np.isnan(affinity) == False)  # not NAN
                valid_rows, valid_cols = np.where(np.isnan(affinity) == False)  # not NAN
                train_folds = []
                valid_fold = train_fold_origin[fold]  # specify 1 fold as valid set
                for i in range(len(train_fold_origin)):
                    if i != fold:
                        train_folds += train_fold_origin[i]
                train_rows, train_cols = train_rows[train_folds], train_cols[train_folds] 
                valid_rows, valid_cols = valid_rows[valid_fold], valid_cols[valid_fold]

                if not os.path.isdir("./data/processed/"):
                    os.mkdir("./data/processed/")
                with open('data/processed/' + dataset + '_' + opt + '.csv', 'w') as f:
                    f.write('compound_iso_smiles,target_sequence,affinity\n')
                    for pair_ind in range(len(rows)):
                        ls = []
                        ls += [ drugs[rows[pair_ind]]  ]
                        ls += [ prots[cols[pair_ind]]  ]
                        ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                        f.write(','.join(map(str,ls)) + '\n')
                    
                with open('data/processed/' + dataset + '_fold-' + str(fold+1) + '_train.csv', 'w') as f:
                    f.write('compound_iso_smiles,target_sequence,affinity\n')
                    for pair_ind in range(len(train_rows)):
                        ls = []
                        ls += [drugs[train_rows[pair_ind]]]
                        ls += [prots[train_cols[pair_ind]]]
                        ls += [affinity[train_rows[pair_ind], train_cols[pair_ind]]]
                        f.write(','.join(map(str, ls)) + '\n')  # csv format

                with open('data/processed/' + dataset + '_fold-' + str(fold+1) + '_valid.csv', 'w') as f:
                    f.write('compound_iso_smiles,target_sequence,affinity\n')
                    for pair_ind in range(len(valid_rows)):
                        ls = []
                        ls += [drugs[valid_rows[pair_ind]]]
                        ls += [prots[valid_cols[pair_ind]]]
                        ls += [affinity[valid_rows[pair_ind], valid_cols[pair_ind]]]
                        f.write(','.join(map(str, ls)) + '\n')  # csv format

                print('train_fold_' + str(fold+1) + ':', len(train_folds))
                print('valid_fold_' + str(fold+1) + ':', len(valid_fold))
                
            print('\ndataset:', dataset)
            print('train_fold:', len(train_fold))
            

        elif opt == 'test':
            rows, cols = np.where(np.isnan(affinity) == False)  # not NAN
            rows, cols = rows[test_fold], cols[test_fold]
            with open('data/processed/' + dataset + '_test.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [ drugs[rows[pair_ind]]  ]
                    ls += [ prots[cols[pair_ind]]  ]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    f.write(','.join(map(str,ls)) + '\n')  # csv format
    print('test_fold:', len(test_fold))

    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))
    all_prots += list(set(prots))
    