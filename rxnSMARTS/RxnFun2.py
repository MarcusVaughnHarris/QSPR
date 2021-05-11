
#-------------------------------------------------------------------------------
# 1.) Epoxidation (mCPBA in DCM, 88% YIELD) [1] 
#-------------------------------------------------------------------------------
def Epoxidation(molecule_list):
    RDLogger.DisableLog('rdApp.*')
    RXN = AllChem.ReactionFromSmarts('[!c;C:1]=[C:2]>[H][O][O][H]>[C:1]1[C:2]O1') # 1 component
    all_products_tuples = tuple(RXN.RunReactants((mol,)) for mol in molecule_list.Mol) # tuple format
    all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
    try:
      [Chem.SanitizeMol(mol) for mol in all_products] 
    except ValueError:
      print("One or more Epoxidation products failed sanitization, removing empty mols.")
    all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
    all_products_mols = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)]
    all_products_unique = pd.DataFrame({"Mol": all_products_mols, #list format
                                        "smiles": [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products_mols],
                                        "rxnSeq": ">> epox"}) #list format
    df_name =[x for x in globals() if globals()[x] is molecule_list][0]
    column_name_list = ' '.join(molecule_list.columns.tolist())
    if "rxnSeq" in column_name_list:
      all_products_unique["rxnSeq"] = ["{}{}".format(str(molecule_list.rxnSeq[0]), i   ) for i in all_products_unique.rxnSeq]
    else:
      all_products_unique["rxnSeq"] = ["{}{}".format(df_name, i   ) for i in all_products_unique.rxnSeq]
    return all_products_unique

#-------------------------------------------------------------------------------
#2.) Cyclic_Carbonation
#-------------------------------------------------------------------------------
def Cyclic_Carbonation(molecule_list):
    RDLogger.DisableLog('rdApp.*')
    RXN = AllChem.ReactionFromSmarts('[C:1]1[C:2][O:3]1>>O=C1O[C:1][C:2][O:3]1')# 1 component
    all_products_tuples = tuple(RXN.RunReactants((mol,)) for mol in molecule_list.Mol) # tuple format
    all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
    try:
      [Chem.SanitizeMol(mol) for mol in all_products] 
    except ValueError:
      print("One or more epoxide_to_CC5 products failed sanitization, removing empty mols.")
    all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
    all_products_mols = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)]
    all_products_unique = pd.DataFrame({"Mol": all_products_mols, #list format
                                        "smiles": [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products_mols],
                                        
                                        "rxnSeq": ">> cc5"}) #list format

    df_name =[x for x in globals() if globals()[x] is molecule_list][0]
    column_name_list = ' '.join(molecule_list.columns.tolist())
    if "rxnSeq" in column_name_list:
      all_products_unique["rxnSeq"] = ["{}{}".format(str(molecule_list.rxnSeq[0]), i   ) for i in all_products_unique.rxnSeq]
    else:
      all_products_unique["rxnSeq"] = ["{}{}".format(df_name, i   ) for i in all_products_unique.rxnSeq]
    return all_products_unique

#-------------------------------------------------------------------------------
#3.) Nuc_ring_opening [O,N;D1:5]>>[O,N:5]
#-------------------------------------------------------------------------------

def Nuc_ring_opening(data, Nucleophiles):
#____________________________________________________________________________________________________________________________________
  def epoxide_opening(data, Nucleophile):
      RXN = AllChem.ReactionFromSmarts('[#6:1]-1-[#6:2]-[#8:3]-1.[O,N;D1:4]>>[O,N:4][#6:2]-[#6:1]-[#8:3]')
      all_products_tuples = tuple(RXN.RunReactants((mol, Nucleophile)) for mol in data.Mol) # tuple format
      all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
      try:
        [Chem.SanitizeMol(mol) for mol in all_products] 
      except ValueError:
        dog = 42
      all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
      all_products_mols = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)]
      return all_products_mols
#____________________________________________________________________________________________________________________________________
  def CC5_opening(data, Nucleophile):
      RXN = AllChem.ReactionFromSmarts('[O:6]=[#6:5]-1-[#8:1]-[#6:2]-[#6:3]-[#8:4]-1.[O,N;D1:7]>>[O,N:7][#6:5](=[O:6])-[#8:4]-[#6:3]-[#6:2]-[#8:1]')
      all_products_tuples = tuple(RXN.RunReactants((mol, Nucleophile)) for mol in data.Mol) # tuple format
      all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
      try:
        [Chem.SanitizeMol(mol) for mol in all_products] 
      except ValueError:
        dog = 42
      all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
      all_products_mols = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)]
      return all_products_mols
#____________________________________________________________________________________________________________________________________
  RDLogger.DisableLog('rdApp.*')
  all_products_mols_ep = [epoxide_opening(data, n) for n in Nucleophiles.Mol] # Reacting FAMES
  allProd_ep = [x for l in all_products_mols_ep for x in l] #Unlisting list of lists
  allProd_ep_unlist = [x for x in allProd_ep if x is not None] #Removing mols that failed sanitization
  all_products_ep_unique = pd.DataFrame({"Mol": allProd_ep_unlist, #list format
                                          "smiles": [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in allProd_ep_unlist],                                        
                                          "rxnSeq": ">> epoxOpening"}) #list format
  df_name =[x for x in globals() if globals()[x] is data][0]
  column_name_list = ' '.join(data.columns.tolist())
  if "rxnSeq" in column_name_list:
        all_products_ep_unique["rxnSeq"] = ["{}{}".format(str(data.rxnSeq[0]), i   ) for i in all_products_ep_unique.rxnSeq]
  else:
        all_products_ep_unique["rxnSeq"] = ["{}{}".format(df_name, i   ) for i in all_products_ep_unique.rxnSeq]
#____________________________________________________________________________________________________________________________________________________________
  all_products_mols_cc5 = [CC5_opening(data, n) for n in Nucleophiles.Mol] # Reacting FAMES
  allProd_cc5 = [x for l in all_products_mols_cc5 for x in l] #Unlisting list of lists
  allProd_cc5_unlist = [x for x in allProd_cc5 if x is not None] #Removing mols that failed sanitization
  all_products_cc5_unique = pd.DataFrame({"Mol": allProd_cc5_unlist, #list format
                                          "smiles": [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in allProd_cc5_unlist],                                        
                                          "rxnSeq": ">> cc5Opening"}) #list format
  df_name =[x for x in globals() if globals()[x] is data][0]
  column_name_list = ' '.join(data.columns.tolist())
  if "rxnSeq" in column_name_list:
        all_products_cc5_unique["rxnSeq"] = ["{}{}".format(str(data.rxnSeq[0]), i   ) for i in all_products_cc5_unique.rxnSeq]
  else:
        all_products_cc5_unique["rxnSeq"] = ["{}{}".format(df_name, i   ) for i in all_products_cc5_unique.rxnSeq]
#____________________________________________________________________________________________________________________________________________________________
  all_products_total_unique = pd.concat([all_products_ep_unique, all_products_cc5_unique])
  return all_products_total_unique

#-------------------------------------------------------------------------------
#4.) fame_reduction
#-------------------------------------------------------------------------------
def fame_reduction(molecule_list):
    RDLogger.DisableLog('rdApp.*')
    RXN = AllChem.ReactionFromSmarts('[#6:1]-[#6:2](-[#8:3])=[O:4]>>[H][#6:2](-[#6:1])-[#8:4]')# 1 component
    all_products_tuples = tuple(RXN.RunReactants((mol,)) for mol in molecule_list.Mol) # tuple format
    all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
    try:
      [Chem.SanitizeMol(mol) for mol in all_products] 
    except ValueError:
      dog = 42
    all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
    all_products_mols = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)]
    all_products_unique = pd.DataFrame({"Mol": all_products_mols, #list format
                                        "smiles": [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products_mols],
                                        
                                        "rxnSeq": ">> FAMred"}) #list format

    df_name =[x for x in globals() if globals()[x] is molecule_list][0]
    column_name_list = ' '.join(molecule_list.columns.tolist())
    if "rxnSeq" in column_name_list:
      all_products_unique["rxnSeq"] = ["{}{}".format(str(molecule_list.rxnSeq[0]), i   ) for i in all_products_unique.rxnSeq]
    else:
      all_products_unique["rxnSeq"] = ["{}{}".format(df_name, i   ) for i in all_products_unique.rxnSeq]
    return all_products_unique


#-------------------------------------------------------------------------------
#5.) fame_transesterifications
#-------------------------------------------------------------------------------
def fame_transesterifications(data, Nucleophiles):
#____________________________________________________________________________________________________________________________________
   def fame_transest_pre_fun(data, Nucleophile):
      RXN = AllChem.ReactionFromSmarts('[#6:1]-[#6:2](-[#8:3])=[O:4].[O,N;D1:5]>>[#6:1]-[#6:2](-[O,N:5])=[O:4]')
      all_products_tuples = tuple(RXN.RunReactants((mol, Nucleophile)) for mol in data.Mol) # tuple format
      all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
      try:
        [Chem.SanitizeMol(mol) for mol in all_products] 
      except ValueError:
        dog = 42
      all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
      all_products_mols = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)]
      return all_products_mols
#____________________________________________________________________________________________________________________________________
   RDLogger.DisableLog('rdApp.*')
   all_products_mols = [fame_transest_pre_fun(data, n) for n in Nucleophiles.Mol] # Reacting FAMES
   allProd = [x for l in all_products_mols for x in l] #Unlisting list of lists
   allProd_unlist = [x for x in allProd if x is not None] #Removing mols that failed sanitization

   all_products_unique = pd.DataFrame({"Mol": allProd_unlist, #list format
                                          "smiles": [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in allProd_unlist],                                        
                                          "rxnSeq": ">> FAMtransEst"}) #list format

   df_name =[x for x in globals() if globals()[x] is data][0]
   column_name_list = ' '.join(data.columns.tolist())
   if "rxnSeq" in column_name_list:
        all_products_unique["rxnSeq"] = ["{}{}".format(str(data.rxnSeq[0]), i   ) for i in all_products_unique.rxnSeq]
   else:
        all_products_unique["rxnSeq"] = ["{}{}".format(df_name, i   ) for i in all_products_unique.rxnSeq]
   #all_products_unique = pd.concat(all_products_unique)
   return all_products_unique




#---------------------------------------------------------------------------------------------------------------------------
#5.) NHOH_limit (limits NH/OH functionality in dataframe, so as to prevent over-generating products and crashing the system
#---------------------------------------------------------------------------------------------------------------------------
def NHOH_limit(MolDF, limit_to = 4):
 MolDF["NH_OH_Count"] = [Descriptors.NumHDonors(mol) for mol in MolDF.Mol]
 NH_OH_lim = MolDF[MolDF.NH_OH_Count <= limit_to]
 NHOHlimited = NH_OH_lim.drop(['NH_OH_Count'], axis=1)
 return NHOHlimited
