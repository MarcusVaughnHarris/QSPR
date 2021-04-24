#====== FUNCTIONS/SMARTS =================================================================================
# Splitting Data by Functional Group 
def FG_split(list, group, keep_group = True):
    has_group = [mol.HasSubstructMatch(group) for mol in list]
    has_group_check = pd.DataFrame({'mols':list,'Group':has_group})
    HAS_group = pd.DataFrame(has_group_check[has_group_check.Group == 1], columns= ['mols','group'])
    DOESNT_HAVE_group = pd.DataFrame(has_group_check[has_group_check.Group == 0], columns= ['mols','group'])
    if keep_group == True:
      return HAS_group.mols
    else:
      return DOESNT_HAVE_group.mols

def NHOH_limit(mols, limit_to = 4):
 mol_data = pd.DataFrame({'Mol': mols})
 mol_data["NH_OH_Count"] = [Descriptors.NumHDonors(mol) for mol in mol_data.Mol]
 NH_OH_lim = Nucleophilic_Reactants[Nucleophilic_Reactants.NH_OH_Count <= limit_to]
 return NH_OH_lim.Mol

#====== RXN FUNCTIONS ====================================================================================
#-------------------------------------------------------------------------------
# 1.) Epoxidation (mCPBA in DCM, 88% YIELD) [1] 
#-------------------------------------------------------------------------------
def Epoxidation(molecule_list):
    RXN = AllChem.ReactionFromSmarts('[!c;C:1]=[C:2]>[H][O][O][H]>[C:1]1[C:2]O1') # 1 component
    all_products_tuples = tuple(RXN.RunReactants((mol,)) for mol in molecule_list) # tuple format
    all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
    try:
      [Chem.SanitizeMol(mol) for mol in all_products] 
    except ValueError:
      print("One or more Epoxidation products failed sanitization, removing empty mols.")
    all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
    all_products_unique = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)] #list format
    return all_products_unique
#-------------------------------------------------------------------------------
#2.) Cyclic_Carbonation
#-------------------------------------------------------------------------------
def Cyclic_Carbonation(molecule_list):
    RXN = AllChem.ReactionFromSmarts('[C:1]1[C:2][O:3]1>>O=C1O[C:1][C:2][O:3]1')# 1 component
    all_products_tuples = tuple(RXN.RunReactants((mol,)) for mol in molecule_list) # tuple format
    all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
    try:
      [Chem.SanitizeMol(mol) for mol in all_products] 
    except ValueError:
      print("One or more epoxide_to_CC5 products failed sanitization, removing empty mols.")
    all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
    all_products_unique = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)] #list format
    return all_products_unique
#-------------------------------------------------------------------------------
#3.) Nuc_ring_opening [O,N;D1:5]>>[O,N:5]
#-------------------------------------------------------------------------------
def Nuc_ring_opening(data, Nucleophiles, return_format):
    # epoxide_opening [O,N;D1:5]>>[O,N:5]
    def epoxide_opening(molecule_list, Nucleophile):
      RXN = AllChem.ReactionFromSmarts('[#6:1]-1-[#6:2]-[#8:3]-1.[O,N;D1:4]>>[O,N:4][#6:2]-[#6:1]-[#8:3]') # 2 component
      all_products_tuples = tuple(RXN.RunReactants((mol, Nucleophile)) for mol in molecule_list) # tuple format
      all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
      try:
        [Chem.SanitizeMol(mol) for mol in all_products] 
      except ValueError:
        print("One or more epoxide_opening products failed sanitization, removing empty mols.")
      all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
      all_products_unique = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)] #list format
      return all_products_unique
    # CC5_opening [O,N;D1:5]>>[O,N:5]
    def CC5_opening(molecule_list, Nucleophile):
      RXN=AllChem.ReactionFromSmarts('[O:6]=[#6:5]-1-[#8:1]-[#6:2]-[#6:3]-[#8:4]-1.[O,N;D1:7]>>[O,N:7][#6:5](=[O:6])-[#8:4]-[#6:3]-[#6:2]-[#8:1]') # 2 component
      all_products_tuples = tuple(RXN.RunReactants((mol, Nucleophile)) for mol in molecule_list) # tuple format
      all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
      try:
        [Chem.SanitizeMol(mol) for mol in all_products] 
      except ValueError:
        print("One or more CC5_opening products failed sanitization, removing empty mols.")
      all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
      all_products_unique = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)] #list format
      return all_products_unique
    data_ep = FG_split(data, epoxy_smarts, keep_group = True) #Seperating epoxides
    data_CC5 = FG_split(data, CC5_smarts, keep_group = True) #Seperating CC5's
    data_ep = [epoxide_opening(data_ep, n) for n in Nucleophiles] # Reacting epoxides
    data_ep = [x for l in data_ep for x in l] #Unlisting list of lists
    data_ep = [x for x in data_ep if x is not None] #Removing mols that failed sanitization
    data_CC5 = [CC5_opening(data_CC5, n) for n in Nucleophiles] # Reacting CC5's
    data_CC5 = [x for l in data_CC5 for x in l] #Unlisting CC5_products
    data_CC5 = [x for x in data_CC5 if x is not None] #Removing mols that failed sanitization
    data_ep_CC5 = [data_ep, data_CC5] #Combining CC5_products & epoxide_products
    data_products = [x for l in data_ep_CC5 for x in l] #Creating single list
    data_products = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in data_products] #list format
    Mol_products_unique = [Chem.MolFromSmiles(smiles) for smiles in set(data_products)] #list format
    Smiles_products_unique = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in Mol_products_unique] #list format
    if return_format == 'Mol':
      return Mol_products_unique
    elif return_format == 'Smiles':
      return Smiles_products_unique
    else:
      return pd.DataFrame({'Mol':Mol_products_unique,'Smiles':Smiles_products_unique})
#-------------------------------------------------------------------------------
#4.) fame_transesterifications
#-------------------------------------------------------------------------------
def fame_transesterifications(data, Nucleophiles, return_format):
  # fame_transest_pre_fun
   def fame_transest_pre_fun(data, Nucleophile):
      RXN = AllChem.ReactionFromSmarts('[#6:1]-[#6:2](-[#8:3])=[O:4].[O,N;D1:5]>>[#6:1]-[#6:2](-[O,N:5])=[O:4]')
      all_products_tuples = tuple(RXN.RunReactants((mol, Nucleophile)) for mol in data) # tuple format
      all_products = list(chain.from_iterable(chain.from_iterable(all_products_tuples))) # list format
      try:
        [Chem.SanitizeMol(mol) for mol in all_products] 
      except ValueError:
        print("One or more transesterification products failed sanitization, removing empty mols.")
      all_products_smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in all_products] #list format
      all_products_unique = [Chem.MolFromSmiles(smiles) for smiles in set(all_products_smiles)] #list format
      return all_products_unique
   data_te = [fame_transest_pre_fun(data, n) for n in Nucleophiles] # Transesterifications
   data_te = [x for l in data_te for x in l] #Unlisting list of lists
   data_te = [x for x in data_te if x is not None] #Removing mols that failed sanitization
   data_products = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in data_te] #list format
   Mol_products_unique = [Chem.MolFromSmiles(smiles) for smiles in set(data_products)] #list format
   Smiles_products_unique = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in Mol_products_unique] #list format
   if return_format == 'Mol':
      return Mol_products_unique
   elif return_format == 'Smiles':
      return Smiles_products_unique
   else:
      return pd.DataFrame({'Mol':Mol_products_unique,'Smiles':Smiles_products_unique})
