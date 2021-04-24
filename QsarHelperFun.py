#========== FUNCTIONS ===============
#----------- 1. Calculate and return an array of descriptors
def FP_other_Descriptors(mol): #______________________________________________________GOOD
  return np.append(Fingerprinter.FingerprintMol(mol)[0],[Descriptors.MolWt(mol),Descriptors.TPSA(mol),Descriptors.NOCount(mol),Descriptors.NumHAcceptors(mol),Descriptors.NumHDonors(mol),Descriptors.NHOHCount(mol),Descriptors.NumValenceElectrons(mol),Descriptors.NumRotatableBonds(mol),Descriptors.MolLogP(mol),Descriptors.MolMR(mol)]) 

#----------- 3. Predict prop using smiles dataset
def ModelPredict_SMIdf(df, model, descriptor_fun): #______________________________________________________GOOD
  df['Mol'] = [Chem.MolFromSmiles(SMILE) for SMILE in df.smiles] #Add Mol object column
  df['Descriptors'] = df['Mol'].apply(descriptor_fun) 
  smi_t2_np_array = np.array(list(df['Descriptors'])) #Convert to Numpy arrays!! (panda cant handle datasets of molecular fingerprints)
  st = StandardScaler()
  smi_t2_np_array_scaled = st.fit_transform(smi_t2_np_array) # Scale array
  smi_t2_predicted_array = model.predict(smi_t2_np_array_scaled) # Predict using
  smi_t2_predicted = [x for l in smi_t2_predicted_array for x in l] #Unlisting list of lists
  df["predicted_CMC"] = smi_t2_predicted #Adding predicted CMC to original Dataframe
  return df


#----------- 4. Predict prop using smiles dataset and show stats
def FG_model_performance(df, property_id, model, descriptor_fun, group):   #______________________________________________________GOOD
  cmc_pred = ModelPredict_SMIdf(df, model, descriptor_fun ) # Predicting CMC using dataframe
  model_measured_pred = pd.DataFrame({'smiles': cmc_pred.smiles, # Trimming dataframe to make it easy to see
                                    'Mol': cmc_pred.Mol,  
                                    'measured_cmc': cmc_pred[property_id], 
                                    'predicted_cmc': cmc_pred['predicted_CMC']})
  model_measured_pred["Percent_Error"] = (abs(model_measured_pred.predicted_cmc - model_measured_pred.measured_cmc)/model_measured_pred.measured_cmc)*100 # Calc percent Error
  print('Total MAE:' , sum(model_measured_pred.Percent_Error)/len(model_measured_pred), '%') # Printing MAE

  def FG_subset(df_mols, group):
    df_mols["Has_FG"] = [mol.HasSubstructMatch(group) for mol in df_mols.Mol]
    data_with_group = pd.DataFrame(df_mols[df_mols['Has_FG'] == True])
    return data_with_group

  model_measured_pred_phytanyl_subset= FG_subset(model_measured_pred, group)
  print('Group MAE:' , sum(model_measured_pred_phytanyl_subset.Percent_Error)/len(model_measured_pred_phytanyl_subset), '%') # Printing MAE
  Abs_error = ["{}{}".format(i,'% error') for i in [str(mol) for mol in [round(num, 2) for num in list(model_measured_pred_phytanyl_subset['Percent_Error'].values)]]]
  return Draw.MolsToGridImage(model_measured_pred_phytanyl_subset.Mol, molsPerRow=8, subImgSize= (250,250), legends = Abs_error)


def PrepSplit_SMIdf(df, property_id, descriptor_fun, TestSize = 0.25):
  NonIonicSurfs['Mol'] = [Chem.MolFromSmiles(SMILE) for SMILE in NonIonicSurfs.smiles] #add 'Mol' Objects to Dataframe
#========= CALC DESCRIPTORS =========
  NonIonicSurfs['Descriptors'] = NonIonicSurfs['Mol'].apply(descriptor_fun)#Calculate Descriptors using pre-defined function
#======= PRE-PROCESSING DATA ========
  Desc_array = np.array(list(NonIonicSurfs['Descriptors'])) #Creating NUMPY array for descriptors  (panda cant handle datasets of molecular fingerprints)
  st = StandardScaler() #Scale X to unit variance and zero mean
  Desc_array_scaled= st.fit_transform(Desc_array)
  Prop_array = NonIonicSurfs[property_id].values # Adding '.values' makes the properties into an array
#==== SPLITTING DATA INTO TRAINING/TESTING SETS ======
  Desc_train,Desc_test,Prop_train,Prop_test= train_test_split(Desc_array_scaled,Prop_array,test_size=TestSize,random_state=42)#Split the DESCRIPTORS arrays and the property values into training and testing sets using 'train_test_split'
  return Desc_train,Desc_test,Prop_train,Prop_test