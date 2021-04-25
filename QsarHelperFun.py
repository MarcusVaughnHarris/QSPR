#--------- 1. Calculate descriptors for smiles dataframe (SMIdf) given a specified fingerprint type and descriptor list
def CalcDescr_SMIdf(SMIdf, FingerPrintType, DescriptorList ):
  SMIdf['Mol'] = [Chem.MolFromSmiles(SMILE) for SMILE in SMIdf.smiles] #add 'Mol' Objects to Dataframe
  fp = "{}{}".format(FingerPrintType,'(mol)[0]') 
  descriptors = ["{}{}".format(i,'(mol)') for i in DescriptorList]
  fp_desc_str = (fp ,descriptors)
  desc_expression = "{}{}".format('np.append',fp_desc_str) 
  def descriptor_fun(mol): 
    return eval(desc_expression.replace("'", ''))
  SMIdf['Descriptors'] = SMIdf['Mol'].apply(descriptor_fun)#Calculate Descriptors using pre-defined function
  return SMIdf


#--------- 2. Preprocess SMIdf with descriptors (SMIdf_descr) to use for training a model
def PreProcess_SMIdf(SMIdf_descr, property_id, TestSize = 0.25):
# PRE-PROCESSING DATA 
  Desc_array = np.array(list(SMIdf_descr['Descriptors'])) #Creating NUMPY array for descriptors  (panda cant handle datasets of molecular fingerprints)
  st = StandardScaler() #Scale X to unit variance and zero mean
  Desc_array_scaled= st.fit_transform(Desc_array)
  Prop_array = SMIdf_descr[property_id].values # Adding '.values' makes the properties into an array
# SPLITTING DATA INTO TRAINING/TESTING SETS 
  Desc_train,Desc_test,Prop_train,Prop_test= train_test_split(Desc_array_scaled,Prop_array,test_size=TestSize,random_state=42)#Split the DESCRIPTORS arrays and the property values into training and testing sets using 'train_test_split'
  return Desc_train,Desc_test,Prop_train,Prop_test


#--------- 3. Predict property using defined model and a SMIdf_descr
def ModelPredict_SMIdf(SMIdf_descr, model): #______________________________________________________GOOD
  smi_t2_np_array = np.array(list(SMIdf_descr['Descriptors'])) #Convert to Numpy arrays!! (panda cant handle datasets of molecular fingerprints)
  st = StandardScaler()
  smi_t2_np_array_scaled = st.fit_transform(smi_t2_np_array) # Scale array
  smi_t2_predicted_array = model.predict(smi_t2_np_array_scaled) # Predict using
  smi_t2_predicted = [x for l in smi_t2_predicted_array for x in l] #Unlisting list of lists
  SMIdf_descr["predicted_prop"] = smi_t2_predicted #Adding predicted CMC to original Dataframe
  return SMIdf_descr


#--------- 4. Predict prop using SMIdf_descr and show stats for specific functional group subsets in the data
def Substructure_model_performance(SMIdf_descr, property_id, model, mol_substructure):   #______________________________________________________GOOD
  cmc_pred = ModelPredict_SMIdf(SMIdf_descr, model ) # Predicting CMC using dataframe

  measured_prop_id = "{}{}".format('Measured_',property_id) 
  predicted_prop_id = "{}{}".format('Predicted_',property_id) 

  model_measured_pred = pd.DataFrame({'smiles': cmc_pred.smiles, # Trimming dataframe to make it easy to see
                                    'Mol': cmc_pred.Mol,  
                                    measured_prop_id: cmc_pred[property_id], 
                                    predicted_prop_id: cmc_pred['predicted_prop']})
  model_measured_pred["Percent_Error"] = (abs(model_measured_pred[predicted_prop_id] - model_measured_pred[measured_prop_id])/model_measured_pred[measured_prop_id])*100 # Calc percent Error
  print('Total MAE:' , sum(model_measured_pred.Percent_Error)/len(model_measured_pred), '%') # Printing MAE

  def FG_subset(df_mols, mol_substructure):
    df_mols["Has_FG"] = [mol.HasSubstructMatch(mol_substructure) for mol in df_mols.Mol]
    data_with_mol_substructure = pd.DataFrame(df_mols[df_mols['Has_FG'] == True])
    return data_with_mol_substructure

  model_measured_pred_subset= FG_subset(model_measured_pred, mol_substructure)
  print('Substructure MAE:' , sum(model_measured_pred_subset.Percent_Error)/len(model_measured_pred_subset), '%') # Printing MAE
  Abs_error = ["{}{}".format(i,'% error') for i in [str(mol) for mol in [round(num, 2) for num in list(model_measured_pred_subset['Percent_Error'].values)]]]
  model_measured_pred_subset['Abs_error'] = Abs_error
  return model_measured_pred_subset
