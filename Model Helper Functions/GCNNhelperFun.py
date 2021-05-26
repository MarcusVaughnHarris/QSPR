#These functions use code from https://towardsdatascience.com/drug-discovery-with-graph-neural-networks-part-1-1011713185eb

def GCNN_Model_Creator(dataset_file, task_name , smiles_field, epochs, batchSize):
  loader = dc.data.CSVLoader(tasks=[task_name],  smiles_field=smiles_field, featurizer=dc.feat.ConvMolFeaturizer())
  dataset = loader.featurize(dataset_file) # Featurizing the dataset with ConvMolFeaturizer
  splitter = dc.splits.RandomSplitter() # Same as train_test_split from sklearn
  train, valid, _ = splitter.train_valid_test_split(dataset, frac_train=0.7, frac_valid=0.29, frac_test=0.01) # frac_test is 0.01 because we only use a train and valid as an example
  normalizer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train,move_mean=True)

  struc_data = normalizer.transform(dataset)
  train = normalizer.transform(train)
  test = normalizer.transform(valid)

  GCNN = GraphConvModel(1, batch_size=batchSize, mode="regression")
  GCNN.fit(train, nb_epoch=epochs) # Fitting the model

  metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
  train_scores = GCNN.evaluate(train, [metric], [normalizer]) 
  valid_scores = GCNN.evaluate(valid, [metric], [normalizer])

  # Predicted values for entire dataset
  cmc_pred = GCNN.predict_on_batch(struc_data.X)
  train_smiles = list(struc_data.ids)
  predicted_prop_id = "{}{}".format('GCNN_',task_name)
  measured_prop_id = "{}{}".format('Measured_',task_name)
  model_measured_pred = pd.DataFrame({'smiles': train_smiles, 
                                      measured_prop_id: [x for l in normalizer.untransform(struc_data.y) for x in l], 
                                      predicted_prop_id:[ round(val,3) for val in     [x for l in normalizer.untransform(cmc_pred) for x in l]] })
  model_measured_pred["Percent_Error"] = round(((model_measured_pred[predicted_prop_id] - model_measured_pred[measured_prop_id])/model_measured_pred[measured_prop_id])*100, 2) # Calc percent Error

  # Predict/plot on trainvset and test set 
  plt.scatter([x for l in normalizer.untransform(train.y) for x in l],
              GCNN.predict_on_batch(train.X), 
              label = 'Train_data',    c='blue') #Plotting training data measured vs predicted

  plt.scatter([x for l in normalizer.untransform(test.y) for x in l], 
              GCNN.predict_on_batch(test.X), 
              c='lightgreen', label='Test_data', alpha = 0.8) #Adding test data measured vs predicted
  plt.title('Model Performance (Train and Test Set)')
  plt.xlabel(measured_prop_id)
  plt.ylabel(predicted_prop_id)
  plt.legend(loc=4)
  plt.show()

  print(train_scores)
  print(valid_scores)
  print('Total MAE:' , round(sum(abs(model_measured_pred.Percent_Error))/len(model_measured_pred),3), '%') # Printing MAE
  return GCNN, model_measured_pred


#_________________________________________________________________________________________________________________________ 

def GraphCNN_Model_Predict_v1(dataset,  training_dataset, model, task_name, smiles_field = "smiles"):

  dataset[task_name]= (np.random.rand( dataset.shape[0]))*max(training_dataset[task_name])
  dataset.to_csv('/content/GM_TopSim.csv')
  dataset_file = '/content/GM_TopSim.csv'

  loader = dc.data.CSVLoader(tasks=[task_name],  smiles_field="smiles", featurizer=dc.feat.ConvMolFeaturizer())
  data = loader.featurize(dataset_file) # Featurizing the dataset with ConvMolFeaturizer
  normalizer = dc.trans.NormalizationTransformer(transform_y=True, dataset=data,move_mean=True)
  struc_data = normalizer.transform(data)
  cmc_pred = model.predict_on_batch(struc_data.X)
  cmc_pred= normalizer.untransform(cmc_pred)
  cmc_pred_list = [x for l in cmc_pred for x in l]
  
  predicted_prop_id = "{}{}".format('GCNN_',task_name)

  train_smiles = list(struc_data.ids)
  len(train_smiles)
  model_measured_pred = pd.DataFrame({'smiles': train_smiles})
  model_measured_pred[predicted_prop_id] = cmc_pred_list #Adding predicted CMC to original Dataframe

  model_measured_pred[predicted_prop_id] = round(model_measured_pred[predicted_prop_id], 3)

  GenMols_pred = dataset.merge(model_measured_pred, left_on='smiles', right_on='smiles')#____________________________________________ Merge model predictions
  return GenMols_pred

def GraphCNN_Model_Predict(dataset,  training_dataset, model, task_name, smiles_field = "smiles"):

  dataset[task_name]= (np.random.rand( dataset.shape[0]))*max(training_dataset[task_name])
  dataset.to_csv('/content/GM_TopSim.csv')
  dataset_file = '/content/GM_TopSim.csv'

  loader = dc.data.CSVLoader(tasks=[task_name],  smiles_field="smiles", featurizer=dc.feat.ConvMolFeaturizer())
  data = loader.featurize(dataset_file) # Featurizing the dataset with ConvMolFeaturizer
  normalizer = dc.trans.NormalizationTransformer(transform_y=True, dataset=data,move_mean=True)
  struc_data = normalizer.transform(data)
  cmc_pred = model.predict_on_batch(struc_data.X)
  cmc_pred= normalizer.untransform(cmc_pred)
  cmc_pred_list = [x for l in cmc_pred for x in l]
  
  predicted_prop_id = "{}{}".format('GCNN_',task_name)

  train_smiles = list(struc_data.ids)
  len(train_smiles)
  model_measured_pred = pd.DataFrame({'smiles': train_smiles})
  model_measured_pred[predicted_prop_id] = cmc_pred_list #Adding predicted CMC to original Dataframe

  model_measured_pred[predicted_prop_id] = round(model_measured_pred[predicted_prop_id], 3)
  GenMols_pred = dataset.merge(model_measured_pred, left_on='smiles', right_on='smiles')#____________________________________________ Merge model predictions
  
  return GenMols_pred.drop([task_name], axis=1)
