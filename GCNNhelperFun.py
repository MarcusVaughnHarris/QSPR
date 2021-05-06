def GCNN_Model_Creator(dataset_file, task_name , smiles_field, epochs, batchSize):
  loader = dc.data.CSVLoader(tasks=[task_name],  smiles_field=smiles_field, featurizer=dc.feat.ConvMolFeaturizer())
  dataset = loader.featurize(dataset_file) # Featurizing the dataset with ConvMolFeaturizer

  splitter = dc.splits.RandomSplitter() # Same as train_test_split from sklearn
  train, valid, _ = splitter.train_valid_test_split(dataset, frac_train=0.7, frac_valid=0.29, frac_test=0.01) # frac_test is 0.01 because we only use a train and valid as an example
  
  normalizer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train,move_mean=True)
  train = normalizer.transform(train)
  test = normalizer.transform(valid)
  metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
  GCNN = GraphConvModel(1, batch_size=batchSize, mode="regression")
  GCNN.fit(train, nb_epoch=epochs) # Fitting the model


  train_scores = GCNN.evaluate(train, [metric], [normalizer]) 
  valid_scores = GCNN.evaluate(valid, [metric], [normalizer])
  cmc_pred = GCNN.predict_on_batch(train.X)

  cmc_pred= normalizer.untransform(cmc_pred)
  cmc_pred_list = [x for l in cmc_pred for x in l]
  train_smiles = list(train.ids)
  #cmc_measured_list = [float(num) for num in train.y] 
  cmc_measured_list= normalizer.untransform(train.y)
  cmc_measured_list = [x for l in cmc_measured_list for x in l]
  
   measured_prop_id = "{}{}".format('Measured_',task_name)
   predicted_prop_id = "{}{}".format('Predicted_',task_name)

  
  model_measured_pred = pd.DataFrame({'smiles': train_smiles})
  model_measured_pred[predicted_prop_id] = cmc_pred_list #Adding predicted CMC to original Dataframe
  model_measured_pred[measured_prop_id] = cmc_measured_list #Adding measured CMC to original Dataframe


  
  model_measured_pred[measured_prop_id] = round(model_measured_pred[measured_prop_id], 3)
  model_measured_pred[predicted_prop_id] = round(model_measured_pred[predicted_prop_id], 3)
  model_measured_pred["Percent_Error"] = round(((model_measured_pred.predicted_cmc - model_measured_pred.measured_cmc)/model_measured_pred.measured_cmc)*100, 2) # Calc percent Error
  plt.scatter(model_measured_pred.measured_cmc,  model_measured_pred.predicted_cmc,   label = 'Train',    c='blue') #Plotting training data measured vs predicted
  plt.title('Model Performance (Train and Test Set)')
  plt.xlabel(measured_prop_id)
  plt.ylabel(predicted_prop_id)
  plt.legend(loc=4)
  plt.show()
  print(train_scores)
  print(valid_scores)
  print('Total MAE:' , round(sum(abs(model_measured_pred.Percent_Error))/len(model_measured_pred),3), '%') # Printing MAE
  return GCNN

def GraphCNN_preprocess_modelPredict(dataset_file, smiles_field = "smiles", model, batchSize = 100):
  loader = dc.data.CSVLoader(tasks=["prop"],  smiles_field="smiles", featurizer=dc.feat.ConvMolFeaturizer())
  dataset = loader.featurize(dataset_file) # Featurizing the dataset with ConvMolFeaturizer
  normalizer = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset,move_mean=True)
  struc_data = normalizer.transform(dataset)
  cmc_pred = model.predict_on_batch(struc_data.X[:batchSize])
  cmc_pred= normalizer.untransform(cmc_pred)
  cmc_pred_list = [x for l in cmc_pred for x in l]
  train_smiles = list(struc_data.ids[:batchSize])
  model_measured_pred = pd.DataFrame({'smiles': train_smiles, 'predicted_cmc': cmc_pred_list})
  model_measured_pred['predicted_cmc'] = round(model_measured_pred.predicted_cmc, 3)
  return model_measured_pred
