def GraphCNN_preprocess_modelTraining(dataset_file):
  loader = dc.data.CSVLoader(tasks=["cmc_NegativeLogM"],  smiles_field="smiles", featurizer=dc.feat.ConvMolFeaturizer())
  dataset = loader.featurize(dataset_file) # Featurizing the dataset with ConvMolFeaturizer

  splitter = dc.splits.RandomSplitter() # Same as train_test_split from sklearn
  train, valid, _ = splitter.train_valid_test_split(dataset, frac_train=0.7, frac_valid=0.29, frac_test=0.01) # frac_test is 0.01 because we only use a train and valid as an example

  normalizer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train,move_mean=True)
  train = normalizer.transform(train)
  test = normalizer.transform(valid)
  metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
  return train, valid, _ 

def GraphCNN_preprocess_modelPredicting(SMIdf):
  dataset = featurize(SMIdf) # Featurizing the dataset with ConvMolFeaturizer
  normalizer = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset, move_mean=True)
  dataset = normalizer.transform(dataset)
  return dataset

def GCNN_Model_Performance(train, batchSize):
  cmc_pred = GCNN.predict_on_batch(train.X[:batchSize])
  cmc_pred_list = [x for l in cmc_pred for x in l]
  train_smiles = list(train.ids[:batchSize])
  #cmc_measured_list = [float(num) for num in train.y] 
  cmc_measured_list = [x for l in train.y[:batchSize] for x in l]
  model_measured_pred = pd.DataFrame({'smiles': train_smiles,'measured_cmc': cmc_measured_list ,'predicted_cmc': cmc_pred_list})
  model_measured_pred['measured_cmc'] = round(normalizer.untransform(model_measured_pred.measured_cmc), 3)
  model_measured_pred['predicted_cmc'] = round(normalizer.untransform(model_measured_pred.predicted_cmc), 3)
  model_measured_pred["Percent_Error"] = round((abs(model_measured_pred.predicted_cmc - model_measured_pred.measured_cmc)/model_measured_pred.measured_cmc)*100, 2) # Calc percent Error
  plt.scatter(model_measured_pred.measured_cmc,  model_measured_pred.predicted_cmc,   label = 'Train',    c='blue') #Plotting training data measured vs predicted
  plt.title('Model Performance (Train and Test Set)')
  plt.xlabel('measured_cmc')
  plt.ylabel('predicted_cmc')
  plt.legend(loc=4)
  plt.show()
  print(train_scores)
  print(valid_scores)
  print('Total MAE:' , round(sum(abs(model_measured_pred.Percent_Error))/len(model_measured_pred.Percent_Error),3), '%') # Printing MAE
  return model_measured_pred


def GCNN_Model_Predict(train, batchSize):
  cmc_pred = GCNN.predict_on_batch(train.X[:batchSize])
  cmc_pred_list = [x for l in cmc_pred for x in l]
  train_smiles = list(train.ids[:batchSize])
  #cmc_measured_list = [float(num) for num in train.y] 
  cmc_measured_list = [x for l in train.y[:batchSize] for x in l]
  model_measured_pred = pd.DataFrame({'smiles': train_smiles, 'predicted_cmc': cmc_pred_list})
  model_measured_pred['predicted_cmc'] = round(normalizer.untransform(model_measured_pred.predicted_cmc), 3)
  return model_measured_pred
