def FP_Similarity_Filter(GenMols, TrainMols, RangeTrain2Compare ):
  def FP_Similarity(GenMols, TrainMols, TrainMol2Compare ):
    GenMols = GenMols.append([TrainMols[TrainMol2Compare:(TrainMol2Compare+1)]], ignore_index=True)
    Tmol2compare = len(GenMols)-1
    GenMols["fps"]= [Chem.RDKFingerprint(mol) for mol in GenMols.Mol]
    GenMols['Dist'] = [DataStructs.FingerprintSimilarity(GenMols.fps[Tmol2compare],fps) for fps in GenMols.fps]
    GenMols['Tmol'] = ["{}{}".format('T', TrainMol2Compare) for dist in GenMols['Dist'] ]
    GenMols['Dist_Str'] = [str(round(dist,4)) for dist in GenMols.Dist]
    GenMols['Distance_str'] = GenMols[['Tmol', 'Dist_Str']].agg(' Similarity:  '.join, axis=1)
    GenMolsSimilarity = GenMols.drop(['Tmol', 'Dist_Str'], axis=1)
    return GenMolsSimilarity
  allTrainingMols = [FP_Similarity(GenMols = GenMols,   TrainMols = TrainMols,    TrainMol2Compare = mol) for mol in RangeTrain2Compare ]
  allTrainingMols = pd.concat(allTrainingMols)
  #allTrainingMols_sorted = allTrainingMols.sort_values(by='Dist', ascending=False)
  Distsummary = allTrainingMols.groupby('smiles', as_index=False).mean()#['Avg_Dist2_Tmols'].mean()
  Dist_sum_merged = Distsummary.merge(GenMols, left_on='smiles', right_on='smiles')  
  Dist_sum_merged_sorted = Dist_sum_merged.sort_values(by = 'Dist', ascending=False)
  return Distsummary, allTrainingMols, Dist_sum_merged_sorted

# Functions derived from https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html
def IsoForest_OutlierEstimator(GenMols, TrainMols, MaxSamples):
  GenMols["Mol"] = [Chem.MolFromSmiles(smi) for smi in GenMols.smiles]
  X_train =  [list(AllChem.GetMorganFingerprintAsBitVect(mol, 2)) for mol in TrainMols.Mol]
  X_test =   [list(AllChem.GetMorganFingerprintAsBitVect(mol, 2)) for mol in GenMols.Mol]
  # fit the model
  pca = PCA(n_components=2)
  pcares_train = pca.fit_transform(X_train)
  clf = IsolationForest( max_samples=MaxSamples, contamination='auto')
  clf.fit(pcares_train)
  y_pred_train = clf.predict(pcares_train)
  pcares_test = pca.fit_transform(X_test)
  y_pred_test = clf.predict(pcares_test)
  cmap = {1:'blue', -1:'red'}
  cs = [cmap[k] for k in y_pred_test]
  # plot the line, the samples, and the nearest vectors to the plane
  xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
  Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z_reshaped = Z.reshape(xx.shape)
  plt.title("IsoForest")
  plt.contourf(xx, yy, Z_reshaped, cmap=plt.cm.Blues_r)
  inline_pcare_test = pcares_test[y_pred_test == 1]
  outlier_pcare_test = pcares_test[y_pred_test == -1]
  b1 = plt.scatter(pcares_train[:, 0], pcares_train[:, 1], c='white',                 s=20, edgecolor='k')
  b2 = plt.scatter(inline_pcare_test[:, 0], inline_pcare_test[:, 1],                  s=20,  c='blue', edgecolor='k')
  c = plt.scatter(outlier_pcare_test[:, 0], outlier_pcare_test[:, 1], c='red',                s=20, edgecolor='k')
  plt.axis('tight')
  plt.xlim((-5, 5))
  plt.ylim((-5, 5))
  plt.legend([b1, b2,c],["Training Observations","Gen Mols Within AD", "Gen Mol Outside AD",],loc="upper left")
  GenMols['GenMols_AD'] = y_pred_test
  print('\n\n# Gen Mols Within AD:', len(inline_pcare_test))
  print('# Gen Mol Outside AD:', len(outlier_pcare_test),'\n\n')
  return GenMols

#Mol distance to model (training set)


def Dist2Model(GeneratedMols, TrainingMols):
  DistSum_T, allTrainMols_T, TrainMols_Dist = FP_Similarity_Filter(GenMols = TrainingMols ,
                                                                 TrainMols = TrainingMols,  
                                                                 RangeTrain2Compare = range(0,(len(TrainingMols)-1)) )
  print('Range of Dist (TrainMols to TrainSet):',round(TrainMols_Dist.Dist.min(),4), '-',round(TrainMols_Dist.Dist.max(),4))
  print('Dist Median (TrainMols to TrainSet):' , round(TrainMols_Dist.Dist.median(),4), '\nDist Mean (TrainMols to TrainSet):' , round(TrainMols_Dist.Dist.mean(),4))
  print('Dist StDev (TrainMols to TrainSet):' ,round(statistics.pstdev(TrainMols_Dist.Dist),4))
  Distsummary, allTrainingMols, GenMols_IF_Dist = FP_Similarity_Filter(GenMols = GeneratedMols, 
                                                                    TrainMols = TrainingMols, 
                                                                    RangeTrain2Compare = range(0, len(GeneratedMols))  )
  GenMols_IF_Dist = pd.DataFrame({"smiles": GenMols_IF_Dist.smiles, "IsoForest_AD":GenMols_IF_Dist.GenMols_AD_x, "Dist2TrainSet": GenMols_IF_Dist.Dist, "Mol":GenMols_IF_Dist.Mol })
  return GenMols_IF_Dist, TrainMols_Dist


def Dist2Model_APfp(GenMols, trainDF):
  pairFps_train = [Pairs.GetAtomPairFingerprint(x) for x in trainDF.Mol]
  pairFps_gen = [Pairs.GetAtomPairFingerprint(x) for x in GenMols.Mol]
  
  def avg_dists(GenMols, trainDF, pairFps_gen, pairFps_train, x):
    n_list = [x] * len(trainDF)
    ZIPcompare = itertools.zip_longest(range(0, len(GenMols)),  n_list)
    dists = [DataStructs.DiceSimilarity(pairFps_gen[n1],pairFps_train[n2])for n1, n2 in ZIPcompare]
    avg_dist = sum(dists)/ len(dists)
    return avg_dist

  train_set_range = range(0, len(GenMols))
  averages = [avg_dists(GenMols, trainDF, pairFps_gen, pairFps_train, x) for x in train_set_range]
  return averages
