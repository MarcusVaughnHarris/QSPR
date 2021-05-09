
NIS_SeqCMC_Descr = CalcDescr_SMIdf(NIS_SeqCMC, FingerPrintType, DescriptorList)

# Create binary data type for predicting if model will predict a compounds accuracy within 4%
NIS_SeqCMC_Descr['ErrLessThanMedian'] =  abs(NIS_SeqCMC_Descr.Percent_Error) < np.median(abs(NIS_SeqCMC_Descr.Percent_Error))

Descr_tr, Descr_ts, Prop_tr, Prop_ts  = PreProcess_SMIdf(NIS_SeqCMC_Descr,
                                                         property_id = 'ErrLessThanMedian',
                                                         TestSize = 0.25) #----- Pre-process

cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
scale = StandardScaler().fit(Descr_tr)
Descr_tr = scale.transform(Descr_tr)
# create grid search dictionary
param_grid = {"max_features": [Descr_tr.shape[1] // 10, Descr_tr.shape[1] // 7, Descr_tr.shape[1] // 5, Descr_tr.shape[1] // 3], 
              "n_estimators": [100, 250, 500]}
RFC_MedError = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=2, cv=cv, verbose=1) # setup model building
RFC_MedError.fit(Descr_tr, Prop_tr)# run modsel building
print('best params', RFC_MedError.best_params_)
print('best score:',RFC_MedError.best_score_)
#Use Applicability Domain estimation to sort generated molecules for those with highest probability of having less than 4% error for predicted CMC

Gen_mol_Desc_array_scaled = scale.transform(np.array(list(GenMols_SMIdf_sorted['Descriptors']))) #Creating NUMPY array for descriptors & scaling
pred = RFC_MedError.predict(Gen_mol_Desc_array_scaled)
pred_prob = RFC_MedError.predict_proba(Gen_mol_Desc_array_scaled)
threshold = np.amax(pred_prob*.90)
da = np.amax(pred_prob, axis=1) > threshold
pred_prob_err = pd.DataFrame(pred_prob, columns = ['ProbabilityThatErrOverMedian', 'ProbabilityThatErrUnderMedian'])
GenMols_SMIdf_sorted["ProbabilityThatErrOverMedian"] = pred_prob_err.ProbabilityThatErrOverMedian.values
GenMols_SMIdf_sorted["ProbabilityThatErrUnderMedian"] = pred_prob_err.ProbabilityThatErrUnderMedian.values


GenMols_SMIdf_sorted["Err_UnderMedian"] = pred
GenMols_SMIdf_sorted = GenMols_SMIdf_sorted[da]
#GenMols_SMIdf_least_error = GenMols_SMIdf_sorted[GenMols_SMIdf_sorted.ModelVerdict_MoreThan4perc ==False]
sorted = GenMols_SMIdf_sorted.sort_values(by='ProbabilityThatErrUnderMedian', ascending=False)

print(sum(sorted.Err_UnderMedian))
sorted

Draw.MolsToGridImage(sorted.Mol.head(n=20), molsPerRow = 8, subImgSize = (250,250), legends = [str(round(val,3)) for val in sorted.Predicted_cmc_NegativeLogM.head(n=20)])
