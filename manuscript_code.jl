

# LOAD PATIENT CLINICAL DATA INTO MEMORY FOR ANALYSIS
using Helpers, TSV, Statistics;
FIRSTLINE_ONLY = true; WITHOUT_CFDNA = false; PARSIMONIOUS = false;

d = read_tsv("clin_ctdna_data_Nov2022.tsv");
if FIRSTLINE_ONLY
	d.data = d.data[d[:, "Collection_timepoint"] .== "1L", :];
end
sample = d[:, "Sample_ID"];
ctdna_frac = d[:, "Final_ctDNA_value"];
timepoint = d[:, "Collection_timepoint"];
features = d[:, indexin(["cfDNA_ng_mL_plasma", "liver_mets_(n-0/y-1)", "lung_mets_(n-0/y-1)", "loc_mets.LN", "PSA_at_initiation_of_Rx", "ECOG_(0/1/2/3/4/UKN)", "Alk_Phos_(u/l)/_ULN", "LDH_(u/l)/_ULN"], d.headers)];
feature_names = split("cfdna liver_mets lung_mets node_mets psa ecog alp ldh");
if WITHOUT_CFDNA
	features = features[:, 2:end]; feature_names = feature_names[2:end];
end
monotonicity = Dict(:monotone_constraints => (1,1,1,1,1,1,1,1));
if PARSIMONIOUS == false   # Extra input features in full model
	num_bone_mets = d[:, "number_of_bone_mets.1-3_(abienza)"];
	num_bone_mets[d[:, "number_of_bone_mets.4-9_(abienza)"] .== 1] .= 2;
	num_bone_mets[d[:, "number_of_bone_mets.10_and_above_(abienza)"] .== 1] .= 3
	features = hcat(features,
		d[:, "mCRPC_1L.time_from_ADT_to_1L_mCRPC_(months)"],
		d[:, "loc_mets.bones"], d[:, "loc_mets.visceral"],
		d[:, "Hb_(g/l)"], d[:, "Albumin_final_(g/l)"], d[:, "Age__mCRPC"],
		d[:, "initial_diagnosis.gleason.Gleason_(6/7/8/9/10/UKN)"],
		d[:, "initial_diagnosis.initial_Dx_loc(0)/met(1)/_ukn(2)"],
		num_bone_mets)
	append!(feature_names, split("time_to_adt node_mets bone_mets visceral_mets albumin hemoglobin age gleason dx_loc num_bone_mets"))
	monotonicity = Dict();
end
features = map(x -> x isa Number ? Float32(x) : missing, features);
feature_stdev = [std(filter(!ismissing, features[:, f])) for f in 1:size(features, 2)];
S = length(sample);





















# PREDICTION OF CTDNA% USING XGBOOST (REGRESSION)
using XGBoost, MachineLearning, Logging;
Logging.disable_logging(Logging.Info);

true_ctdna = zeros(0); predicted_ctdna = zeros(0);
for validation in kfolds(S, 20)
	training = setdiff(1:S, validation)

	# Impute missing training and validation values using the KNN method
	data = copy(features)
	data[training, :] = impute_knn(features[training, :], feature_stdev)
	data[validation, :] = impute_knn(features[validation, :], feature_stdev)

	# Identify optimal hyperparameters for learning
	hyperparams = [(rounds, depth, eta, subsample) for rounds in [10, 20, 50, 100, 200, 500], depth in 2:8, eta in [0.005, 0.01, 0.02, 0.05, 0.1], subsample in 0.25:0.25:1]
	hyper_abandoned = falses(size(hyperparams))
	mae = fill(NaN, size(hyperparams))

	for H in 1:10
		if length(hyperparams) == 1; break; end
		hypertuning_folds = kfolds(length(training), 5; repeat=2^(H-1))
		for k in eachindex(hyperparams)
			if hyper_abandoned[k]; continue; end
			(rounds, depth, eta, subsample) = hyperparams[k]

			inner_true_ctdna = zeros(0); inner_predicted_ctdna = zeros(0);
			for fold in hypertuning_folds
				# Further subdivide the training set into an inner training and
				# validation set for hyperparameter tuning
				inner_validation = training[fold]
				inner_training = setdiff(training, inner_validation)
				booster = xgboost((data[inner_training, :], ctdna_frac[inner_training]); nthread=20, num_round=rounds, verbosity=0, max_depth=depth, eta=eta, subsample=subsample, colsample_bytree=subsample, colsample_bylevel=subsample,
					objective="reg:absoluteerror", monotonicity...)
				append!(inner_predicted_ctdna, predict(booster, data[inner_validation, :]))
				append!(inner_true_ctdna, ctdna_frac[inner_validation])
			end

			mae[k] = mean(abs.(inner_predicted_ctdna .- inner_true_ctdna))
		end

		prev_eligible = sum(!hyper_abandoned)
		median_mae = median(mae[k] for k in eachindex(hyperparams) if !hyper_abandoned[k])
		for k in eachindex(hyperparams)
			if !hyper_abandoned[k] && mae[k] > median_mae
				hyper_abandoned[k] = true
			end
		end

		now_eligible = sum(!hyper_abandoned)
		@printf("Round %d (%d iterations of CV): hyperparams %d -> %d\n",
			H, 2^(H-1), prev_eligible, now_eligible)

		if now_eligible == 1; break; end
	end

	# At this point we only have one eligible set of hyperparameters left,
	# or we are unable to distinguish between the best hyperparameters.
	# So let's pick the winner now.
	best = argmin(mae .+ hyper_abandoned .* Inf)
	(rounds, depth, eta, subsample) = hyperparams[best]
	@printf("Best MAE: %.3f (rounds = %d, depth = %d, eta = %.3f, subsample = %.2f)\n", mae[best], rounds, depth, eta, subsample)

	# Train a model using the optimal hyperparameters, and evaluate prediction
	# accuracy using the outer validation set.
	booster = xgboost((data[training, :], ctdna_frac[training]); nthread=20, num_round=rounds, verbosity=0, max_depth=depth, eta=eta, subsample=subsample, colsample_bytree=subsample, colsample_bylevel=subsample, objective="reg:absoluteerror", monotonicity...)
	append!(predicted_ctdna, predict(booster, data[validation, :]))
	append!(true_ctdna, ctdna_frac[validation])
end

predicted_ctdna = max.(predicted_ctdna, 0);
mae = mean(abs.(predicted_ctdna .- true_ctdna))











# PREDICTION OF P(CTDNA > 2%) USING XGBOOST (CLASSIFICATION)
using XGBoost, MachineLearning, Logging;
Logging.disable_logging(Logging.Info);

true_ctdna = zeros(0); predicted_ctdna = zeros(0);
for validation in kfolds(S, 20)
	training = setdiff(1:S, validation)

	# Impute missing training and validation values using the KNN method
	data = copy(features)
	data[training, :] = impute_knn(features[training, :], feature_stdev)
	data[validation, :] = impute_knn(features[validation, :], feature_stdev)

	# Identify optimal hyperparameters for learning
	hyperparams = [(rounds, depth, eta, subsample) for rounds in [10, 20, 50, 100, 200, 500], depth in 2:8, eta in [0.005, 0.01, 0.02, 0.05, 0.1], subsample in [0.25, 0.5, 0.75]]
	hyper_abandoned = falses(size(hyperparams))
	auc = fill(Inf, size(hyperparams))
	
	for H in 1:1000
		if length(hyperparams) == 1; break; end
		hypertuning_folds = kfolds(length(training), 5; repeat=2^(H-1))
		for k in eachindex(hyperparams)
			if hyper_abandoned[k]; continue; end
			(rounds, depth, eta, subsample) = hyperparams[k]

			inner_true_ctdna = zeros(0); inner_predicted_ctdna = zeros(0);
			for fold in hypertuning_folds
				# Further subdivide the training set into an inner training and
				# validation set for hyperparameter tuning
				inner_validation = training[fold]
				inner_training = setdiff(training, inner_validation)
				booster = xgboost((data[inner_training, :], ctdna_frac[inner_training] .>= 0.02); nthread=20, num_round=rounds, verbosity=0, max_depth=depth, eta=eta, subsample=subsample, colsample_bytree=subsample, colsample_bylevel=subsample,
					objective="binary:logistic", monotonicity...)
				append!(inner_predicted_ctdna, predict(booster, data[inner_validation, :]))
				append!(inner_true_ctdna, ctdna_frac[inner_validation])
			end

			auc[k] = auc(inner_predicted_ctdna, inner_true_ctdna .>= 0.02)
		end

		prev_eligible = sum(!hyper_abandoned)
		median_auc = median(auc[k] for k in eachindex(hyperparams) if !hyper_abandoned[k])
		for k in eachindex(hyperparams)
			if !hyper_abandoned[k] && auc[k] < median_auc
				hyper_abandoned[k] = true
			end
		end

		now_eligible = sum(!hyper_abandoned)
		@printf("Round %d (%d iterations of CV): hyperparams %d -> %d\n",
			H, 2^(H-1), prev_eligible, now_eligible)

		if now_eligible == 1; break; end
	end

	# At this point we only have one eligible set of hyperparameters left,
	# or we are unable to distinguish between the best hyperparameters.
	# So let's pick the winner now.
	best = argmax(auc .- hyper_abandoned .* Inf)
	(rounds, depth, eta, subsample) = hyperparams[best]
	@printf("Best AUC: %.3f (rounds = %d, depth = %d, eta = %.3f, subsample = %.2f)\n", auc[best], rounds, depth, eta, subsample)

	# Train a model using the optimal hyperparameters, and evaluate prediction
	# accuracy using the outer validation set.
	booster = xgboost((data[training, :], ctdna_frac[training] .>= 0.02); nthread=20, num_round=rounds, verbosity=0, max_depth=depth, eta=eta, subsample=subsample, colsample_bytree=subsample, colsample_bylevel=subsample, objective="binary:logistic", monotonicity...)
	append!(predicted_ctdna, predict(booster, data[validation, :]))
	append!(true_ctdna, ctdna_frac[validation])
end

println(auc(predicted_ctdna, true_ctdna .>= 0.02))










# DIMENSIONALLY WEIGHTED K-NEAREST NEIGHBOR MODEL
using Helpers, Printf, TSV, Statistics, Optim;
# Run the code "JULIA XGBOOST BASED PREDICTION, INITIALIZATION CODE"
features = map(x -> ismissing(x) ? NaN : Float32(x), features);

K = 20;
S = length(ctdna_frac);

function weighted_dist(a::Vector, b::Vector, weights::Vector)
	dist = 0
	for k in 1:length(a); dist += abs((a[k] - b[k]) * weights[k]); end
	return dist
end

function train_knn(ctdna::Vector, predictors::Matrix, K::Int)
	S = length(ctdna)
	weights = map(f -> 1 / std(Float64.(predictors[:, f])), 1:size(predictors, 2))
	function prediction_error(weights::Vector)
		prediction = fill(NaN, S)
		for s in 1:S
			dist = map(n -> weighted_dist(predictors[s, :], predictors[n, :], weights), 1:S)
			dist[s] = Inf   # Don't count the sample itself as neighbor
			prediction[s] = median(ctdna[partialsortperm(dist, 1:K)])
		end
		return mean(abs.(prediction .- ctdna))
	end
	res = optimize(prediction_error, weights)
	return res.minimizer
end

# Leave-one-out
predicted_ctdna_frac = fill(NaN, length(ctdna_frac));
neighbors_above_2pct = zeros(Int, length(ctdna_frac));
for s in 1:S
	required = isfinite.(features[s, :])
	training = [r == s ? false : all(isfinite, features[r, required])
		for r in 1:S]
	weights = train_knn(ctdna_frac[training], features[training, required], K)
	dist = map(n -> weighted_dist(features[s, required], features[n, required], weights), 1:S)
	dist[s] = Inf
	nearest = partialsortperm(dist, 1:K)
	predicted_ctdna_frac[s] = median(ctdna_frac[nearest])
	neighbors_above_2pct[s] = sum(ctdna_frac[nearest] .>= 0.02)
	@printf("%s\t%.1f\t%d\n", sample[s], predicted_ctdna_frac[s] * 100, neighbors_above_2pct[s] / K * 100)
end



















# GENERATE FINAL XGBOOST CLASSIFICATION MODELS FOR MISSING DATA COMBINATIONS
using JSON;
for C in 511:-1:1
	model_vars = [UInt16(C) & (1 << k) > 0 for k in 0:8]
	@printf("Training model %d (%d variables):\n%s\n", C, sum(model_vars),
		join(feature_names[model_vars], ", "))
	if sum(model_vars) >= 4
		data = impute_knn(features[:, model_vars], feature_stdev[model_vars])
		ctdna = copy(ctdna_frac)
	else
		data = features[:, model_vars]
		incomplete = [any(ismissing, r) for r in eachrow(data)]
		data = Float32.(data[!incomplete, :])
		ctdna = ctdna_frac[!incomplete]
	end

	S = size(data, 1)

	# Identify optimal hyperparameters for learning
	hyperparams = [(rounds, depth, eta, subsample) for rounds in [10, 20, 50, 100, 200, 500], depth in 2:8, eta in [0.005, 0.01, 0.02, 0.05, 0.1], subsample in [0.25, 0.5, 0.75]]
	hyper_abandoned = falses(size(hyperparams))
	auc = fill(Inf, size(hyperparams))

	for H in 1:8
		if length(hyperparams) == 1; break; end
		hypertuning_folds = kfolds(S, 5; repeat=2^(H-1))
		for k in eachindex(hyperparams)
			if hyper_abandoned[k]; continue; end
			(rounds, depth, eta, subsample) = hyperparams[k]

			inner_true_ctdna = zeros(0); inner_predicted_ctdna = zeros(0);
			for inner_validation in hypertuning_folds
				inner_training = setdiff(1:S, inner_validation)
				booster = xgboost((data[inner_training, :], ctdna[inner_training] .>= 0.02); nthread=10, num_round=rounds, verbosity=0, max_depth=depth, eta=eta, subsample=subsample, colsample_bytree=subsample, colsample_bylevel=subsample,
					objective="binary:logistic", monotonicity...)
				append!(inner_predicted_ctdna, predict(booster, data[inner_validation, :]))
				append!(inner_true_ctdna, ctdna[inner_validation])
			end

			auc[k] = auc(inner_predicted_ctdna, inner_true_ctdna .>= 0.02)
		end

		# Calculate median logloss among hyperparameters that are still
		# competing
		prev_eligible = sum(!hyper_abandoned)
		median_auc = median(auc[k] for k in eachindex(hyperparams) if !hyper_abandoned[k])
		for k in eachindex(hyperparams)
			if !hyper_abandoned[k] && auc[k] < median_auc
				hyper_abandoned[k] = true
			end
		end

		now_eligible = sum(!hyper_abandoned)
		@printf("Round %d (%d iterations of CV): hyperparams %d -> %d\n",
			H, 2^(H-1), prev_eligible, now_eligible)
		if now_eligible == 1; break; end
	end

	# At this point we only have one eligible set of hyperparameters left,
	# or we are unable to distinguish between the best hyperparameters.
	# So let's pick the winner now.
	best = argmax(auc .- hyper_abandoned .* Inf)
	(rounds, depth, eta, subsample) = hyperparams[best]
	@printf("Best AUC: %.3f (rounds = %d, depth = %d, eta = %.3f, subsample = %.2f)\n", auc[best], rounds, depth, eta, subsample)

	# Train final model using the optimal hyperparameters
	booster = xgboost((data, ctdna .>= 0.02); nthread=20, num_round=rounds, verbosity=0, max_depth=depth, eta=eta, subsample=subsample, colsample_bytree=subsample, colsample_bylevel=subsample, objective="binary:logistic", monotonicity...)
	XGBoost.save(booster, "final_models/$C.xgboost_model")
	out = open("final_models/$(C)_hyperparams.json", "w")
	write(out, JSON.json(Dict("rounds" => rounds, "depth" => depth, "eta" => eta, "subsample" => subsample)))
	close(out)
end







# GENERATE FINAL XGBOOST REGRESSION MODELS FOR MISSING DATA COMBINATIONS
using JSON;
for C in 511:-1:1
	model_vars = [UInt16(C) & (1 << k) > 0 for k in 0:8]
	@printf("Training model %d (%d variables):\n%s\n", C, sum(model_vars),
		join(feature_names[model_vars], ", "))
	if sum(model_vars) >= 4
		data = impute_knn(features[:, model_vars], feature_stdev[model_vars])
		ctdna = copy(ctdna_frac)
	else
		data = features[:, model_vars]
		incomplete = [any(ismissing, r) for r in eachrow(data)]
		data = Float32.(data[!incomplete, :])
		ctdna = ctdna_frac[!incomplete]
	end

	S = size(data, 1)

	# Identify optimal hyperparameters for learning
	hyperparams = [(rounds, depth, eta, subsample) for rounds in [10, 20, 50, 100, 200, 500], depth in 2:8, eta in [0.005, 0.01, 0.02, 0.05, 0.1], subsample in [0.25, 0.5, 0.75]]
	hyper_abandoned = falses(size(hyperparams))
	mae = fill(Inf, size(hyperparams))

	for H in 1:8
		if length(hyperparams) == 1; break; end
		hypertuning_folds = kfolds(S, 5; repeat=2^(H-1))
		for k in eachindex(hyperparams)
			if hyper_abandoned[k]; continue; end
			(rounds, depth, eta, subsample) = hyperparams[k]

			inner_true_ctdna = zeros(0); inner_predicted_ctdna = zeros(0);
			for inner_validation in hypertuning_folds
				inner_training = setdiff(1:S, inner_validation)

				booster = xgboost((data[inner_training, :], ctdna[inner_training]); nthread=10, num_round=rounds, verbosity=0, max_depth=depth, eta=eta, subsample=subsample, colsample_bytree=subsample, colsample_bylevel=subsample,
					objective="reg:absoluteerror", monotonicity...)
				append!(inner_predicted_ctdna, predict(booster, data[inner_validation, :]))
				append!(inner_true_ctdna, ctdna[inner_validation])
			end

			mae[k] = mean(abs.(inner_predicted_ctdna .- inner_true_ctdna))
		end

		# Calculate median logloss among hyperparameters that are still
		# competing
		prev_eligible = sum(!hyper_abandoned)
		median_mae = median(mae[k] for k in eachindex(hyperparams) if !hyper_abandoned[k])
		for k in eachindex(hyperparams)
			if !hyper_abandoned[k] && mae[k] > median_mae
				hyper_abandoned[k] = true
			end
		end

		now_eligible = sum(!hyper_abandoned)
		@printf("Round %d (%d iterations of CV): hyperparams %d -> %d\n",
			H, 2^(H-1), prev_eligible, now_eligible)
		if now_eligible == 1; break; end
	end

	# At this point we only have one eligible set of hyperparameters left,
	# or we are unable to distinguish between the best hyperparameters.
	# So let's pick the winner now.
	best = argmin(mae .+ hyper_abandoned .* Inf)
	(rounds, depth, eta, subsample) = hyperparams[best]
	@printf("Best MAE: %.3f (rounds = %d, depth = %d, eta = %.3f, subsample = %.2f)\n", mae[best], rounds, depth, eta, subsample)

	# Train final model using the optimal hyperparameters
	booster = xgboost((data, ctdna); nthread=20, num_round=rounds, verbosity=0, max_depth=depth, eta=eta, subsample=subsample, colsample_bytree=subsample, colsample_bylevel=subsample, objective="reg:absoluteerror", monotonicity...)
	XGBoost.save(booster, "final_models_reg/$C.xgboost_model")
	out = open("final_models_reg/$(C)_hyperparams.json", "w")
	write(out, JSON.json(Dict("rounds" => rounds, "depth" => depth, "eta" => eta, "subsample" => subsample)))
	close(out)
end


