3
Prateek Chaudhry, Abhishek, Shivam Garg


I tried many of the approaches mentioned in the slides. Few of the results are shown below. 

I implemented and tested the following variants of Domain Adaptation.

Target only gave 
	MacroScore: 45-65 (for 90-10 split between train and test data)

Weighted combination of Target and src gave bad performance:
	MacroFscore < 45


Daume: 
	penalty='l1'
		0.517647058824
		confusion matrix :  [[0, 0, 41], [0, 0, 0], [0, 0, 44]]
		macroF_Score is 0.743056, 
		macroPrecision is 0.839216
		macroRecall is 0.666667

	with 1% src data, with weights 0.001 for src tweets and l1 norm
		0.364705882353
		confusion matrix :  [[0, 42, 0], [1, 31, 0], [1, 10, 0]]
		macroF_Score is 0.378717, 
		macroPrecision is 0.457831
		macroRecall is 0.322917


	with 10% src data, with weights 0.001 for src tweets and l1 norm  
		0.341176470588
		confusion matrix :  [[0, 42, 0], [0, 29, 0], [0, 14, 0]]
		macroF_Score is 0.467136, 
		macroPrecision is 0.780392
		macroRecall is 0.333333

	with 10% src data, with weights 0.001 for src tweets and l1 norm  and solver='newton-cg' | CountVectorizer
		0.235294117647
		confusion matrix :  [[6, 8, 25], [11, 4, 15], [4, 2, 10]]
		macroF_Score is 0.278640, 
		macroPrecision is 0.257143
		macroRecall is 0.304060

	with 10% src data, with no weights src tweets and l2 norm  and solver='newton-cg' | TfidfVectorizer
		0.576470588235
		confusion matrix :  [[48, 0, 0], [23, 1, 0], [13, 0, 0]]
		macroF_Score is 0.494234, 
		macroPrecision is 0.857143
		macroRecall is 0.347222

	with 20% src data, with no weights src tweets and l2 norm  and solver='lbfgs' | TfidfVectorizer
		0.470588235294
		confusion matrix :  [[40, 0, 0], [38, 0, 0], [7, 0, 0]]
		macroF_Score is 0.474576, 
		macroPrecision is 0.823529
		macroRecall is 0.333333

----------------------------------------------------------------------------------------------------
Added important featured like removed repeated chars, Separated camelcase and removed redundent features.

	with 20% src data, with no weights src tweets and l2 norm  and solver='lbfgs' | TfidfVectorizer
		0.588235294118
		confusion matrix :  [[39, 3, 0], [20, 10, 0], [7, 5, 1]]
		macroF_Score is 0.549690, 
		macroPrecision is 0.715488
		macroRecall is 0.446276

	with 50% src data, with no weights src tweets and l2 norm  and solver='lbfgs' | TfidfVectorizer
		0.552941176471
		confusion matrix :  [[39, 3, 0], [21, 8, 0], [10, 4, 0]]
		macroF_Score is 0.509440, 
		macroPrecision is 0.696825
		macroRecall is 0.401478

	with 10% src data, with no weights src tweets and l1 norm | TfidfVectorizer
		0.494117647059
		confusion matrix :  [[35, 5, 0], [23, 7, 0], [12, 3, 0]]
		macroF_Score is 0.472569, 
		macroPrecision is 0.655556
		macroRecall is 0.369444


	with 51% src data, with no weights src tweets and l2 norm | solver='lbfgs' | TidfVectorizer
		0.564705882353
		confusion matrix :  [[41, 7, 0], [20, 7, 0], [7, 3, 0]]
		macroF_Score is 0.478076, 
		macroPrecision is 0.671569
		macroRecall is 0.371142
			











