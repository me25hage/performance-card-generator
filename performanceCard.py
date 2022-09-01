import numpy as np
import skimage.io as io
import pandas as pd
from sklearn.metrics import roc_curve, auc, mean_absolute_error, r2_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from math import sqrt
from matplotlib import use
use("Agg")
import matplotlib.pyplot as plt
import csv,argparse
import warnings
warnings.filterwarnings("ignore")

from reportlab.pdfgen import canvas

def classification_metrics(df,data,val):

	TP = len(df.loc[(df[data] == 1) & (df[val] == 1)])
	TN = len(df.loc[(df[data] == 0) & (df[val] == 0)])
	
	FP = len(df.loc[(df[data] == 1) & (df[val] == 0)])
	FN = len(df.loc[(df[data] == 0) & (df[val] == 1)])


	acc = (TP + TN)/(TP+FP+TN+FN)
	precision = TP/(TP+FP)
	recall = TP/(TP+FN)

	f1 = 2 * (precision * recall) / (precision + recall)

	return acc, precision, recall, f1

def get_roc(df,data,eval,title = "",plot=1):
	df1 = df[[data,eval]].dropna()
	fpr, tpr, thresholds = roc_curve(df1[eval], df1[data])
	ks=np.abs(tpr-fpr)
	if plot==1:
		# Plot ROC curve
		plt.figure(figsize=(6,4))
		plt.plot(fpr, tpr, label='AUC=%0.2f KS=%0.2f' %(auc(fpr, tpr),ks.max()))
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.grid(b=True, which='both', color='0.65',linestyle='-')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(title+'Receiver Operating Characteristic')
		plt.legend(loc="lower right")
		plt.savefig("Graphics/ROC.png")
	return auc(fpr, tpr),np.max(np.abs(tpr-fpr)),thresholds[ks.argmax()]



def rsq(df, data, val, eval = None):
	yavg = df[val].sum()/len(df[val])
	denom = (df[val]-yavg)**2
	denom = denom.sum()

	num = (df[val] - df[data])**2
	num = num.sum()

	rsquared = 1 - num/denom

	return rsquared

def mse(df, data, val, title = "", plot = 1):
	err = (df[val] - df[data])**2

	if plot==1:
		plt.figure(figsize=(6,4))
		plt.hist(err)
		plt.grid(b=True, which='both', color='0.65',linestyle='-')
		plt.title(title+'Square Error')
		plt.savefig("Graphics/MSE.png")

	err = err.sum()/len(df[val])
	return err

def mae(df, data, val, title = "",plot=1):
	err = abs(df[val] - df[data])

	if plot==1:
		plt.figure(figsize=(6,4))
		plt.hist(err)
		plt.grid(b=True, which='both', color='0.65',linestyle='-')
		plt.title(title+'Absolute Error')
		plt.savefig("Graphics/MAE.png")


	err = err.sum()/len(df[val])

	return err

def mbe(df, data, val, title = ""):
	err = df[val] - df[data]

	err = err.sum()/len(df[val])

	return err

def create_regression_pdf(df,data, val, eval,title):
	can = canvas.Canvas("RegressionPerformanceCard.pdf")
	x = 700
	y = 500
	can.setPageSize((x, y))

	can.setStrokeColorRGB(62/256.0,55/256.0,92/256.0)
	can.setFillColorRGB(62/265.0,55/256.0,92/256.0)

	can.rect(0,400,700,100, fill=1)

	img0 = io.imread("Graphics/xd-logo-light-01.png")
	ar = (1.0 * img0.shape[1])/img0.shape[0]
	can.drawImage('Graphics/xd-logo-light-01.png', 5, 405, width=90*ar, height=90, mask='auto')

	can.setStrokeColorRGB(1,1,1)
	can.setFillColorRGB(1,1,1)
	can.setFont('Helvetica',28)
	can.drawCentredString(350, 450, title + " Performance Card")
	can.setFont('Helvetica',18)
	can.drawCentredString(350, 425, "Model type: Regression, %d data points" % len(df[data]))

	can.setStrokeColorRGB(0,0,0)
	can.setFillColorRGB(0,0,0)

	can.setFont('Helvetica',14)
	can.setStrokeColorRGB(72/256.0,45/256.0,108/256.0)
	can.setFillColorRGB(72/265.0,45/256.0,108/256.0)

	can.drawCentredString(200,365,"Mean Square Error = %.8f" % mse(df,data,val))

	can.drawCentredString(200,295,"Root Mean Square Error = %.8f" % sqrt(mse(df,data,val)))

	can.drawCentredString(200,225,"Mean Absolute Error = %.8f" % mae(df,data,val))

	can.drawCentredString(200,155,"R-Squared = %.8f" % rsq(df,data,val))



	# DEFINTIONS
	can.setFont('Helvetica',8)
	can.setStrokeColorRGB(0,0,0)
	can.setFillColorRGB(0,0,0)
	can.drawCentredString(200,355,"Mean square error informs the user how close the regression")
	can.drawCentredString(200,345,"line is to a set of data points. The errors are the distance")
	can.drawCentredString(200,335,"from the point to the line and the difference is squared")
	can.drawCentredString(200,325,"to account for negatives and provide weight to larger")
	can.drawCentredString(200,315,"differences. MSE should be minimized as 0 is no error.")

	can.drawCentredString(200,285,"Root mean square error is the standard deviation of the")
	can.drawCentredString(200,275,"prediction errors, or the spread of the errors about")
	can.drawCentredString(200,265,"the line of best fit. If the RMSE is 0, the data points lie")
	can.drawCentredString(200,255,"exactly on the line of best fit. RMSE is scale dependent so")
	can.drawCentredString(200,245,"it should not be used to compare different types of data.")

	can.drawCentredString(200,215,"Mean absolute error is the average magnitude of the errors")
	can.drawCentredString(200,205,"without accounting for the error direction. Similarly to")
	can.drawCentredString(200,195,"MSE and RMSE, the mean absolute error should be minimized,")
	can.drawCentredString(200,185,"with considerations for overfitting, to reduce the error")
	can.drawCentredString(200,175,"in the best fit line.")

	can.drawCentredString(200,145,"R-squared, or the coefficient of determination, measures")
	can.drawCentredString(200,135,"the proportion of variance of the dependent variable that")
	can.drawCentredString(200,125,"is described by the independent variable. An R-squared value")
	can.drawCentredString(200,115,"close to 1 tells the user how well the model describes the data.")
	can.drawCentredString(200,105,"A high R-squared value is not always optimal and can indicate issues.")




	img0 = io.imread("Graphics/MAE.png")
	ar = (1.0 * img0.shape[1])/img0.shape[0]
	can.drawImage('Graphics/MAE.png', 375, 200, width=200*ar, height=200, mask='auto')


	img0 = io.imread("Graphics/MSE.png")
	ar = (1.0 * img0.shape[1])/img0.shape[0]
	can.drawImage('Graphics/MSE.png', 375, 0, width=200*ar, height=200, mask='auto')



	can.showPage()

	for ev in eval:
		
		can.setStrokeColorRGB(62/256.0,55/256.0,92/256.0)
		can.setFillColorRGB(62/265.0,55/256.0,92/256.0)

		can.rect(0,460,700,40, fill=1)

		img0 = io.imread("Graphics/xd-logo.png")
		ar = (1.0 * img0.shape[1])/img0.shape[0]
		can.drawImage('Graphics/xd-logo.png', 645, 5, width=50*ar, height=50, mask='auto')

		can.setStrokeColorRGB(1,1,1)
		can.setFillColorRGB(1,1,1)
		can.setFont('Helvetica',20)
		can.drawCentredString(350, 470, ev)
		can.setStrokeColorRGB(0,0,0)
		can.setFillColorRGB(0,0,0)
		can.setFont('Helvetica',18)
		can.drawCentredString(250, 440, "R-Squared")
		can.drawCentredString(350, 440, "MSE")
		can.drawCentredString(450, 440, "MAE")
		can.drawCentredString(550, 440, "RMSE")

		i = 1
		n = len(df[ev].unique())
		if n >= 10:
			n = 10
		
		rsquareds = []
		mses = []
		rmses = []
		maes = []		
		sts = []

		for st in df[ev].unique():
			

			df_temp = df.loc[df[ev] == st]

			sts.append(st)		

			rsquareds.append(rsq(df_temp, data, val))

			mses.append(mse(df_temp, data, val))

			rmses.append(sqrt(mse(df_temp, data, val)))

			maes.append(mae(df_temp, data, val))

		for j in range(0,len(df[ev].unique())):

			can.setFont('Helvetica', 18)
			can.setFillColorRGB(0,0,0)
			can.drawString(20, i*(y-120)/n, sts[j])
			can.setFont('Helvetica', 15)

			percs = np.percentile(rsquareds,[10,90])
			
			if rsquareds[j] <= percs[0]:
				R = 1
				G = 0
				B = 0

			elif rsquareds[j] <= percs[1]:
				R = 0
				G = 0
				B = 0

			else:
				R = .2
				G = 1
				B = 0 

			can.setFillColorRGB(int(R),int(G),int(B))
			can.drawCentredString(250, i*(y-120)/n, "%.5f" % rsquareds[j])

			percs = np.percentile(mses,[10,90])
			
			if mses[j] <= percs[0]:
				R = .2
				G = 1
				B = 0

			elif mses[j] <= percs[1]:
				R = 0
				G = 0
				B = 0

			else:
				R = 1
				G = 0
				B = 0 

			can.setFillColorRGB(int(R),int(G),int(B))
			can.drawCentredString(350, i*(y-120)/n, "%.5f" % mses[j])

			percs = np.percentile(maes,[10,90])
			
			if maes[j] <= percs[0]:
				R = .2
				G = 1
				B = 0

			elif maes[j] <= percs[1]:
				R = 0
				G = 0
				B = 0

			else:
				R = 1
				G = 0
				B = 0 

			can.setFillColorRGB(int(R),int(G),int(B))
			can.drawCentredString(450, i*(y-120)/n, "%.5f" % maes[j])

			percs = np.percentile(rmses,[25,50,75,100])
			
			if rmses[j] <= percs[0]:
				R = .2
				G = 1
				B = 0

			elif rmses[j] <= percs[2]:
				R = 0
				G = 0
				B = 0

			else:
				R = 1
				G = 0
				B = 0 

			can.setFillColorRGB(int(R),int(G),int(B))
			can.drawCentredString(550, i*(y-120)/n, "%.5f" % sqrt(mses[j]))
		
			i += 1

			if i%10 == 1 and j < len(df[ev].unique())-1:
				can.showPage()
				can.setStrokeColorRGB(62/256.0,55/256.0,92/256.0)
				can.setFillColorRGB(62/265.0,55/256.0,92/256.0)

				can.rect(0,460,700,40, fill=1)

				img0 = io.imread("Graphics/xd-logo.png")
				ar = (1.0 * img0.shape[1])/img0.shape[0]
				can.drawImage('Graphics/xd-logo.png', 645, 5, width=50*ar, height=50, mask='auto')

				can.setStrokeColorRGB(1,1,1)
				can.setFillColorRGB(1,1,1)

				can.setFont('Helvetica',20)
				can.drawCentredString(350, 470, ev)
				can.setStrokeColorRGB(0,0,0)
				can.setFillColorRGB(0,0,0)
				can.setFont('Helvetica',18)
				can.drawCentredString(250, 440, "R-Squared")
				can.drawCentredString(350, 440, "MSE")
				can.drawCentredString(450, 440, "MAE")
				can.drawCentredString(550, 440, "RMSE")

				i = 1

		can.showPage()
	can.save()

def create_classification_pdf(df,data, val, eval,title=""):

	can = canvas.Canvas("ClassificationPerformanceCard.pdf")

	x = 700
	y = 500
	can.setPageSize((x, y))

	can.setStrokeColorRGB(62/256.0,55/256.0,92/256.0)
	can.setFillColorRGB(62/265.0,55/256.0,92/256.0)

	can.rect(0,400,700,100, fill=1)

	img0 = io.imread("Graphics/xd-logo-light-01.png")
	ar = (1.0 * img0.shape[1])/img0.shape[0]
	can.drawImage('Graphics/xd-logo-light-01.png', 5, 405, width=90*ar, height=90, mask='auto')

	can.setStrokeColorRGB(1,1,1)
	can.setFillColorRGB(1,1,1)
	can.setFont('Helvetica',28)
	can.drawCentredString(350, 450, title + " Performance Card")
	can.setFont('Helvetica',18)
	can.drawCentredString(350, 425, "Model type: Classification, %d data points" % len(df[data]))

	can.setStrokeColorRGB(0,0,0)
	can.setFillColorRGB(0,0,0)


	auc,ks,ks_score=get_roc(df,data,val)

	acc,pre,rec,f1 = classification_metrics(df,data,val)

	can.setFont('Helvetica',14)
	can.setStrokeColorRGB(72/256.0,45/256.0,108/256.0)
	can.setFillColorRGB(72/265.0,45/256.0,108/256.0)
	can.drawCentredString(125,365,"Accuracy = %.8f" % acc)

	can.drawCentredString(125,295,"Precision = %.8f" % pre)

	can.drawCentredString(125,225,"Recall = %.8f" % rec)

	can.drawCentredString(125,155,"F-1 = %.8f" % f1)

	can.drawCentredString(125,85,"AUC = %.8f" % auc)


	# DEFINTIONS
	can.setFont('Helvetica',8)
	can.setStrokeColorRGB(0,0,0)
	can.setFillColorRGB(0,0,0)
	can.drawCentredString(125,355,"Accuracy is the fraction of predictions the model got right.")
	can.drawCentredString(125,345,"Accuracy has a range of 0 to 1. This may be a deceptive")
	can.drawCentredString(125,335,"metric as the model could perform best for one class")
	can.drawCentredString(125,325,"and this metric masks the performance of the other.")
	can.drawCentredString(125,315,"It is best used alongside percision and recall.")
	
	can.drawCentredString(125,285,"Precision is the ratio between the number of positive cases")
	can.drawCentredString(125,275,"and the total number of cases classified as positive.")
	can.drawCentredString(125,265,"Precision is closest to one when the model maximizes correct")
	can.drawCentredString(125,255,"positive classifications (true positives) and minimizes")
	can.drawCentredString(125,245,"incorrect positive classifications (false positives.)")

	can.drawCentredString(125,215,"Recall is the fraction of true positives over the total")
	can.drawCentredString(125,205,"number of positive cases. When recall is high, more positive")
	can.drawCentredString(125,195,"cases were detected. This metric is 1 when false negatives")
	can.drawCentredString(125,185,"are 0. Thus, one can trust the model to detect positives.")
	
	can.drawCentredString(125,145,"F1 is the harmonic mean of precision and recall with the")
	can.drawCentredString(125,135,"goal of combining the two into one metric to work well")
	can.drawCentredString(125,125,"on imbalanced data. Often, one has to pick a single metric")
	can.drawCentredString(125,115,"to optimize based on their use case and F1 allows both recall")
	can.drawCentredString(125,105,"and precision to be optimized.")
	
	can.drawCentredString(125,75,"Area Under the Curve (AUC) refers to the Receiver Operating")
	can.drawCentredString(125,65,"Characteristic (ROC) curve. This curve shows the performance")
	can.drawCentredString(125,55,"of a classification model and the AUC shows the measure of")
	can.drawCentredString(125,45,"across all classification thresholds. AUC ranges from 0 to 1 ")
	can.drawCentredString(125,35,"with 1 being perfect perfomance and .5 being due to chance.")





	img0 = io.imread("Graphics/ROC.png")
	ar = (1.0 * img0.shape[1])/img0.shape[0]
	can.drawImage('Graphics/ROC.png', 250, 50, width=300*ar, height=300, mask='auto')

	can.showPage()

	######## PASTED

	for ev in eval:

		can.setStrokeColorRGB(62/256.0,55/256.0,92/256.0)
		can.setFillColorRGB(62/265.0,55/256.0,92/256.0)

		can.rect(0,460,700,40, fill=1)

		img0 = io.imread("Graphics/xd-logo-light-01.png")
		ar = (1.0 * img0.shape[1])/img0.shape[0]
		can.drawImage('Graphics/xd-logo-light-01.png', 5, 465, width=30*ar, height=30, mask='auto')

		can.setStrokeColorRGB(1,1,1)
		can.setFillColorRGB(1,1,1)
		can.setFont('Helvetica',20)
		can.drawCentredString(350, 470, ev)
		can.setStrokeColorRGB(0,0,0)
		can.setFillColorRGB(0,0,0)
		can.setFont('Helvetica',18)
		can.drawCentredString(250, 440, "Accuracy")
		can.drawCentredString(350, 440, "Precision")
		can.drawCentredString(450, 440, "Recall")
		can.drawCentredString(550, 440, "F1")
		can.drawCentredString(650, 440, "AUC")

		i = 1
		n = len(df[ev].unique())
		if n >= 10:
			n = 10

		accs = []
		pres = []
		recs = []
		aucs = []
		f1s = []
		sts = []

		for st in df[ev].unique():


			df_temp = df.loc[df[ev] == st]

			auc,ks,ks_score=get_roc(df_temp,data,val, plot = 0)

			acc,pre,rec,f1 = classification_metrics(df_temp,data,val)

			sts.append(st)

			accs.append(acc)

			pres.append(pre)

			recs.append(rec)

			f1s.append(f1)

			aucs.append(auc)

		for j in range(0,len(df[ev].unique())):

			can.setFont('Helvetica', 18)
			can.setFillColorRGB(0,0,0)
			can.drawString(20, i*(y-120)/n, sts[j])
			can.setFont('Helvetica', 15)

			percs = np.percentile(accs,[10,90])

			if accs[j] <= percs[0]:
				R = 1
				G = 0
				B = 0

			elif accs[j] <= percs[1]:
				R = 0
				G = 0
				B = 0
			else:
				R = .2
				G = 1
				B = 0

			can.setFillColorRGB(int(R),int(G),int(B))
			can.drawCentredString(250, i*(y-120)/n, "%.5f" % accs[j])

			percs = np.percentile(pres,[10,90])

			if pres[j] <= percs[0]:
				R = 1
				G = 0
				B = 0

			elif pres[j] <= percs[1]:
				R = 0
				G = 0
				B = 0
			else:
				R = .2
				G = 1
				B = 0


			can.setFillColorRGB(int(R),int(G),int(B))
			can.drawCentredString(350, i*(y-120)/n, "%.5f" % pres[j])

			percs = np.percentile(recs,[10,90])

			if recs[j] <= percs[0]:
				R = 1
				G = 0
				B = 0

			elif recs[j] <= percs[1]:
				R = 0
				G = 0
				B = 0
			else:
				R = .2
				G = 1
				B = 0


			can.setFillColorRGB(int(R),int(G),int(B))
			can.drawCentredString(450, i*(y-120)/n, "%.5f" % recs[j])

			percs = np.percentile(f1s,[10,90])

			if f1s[j] <= percs[0]:
				R = 1
				G = 0
				B = 0

			elif f1s[j] <= percs[1]:
				R = 0
				G = 0
				B = 0
			else:
				R = .2
				G = 1
				B = 0

			can.setFillColorRGB(int(R),int(G),int(B))
			can.drawCentredString(550, i*(y-120)/n, "%.5f" % f1s[j])

			percs = np.percentile(aucs,[10,90])

			if aucs[j] <= percs[0]:
				R = 1
				G = 0
				B = 0

			elif aucs[j] <= percs[1]:
				R = 0
				G = 0
				B = 0
			else:
				R = .2
				G = 1
				B = 0

			can.setFillColorRGB(int(R),int(G),int(B))
			can.drawCentredString(650, i*(y-120)/n, "%.5f" % aucs[j])

			i += 1

			if i%10 == 1 and j < len(df[ev].unique())-1:
				can.showPage()
				can.setStrokeColorRGB(62/256.0,55/256.0,92/256.0)
				can.setFillColorRGB(62/265.0,55/256.0,92/256.0)

				can.rect(0,460,700,40, fill=1)

				img0 = io.imread("Graphics/xd-logo-light-01.png")
				ar = (1.0 * img0.shape[1])/img0.shape[0]
				can.drawImage('Graphics/xd-logo-light-01.png', 5, 465, width=30*ar, height=30, mask='auto')

				can.setStrokeColorRGB(1,1,1)
				can.setFillColorRGB(1,1,1)

				can.setFont('Helvetica',20)
				can.drawCentredString(350, 470, ev)
				can.setStrokeColorRGB(0,0,0)
				can.setFillColorRGB(0,0,0) 
				can.setFont('Helvetica',18)
				can.drawCentredString(250, 440, "Accuracy")
				can.drawCentredString(350, 440, "Precision")
				can.drawCentredString(450, 440, "Recall")
				can.drawCentredString(550, 440, "F1")
				can.drawCentredString(650, 440, "AUC")

				i = 1

		can.showPage()



		######### PASTED




	can.save()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", help="Input file")
	parser.add_argument("--delimiter", help="Delimiter. Default: Comma")
	parser.add_argument("--data", help="Data Column, model output. Default: data")
	parser.add_argument("--val", help="Validation Column, expected output. Default: validation")
	parser.add_argument("--strata", nargs='+', help="Columns to be evaluated for bias. Default: none")
	parser.add_argument("--model", help="Model type: regression or classification. Default: regression")
	parser.add_argument("--title", help="Model title. Default: None")


	args = parser.parse_args()
	input = args.input
	data = args.data if args.data else 'data'
	val = args.val if args.val else 'validation'
	strata = args.strata if args.strata else None
	model = args.model if args.model else 'regression'
	title = args.title if args.title else None 
	delimiter = args.delimiter if args.delimiter else ','
	model_flag = 1


	pd_Data=pd.read_csv(input,delimiter=delimiter)


	if model.lower() == "regression":
		rsquared = rsq(pd_Data, data, val)
		mse_err = mse(pd_Data, data, val)
		rmse = sqrt(mse_err)
		mae_err = mae(pd_Data, data, val)

		create_regression_pdf(pd_Data,data,val,strata,title)

	elif model.lower() == "classification":
		create_classification_pdf(pd_Data,data,val,strata,title)	
	else:
		print("Model type not supported. Please select regression or classification")
		model_flag = 0

