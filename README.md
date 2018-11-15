[![DOW](https://github.com/joeshull/twosigma_kaggle/tree/master/graphics/dow.jpg)](#)

# Predicting Stocks Using the News
A quant hedge fund, Two Sigma, is sponsoring a Kaggle competition to see if news headlines can be used to help predict stock price movements. The prediction deliverable is a 1 to -1 confidence-level (Up or Down) for the next 10 days for a ~4000 company subset of U.S Listed Companies.

https://www.kaggle.com/c/two-sigma-financial-news


## Outline

**[Data Exploration](#data-exploration):** 

**[Data Processing](#data-processing):** 

**[Modeling](#image-eda):** 
	## Baseline - Logistic and GBM
	## LSTM
		Embedding - Didn't work (Sine Waves)
		Trained on AAPL only, ROC/AUC and Posted Kaggle Score

**[Results](#results):**

ROC/AUC of AAPL trained model on entire hold-out set. KAGGLE score.

**[Future Work](#Future Work):** 


## Data Exploration

The training data are in two dataframes and no other data is allowed:

	* Market Data (2007 to Present) - Provided by Intrinio, contains financial market information such as opening price, closing price, trading volume, calculated returns, etc.

	* News Data (2007 to Present) - Provided by Reuters, contains information about news articles/alerts published about assets, such as article details, sentiment, and other commentary.


###	Market data
4,072,956 rows and 16 features in the market dataset.
<img src="screencap20" height="300px"></img>

Some of the features are self-explanatory, but here some of the interesting ones:

	The marketdata contains a variety of returns calculated over different timespans. All of the returns in this set of marketdata have these properties:

		* Returns are always calculated either open-to-open (from the opening time of one trading day to the open of another) or close-to-close (from the closing time of one trading day to the open of another).
		* Returns are either raw, meaning that the data is not adjusted against any benchmark, or market-residualized (Mktres), meaning that the movement of the market as a whole has been accounted for, leaving only movements inherent to the instrument.



Target variable is?

		returnsOpenNextMktres10(float64) - 10 day, market-residualized return.

		https://www.kaggle.com/marketneutral/eda-what-does-mktres-mean

	Show head:



closing price trend
<screencap17>

Random stocks
<screencap15>



### News Data
9,328,750 samples and 35 features.

[Screencap19]

[Screencap18]


First got a wordcloud to visualize the headlines

<img src="wordcloud.png" width="600px" height="300px"></img>

Looks like normal financial news, What about this sentiment analysis?


	a. A quick foray into the Sentimental analysis via sense2vec
	The reuters sentiment analysis is proprietary "Thomson Reuters has delivered a unique capability in Eikon that takes feeds from both Twitter and StockTwits and weights and analyses sentiment using a proprietary methodology.""https://www.thomsonreuters.com/en/press-releases/2014/thomson-reuters-adds-unique-twitter-and-news-sentiment-analysis-to-thomson-reuters-eikon.html
	But we can guess they are using an sense-aware neural net that uses context embedding vectors as opposed to a bag-of-words approach




	https://www.groundai.com/project/sense2vec-a-fast-and-accurate-method-for-word-sense-disambiguation-in-neural-word-embeddings/

	 a novel approach which addresses these concerns by modeling multiple embeddings for each word based on supervised disambiguation

	 These “neural language models” embed a vocabulary into a smaller dimensional linear space that models “the probability function for word sequences, expressed in terms of these representations” [3]

	 Instead of predicting a token given surrounding tokens, this model predicts a word sense given surrounding senses.

[screencap1]

### Data Processing
2. Feature engineering and merging - Means vs sums

PRICE OUTLIERS
<screencap14>
<screencap13>
<screencap12>
<screencap11>
<screencap10>
<screencap9>
<screencap7>

Target 
screencap2
screencap3

	Graphic of AAPL before - single stock price for the day, all news articles
	Graphic of AAPL after - single stock with aggregated news data from the last 24 hrs
<merge 2>
<merge 1>
<code for merging on trading day>



### Modeling

### PCA Visualization (hoping to find cluster to build models around)
pca.png



#### Start with a baseline company -- AAPL
3. LightGBM AAPL (gradient boosted forest)
	
	
	AAPL ONLY
	lgbmAAPL.png
	Feature Importances
	lgbm_feat_import_AAPL.png



4. Logistic Regression - Get some Coefficients for AAPL
	Results - ROC/AUC plot - Coefficients
	log_roc.png
	log_coefs.png

5. LSTM 
	model build code
	Results on AAPL Only
	initial_LSTM_ROC_actual.png

	Embedded layer for assetcodes
	Results for multiple companies,
		3embed_aapl.png
		3embed_advance.png
		3embed_allstate.png
	Results on Sine Waves
		sin_multiply_merge_solotrain.png
		sin_multiple.png
		sinlong_multiple.png


	5-day window:
	noembed_aapltrain_5day.png
	noembedALLSTATE_aapltrain_5day.png
	noembedADVANCE_aapltrain_5day.png



### Results
	Final Model : Trained on AAPL only 
	I didn't have a problem with overfitting, so more neurons and more layers.
	Architecture layer graphic (tensorboard?)
	ROC/AUC on local machine for whole data set (LGBM and LSTM)
	RocLstmAll.png
	lgbm.png

	kaggle.png
	

6. Future Work

	






