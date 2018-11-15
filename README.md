[![DOW](https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/dow.jpg)](#)

# <a href="https://www.kaggle.com/c/two-sigma-financial-news"> Predicting Stocks Using the News</a>
A quant hedge fund, Two Sigma, is sponsoring a Kaggle competition to see if news headlines can be used to help predict stock price movements. The prediction deliverable is a 1 to -1 confidence-level (Up or Down) for the next 10 days for a ~4000 company subset of U.S Listed Companies.




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
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap20.png"></img>

Some of the features are self-explanatory, but here some of the interesting ones:

	The marketdata contains a variety of returns calculated over different timespans. All of the returns in this set of marketdata have these properties:

		* Returns are always calculated either open-to-open (from the opening time of one trading day to the open of another) or close-to-close (from the closing time of one trading day to the open of another).
		* Returns are either raw, meaning that the data is not adjusted against any benchmark, or market-residualized (Mktres), meaning that the movement of the market as a whole has been accounted for, leaving only movements inherent to the instrument.



Target variable is?

		returnsOpenNextMktres10(float64) - 10 day, market-residualized return.
Target 
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap2.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap3.png"></img>


More EDA - Exploring Close prices


closing price trend
<screencap17>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap17.png"></img>
Random stocks
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap15.png"></img>



### News Data
9,328,750 samples and 35 features.

<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap19.png"></img>

<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap18.png"></img>


First got a wordcloud to visualize the headlines

<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/wordcloud.png" width="600px" height="600px"></img>

Looks like normal financial news, What about this sentiment analysis?


	a. A quick foray into the Sentimental analysis via sense2vec
	The reuters sentiment analysis is proprietary "Thomson Reuters has delivered a unique capability in Eikon that takes feeds from both Twitter and StockTwits and weights and analyses sentiment using a proprietary methodology.""https://www.thomsonreuters.com/en/press-releases/2014/thomson-reuters-adds-unique-twitter-and-news-sentiment-analysis-to-thomson-reuters-eikon.html
	But we can guess they are using an sense-aware neural net that uses context embedding vectors as opposed to a bag-of-words approach




	https://www.groundai.com/project/sense2vec-a-fast-and-accurate-method-for-word-sense-disambiguation-in-neural-word-embeddings/

	 a novel approach which addresses these concerns by modeling multiple embeddings for each word based on supervised disambiguation

	 These “neural language models” embed a vocabulary into a smaller dimensional linear space that models “the probability function for word sequences, expressed in terms of these representations” [3]

	 Instead of predicting a token given surrounding tokens, this model predicts a word sense given surrounding senses.

<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap1.png"></img>


### Data Processing
2. Feature engineering and merging - Means vs sums

PRICE OUTLIERS
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap14.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap12.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap9.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/code/code1.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/screencap7.png"></img>




News happens every day, Trading only during the weekdays. How did I merge the days?

<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/code/code2.png"></img>

<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/merge1.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/merge2.png"></img>



### Modeling

### PCA Visualization (hoping to find cluster to build models around)
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/eda/pca.png"></img>



#### Start with a baseline company -- AAPL
3. LightGBM AAPL (gradient boosted forest)
	
	
	AAPL ONLY
	lgbmAAPL.png
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/lgbmAAPL.png"></img>

	Feature Importances
	lgbm_feat_import_AAPL.png
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/lgbm_feat_import_AAPL.png"></img>



4. Logistic Regression - Get some Coefficients for AAPL
	Results - ROC/AUC plot - Coefficients
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/log_roc.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/log_coefs.png"></img>

5. LSTM - AAPL
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/code/code3.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/code/lstmgraph.png"></img>


Results on AAPL Only
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/initial_LSTM_ROC_actual.png"></img>

	Embedded layer for assetcodes - A Panacea?
	Results for multiple companies
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/embedding/3embed_aapl.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/embedding/3embed_advance.png"></img>


	Results on Sine Waves

<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/sine_waves/sin_multiply_merge_solotrain.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/sine_waves/sin_multiple.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/sine_waves/sinlong_multiple.png"></img>



	5-day window:
	noembed_aapltrain_5day.png
	noembedALLSTATE_aapltrain_5day.png
	noembedADVANCE_aapltrain_5day.png
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/noembed_aapltrain_5day.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/noembedADVANCE_aapltrain_5day.png"></img>
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/noembedALLSTATE_aapltrain_5day.png"></img>



### Results
	Final Model : Trained on AAPL only 
	I didn't have a problem with overfitting, so more neurons and more layers.
	Architecture layer graphic (tensorboard?)
	ROC/AUC on local machine for whole data set (LGBM and LSTM)
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/init_pred/lgbm.png"></img>

<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/final_pred/RocLstmAll.png"></img>

Current Kaggle Score
<img src="https://github.com/joeshull/twosigma_kaggle/blob/master/graphics/final_pred/kaggle.png" width="1200px" height="300px"></img>



## Future Work
Going forward, I would like to build a "boosted" version of the LSTM. A model that trains on the companies that give the last model a logloss above the .5 threshold. With the current limitations on the Kaggle competition, build a comprehensive, robust model will be difficult

## AUTHOR
[auth]: #author 
You can follow me on [twitter](https://twitter.com/joeyshull) or just [email](mailto:joseph.shull@gmail.com) me.

## ACKNOWLEDGMENTS
[acc]: acknowledgments

List of people that I would like to thank:

- Jamie Sloat for her endless support.
- Michael Dyer for being a dope keras architect.
- Frank Burkholder for being a fountain of great ideas when I get stuck.


Copyright © 2018 Joe Shull


