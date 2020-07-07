# Stock prediction

## Description
The goal of this project is to predict the value of a stock for the next day and use the data to simulate a trading algorithm. I chose to use the average of the High and Low values for the stock to train a LSTM model.

## Results
Here we can see an example of the next day predictions for 3 stocks.
Parameters : using the last 20 days for the predictions, 25 epochs of training, 200 hidden neurons for the LSTM

![Microsoft](microsoft_20in_25ep_200h.png)
![Apple](apple_20in_25ep_200h.png)
![Amazon](amazon_20in_25ep_200h.png)

Microsoft RMSE : 5.138481244485672
Apple RMSE : 10.82418362089971
Amazon RMSE : 82.2145559000393

We see that the model can predict the next day's mean stock value pretty well for some stocks (i.e. low error), but what is more important to us is that we don't predict it to go up when it goes down.

If we only take the binary result of up or down from this model, we get these results :
Amazon : 8 wrong predictions out of 20
Apple : 8 wrong predictions out of 20
Microsoft : 7 wrong predictions out of 20

## Further work
- Use the differenced dataframe to train the network, which may lead to a better result in the binary task of predicting "up" or "down" for the next day and would eliminate the need for smoothing windows
- Try building a binary classification model (i.e. only predicting "up" or "down")
- Try using different stocks
- Tune hyperparameters (hidden layer size, input sequence length, training epochs, learning rate, etc.)
- Try a different LSTM architecture (for example 2 hidden layers)
- Clean code
- Find a way to figure out how sure the model is of its prediction
- Integrate other data to train the model (for example a Google Trend search for the company using pytrends)