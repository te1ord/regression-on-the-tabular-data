from src.data.make_dataset import AnonymizedDataset
from src.model.script import PolynomialRegression
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np

train_dataset = AnonymizedDataset('/Users/ivanbashtovoy/Documents/quantum/regression-on-the-tabular-data/data/train_preprocessed/train_data.csv')
train_loader = DataLoader(train_dataset, batch_size=64  , shuffle=False)
test_dataset = AnonymizedDataset('/Users/ivanbashtovoy/Documents/quantum/regression-on-the-tabular-data/data/train_preprocessed/test_data.csv')
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


net = PolynomialRegression(1, 50, 1)
net.train_model(train_loader, 1000, 0.001)


predictions_train, truth_values_train = net.predict(train_loader)
print('TRAIN RMSE', mean_squared_error(predictions_train, truth_values_train, squared=True))


predictions_test, truth_values_test = net.predict(test_loader)
print('TEST RMSE', mean_squared_error(predictions_test, truth_values_test, squared=False))

predict_dataset = AnonymizedDataset('/Users/ivanbashtovoy/Documents/quantum/regression-on-the-tabular-data/data/predict_preprocessed/predict_data.csv')
predict_dataloader = DataLoader(predict_dataset, batch_size=64)

predictions, _ = net.predict(predict_dataloader)


predictions.tofile('/Users/ivanbashtovoy/Documents/quantum/regression-on-the-tabular-data/predictions.csv', sep = ',')