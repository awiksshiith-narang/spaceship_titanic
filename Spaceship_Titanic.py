import pandas as p
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb

#Gathering data for training and testing the model:
train_and_test_data = p.read_csv( "C:/Users/AWIKSSHIITH/OneDrive/Desktop/train_and_test_data.csv" )
train_and_test_data.dropna( inplace = True )
y = train_and_test_data[ 'Transported' ].values #Values in 'Transported' column are target values.
x = train_and_test_data.drop( [ 'Transported', 'Name' ], axis = 1 ).values #Values which are not in 'Transported' column are input data. Name doesn't affect the target value.
label_encoder = LabelEncoder() #Assigns a numerical value to the string data.
x[ :, 0 ] = label_encoder.fit_transform( x[ :, 0 ] )
x[ :, 1 ] = label_encoder.fit_transform( x[ :, 1 ] )
x[ :, 2 ] = label_encoder.fit_transform( x[ :, 2 ] )
x[ :, 3 ] = label_encoder.fit_transform( x[ :, 3 ] )
x[ :, 4 ] = label_encoder.fit_transform( x[ :, 4 ] )
x[ :, 6 ] = label_encoder.fit_transform( x[ :, 6 ] )
y = label_encoder.fit_transform( y )

#Splitting the data into train and test sets:
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 0 )

##Training different models:
#Logistic Regression:
log_reg = LogisticRegression()
log_reg.fit( x_train, y_train )
print( 'Test accuracy of Logistic Regression is {}%.'.format( log_reg.score( x_test, y_test ) * 100 ) )
#K-nearest neighbors:
scorelist = []
for i in range( 1, 20 ):
    knn = KNeighborsClassifier( n_neighbors = i )
    knn.fit( x_train, y_train )
    scorelist.append( knn.score( x_test, y_test ) * 100 )
plt.plot( range( 1, 20 ), scorelist )
plt.title( 'Plot of %Accuracy of KNN with respect to no, of neighbors' )
plt.xlabel( 'no, of neighbors' )
plt.ylabel( '%Accuracy' )
plt.show()
print( 'Test accuracy of K-nearest neighbors is {} for no, of neighbors {}.'.format( max( scorelist ), scorelist.index( max( scorelist ) ) + 1 ) )
#Support Vector Machine algorithm:
svm = SVC( random_state = 1 )
svm.fit( x_train, y_train )
print( 'Test accuracy of Support Vector Machine is {}%.'.format( svm.score( x_test, y_test ) * 100 ) )
#Naive Bayes algorithm:
nb = GaussianNB()
nb.fit( x_train, y_train )
print( 'Test accuracy of Naive Bayes is {}%.'.format( nb.score( x_test, y_test ) * 100 ) )
#Desicion Tree algorithm:
dt = DecisionTreeClassifier()
dt.fit( x_train, y_train )
print( 'Test accuracy of Desicion Tree is {}%.'.format( dt.score( x_test, y_test ) * 100 ) )
#Random Forest Algorithm:
rf = RandomForestClassifier( n_estimators = 1000, random_state = 1 )
rf.fit( x_train, y_train )
print( 'Test accuracy of Random Forest is {}%.'.format( rf.score( x_test, y_test ) * 100 ) )
#Comparing above models:
methods = [ 'Logistic Regression', 'KNN', 'SVM', 'Naive Bayes', 'Desicion Tree', 'Random Forest' ]
scores = [ log_reg.score( x_test, y_test ) * 100, max( scorelist ), svm.score( x_test, y_test ) * 100, nb.score( x_test, y_test ) * 100, dt.score( x_test, y_test ) * 100, rf.score( x_test, y_test ) * 100 ]
colors = [ 'purple', 'blue', 'green', 'yellow', 'orange', 'red' ]
sb.set_style( 'whitegrid' )
plt.figure( figsize = ( 16, 5 ) )
sb.barplot( x = methods, y = scores, palette = colors )
plt.title( 'Comparing different models' )
plt.xlabel( 'Models' )
plt.ylabel( '%Accuracies' )
plt.show()

#Choosing the right model and making predictions:
print( 'As seen above, Logistic Regression has the highest accuracy.' )
print( 'We are going to use it for our predictions.' )
data_for_predictions = p.read_csv( "C:/Users/AWIKSSHIITH/OneDrive/Desktop/data_for_predictions.csv" )
data_for_predictions.dropna( inplace = True )
x_preds = data_for_predictions.drop( [ 'Name' ], axis = 1 )
x_preds[ 'PassengerId' ] = label_encoder.fit_transform( x_preds[ 'PassengerId' ] )
x_preds[ 'HomePlanet' ] = label_encoder.fit_transform( x_preds[ 'HomePlanet' ] )
x_preds[ 'CryoSleep' ] = label_encoder.fit_transform( x_preds[ 'CryoSleep' ] )
x_preds[ 'Cabin' ] = label_encoder.fit_transform( x_preds[ 'Cabin' ] )
x_preds[ 'Destination' ] = label_encoder.fit_transform( x_preds[ 'Destination' ] )
x_preds[ 'VIP' ] = label_encoder.fit_transform( x_preds[ 'VIP' ] )
y_preds = log_reg.predict( x_preds )
x_preds[ 'Prediction' ] = y_preds
result = p.merge( data_for_predictions, x_preds[ 'Prediction' ], left_index = True, right_index = True )
print( result )