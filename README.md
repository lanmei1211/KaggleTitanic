# KaggleTitanic

*Repo for my first attempts at the Titanic challenge on Kaggle*

I will start from the strategies employed off of my other repo playing with the titanic dataset found at 
https://github.com/nb137/EasyML
which also lists my refernces and influences for those code attempts

###Submission 5 is my highest scoring submission, the strategies of 5 were:  
Use class, sex, age SibSp, parch, fare, embarked.  
Pull out titles from names, map them to Officer, royalty, regular, master  
predict missing ages using Lasso  
Predict survive using SVC linear  
The final score was _0.78468_  

###The order of scores and submissions was as follows:  
5_AgeAndTitle - 0.78468  
6_FewerParams_alltitles - 0.77990  
6_FewerParams- 0.77033  
4_AgePredict - 0.76555  
AgePredict-lasso - 0.76555  
1_FirstTry - 0.76555  
7_GradBoost - 0.76555  
8_dropParamsSVC - 0.76076  
3_RandomForest - 0.72248  


###The description of models in order of attempt is as follows:  
1 - Initial Attempt 
Import data, drop Name, Ticket, Cabin  
Fill nan ages with mean of ages  
map sex to male:0 female:1  
Map embarked to 1,2,3 and then move them into dummy variable columns  
Predict using SVC(kernel='linear')  

2 - KNeighbors  
Same as above, but try KNeighborsClassifier instead of SVC  

3 - Random Forest  
Same as above, use DecisionTreeClassifier as the model  

4 - NaN predict  
First attempt at predicting ages. Do preparation as in 1, but do not set ages to mean. Then predict ages usingLasso model.  
Fit with SVC classifier. This does not change the final score.  

5 - Age and Title  
Same as above, but use title as prediction input.  
Parse names and map to a title of the passenger.  
Fit with SVC linear  

6 - Drop Some Params  
Same as above, but now we use RandomForestClassifier and SelectFromModel to select the important parameters to use from the RFC fit.  
Then fit model with RFC. Did not end up as good.  
Here I also put the title column in before predicting ages to help with age prediction.  

7 - GradBoost  
Same as above, but final model is GradientBoostingClassifier, attempt with both the reduced parameters and the non-reduced. Reduced worked better but wasn't as good as the first model.  

8 - Drop Params SVC  
Because we saw an increase from dropping parameters, but random forest was previously not great, I try the same strategy but with SVC, which was a previously best model.  
This did not improve the fit.  
