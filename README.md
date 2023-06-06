# RevenuePrediction

## For the Revenue prediction task:
- The model achieved an accuracy of 90% on the test set.
- The precision for the 'False' class (No revenue) was 0.91, indicating that 91% of the predicted 'False' instances were correctly classified.
- The recall for the 'False' class was 0.97, indicating that 97% of the actual 'False' instances were correctly classified.
- The precision for the 'True' class (Revenue) was 0.76, indicating that 76% of the predicted 'True' instances were correctly classified.
- The recall for the 'True' class was 0.55, indicating that 55% of the actual 'True' instances were correctly classified.

## For the Weekend prediction task:
- The model achieved a perfect accuracy of 100% on the test set.
- Both precision and recall for both classes ('False' and 'True') were 1.0, indicating perfect classification performance.

## For the Informational_Duration regression task:
- The Mean Absolute Error (MAE) was 0.2679, which represents the average absolute difference between the predicted and actual values of the Informational_Duration feature. A lower MAE indicates a better fit of the regression model.

Overall, the code successfully trained and evaluated the ensemble learning models for the Revenue and Weekend prediction tasks, achieving high accuracy and demonstrating excellent classification performance. Additionally, it successfully applied a Random Forest Regressor to predict the Informational_Duration, providing a measure of the model's accuracy through the MAE.
